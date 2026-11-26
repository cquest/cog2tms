import asyncio
import math
from collections import OrderedDict
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from . import config, crs
from . import __version__ as version
from .cogpool import ReaderPool
from .crs import UnsupportedCRS
from .empty import empty_tile
from .geotiff import CogError, compression_name, synth_floor_px
from .httprange import (CogChanged, CogFetch, CogNotFound, CogRedirect,
                        forget, last_modified, revalidate)
from .security import check_url
from .tiles import EmptyTile, MissingRescale, web_bbox_lonlat

app = FastAPI(
    title="cog2tms", version=version,
    description="Cloud-Optimized GeoTIFF in, EPSG:3857 XYZ (OSM numbering, y-down) "
                "tiles out: pixels are read from the remote COG by HTTP byte-range "
                "requests only, never the whole file. The machine-readable contract "
                "of a single COG is its TileJSON, `/c2t.tms.json?u=…`.",
    openapi_tags=[{"name": "tiles", "description": "rendered tile images"},
                  {"name": "metadata", "description": "footprint and TileJSON of a COG"},
                  {"name": "ops", "description": "liveness probe and robots policy"}])
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "HEAD"])


class _HeadAsGet:
    """HEAD means the same route, minus the body. FastAPI's @app.get() is GET-only
    (a bare Starlette Route would have added HEAD), and putting HEAD into
    route.methods instead documents a HEAD twin per route with a duplicate
    operationId - so the router only ever sees a GET.

    Dropping the body is ours to do: h11 does it for us, uvicorn's httptools path
    (the default of uvicorn[standard], so the Docker image) does not - it writes
    the whole tile after the headers of a HEAD, which desynchronises a keep-alive
    connection. Content-Length stays as the GET would have declared it, and the
    answer is still produced in full: the tile is rendered and cached, so the GET
    that follows a HEAD fetches nothing upstream."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        head = scope["method"] == "HEAD"
        if head:
            # a copy: uvicorn keeps its own "HEAD" in its scope, so its access log
            # does not report a GET
            scope = dict(scope, method="GET")

        async def forward(message):
            if message["type"] == "http.response.start":
                message = _honest_allow(message)
            elif head:
                if message.get("more_body"):
                    return  # swallowed, but the app keeps going: same cache entry
                message = {"type": "http.response.body", "body": b""}
            await send(message)

        await self.app(scope, receive, forward)


def _honest_allow(message: dict) -> dict:
    """A 405 must not deny HEAD in its Allow header when HEAD works: Starlette
    builds that header from route.methods, and HEAD is deliberately not declared
    there (see _HeadAsGet). Only ever touches a GET-only answer."""
    if message["status"] != 405:
        return message
    headers = list(message["headers"])
    for i, (name, value) in enumerate(headers):
        if name.lower() == b"allow" and b"get" in value.lower() and b"head" not in value.lower():
            headers[i] = (name, value + b", HEAD")
    return {**message, "headers": headers}


app.add_middleware(_HeadAsGet)


@app.middleware("http")
async def _no_index(request: Request, call_next):
    """robots.txt alone would not cover tile urls (they are images): tag every
    response so well-behaved crawlers keep out of the whole service."""
    resp = await call_next(request)
    resp.headers["X-Robots-Tag"] = "nofollow, noimageindex"
    return resp


pool = ReaderPool(config.POOL_SIZE, config.READER_TTL, config.MAX_CONCURRENT_READS)

_STATIC = Path(__file__).parent / "static"  # web UI: maplibre.html + vendored maplibre-gl
app.mount("/static", StaticFiles(directory=_STATIC), name="static")

MIME = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp"}
_TMS_SIZE = 256  # TileJSON zooms assume a 256 px grid: the tile template pins that size
CACHE = {"Cache-Control": "public, max-age=86400"}
_tiles_cache: OrderedDict[str, bytes] = OrderedDict()
_inflight: dict[str, asyncio.Lock] = {}


def _err(code: int, why: str) -> dict:
    """One error of the OpenAPI schema: every failure path raises HTTPException,
    so the body is always {"detail": "..."}."""
    return {code: {"description": why,
                   "content": {"application/json": {"schema": {
                       "type": "object", "properties": {"detail": {"type": "string"}}}}}}}


_ERRORS = {**_err(400, "zoom/x/y outside the matrix, bad format or size, refused url or "
                       "scheme, bad bidx/rescale/blackmax, signed or wide/float data without "
                       "rescale, unsupported CRS, refused or looping redirect chain"),
           **_err(404, "COG unreachable, not a GeoTIFF, rotated raster, unreadable samples"),
           **_err(502, "upstream error")}


def clear_caches():
    """Drop cached encoded tiles (tests)."""
    _tiles_cache.clear()
    _inflight.clear()


def _purge(u: str):
    """The remote COG was replaced under the same URL: everything derived from
    the previous version (reader, decoded blocks, encoded tiles) must go."""
    pool.invalidate(u)
    forget(u)
    prefix = u + "|"
    for key in [k for k in _tiles_cache if k.startswith(prefix)]:
        del _tiles_cache[key]


def _parse_bidx(bidx: str | None) -> list[int] | None:
    if not bidx:
        return None
    try:
        idx = [int(v) for v in bidx.split(",")]
    except ValueError:
        raise HTTPException(400, "bad bidx")
    if not idx or any(i < 1 for i in idx):
        raise HTTPException(400, "bad bidx")
    return idx


def _parse_rescale(rescale: str | None) -> tuple[float, float] | None:
    if not rescale:
        return None
    try:
        lo, hi = (float(v) for v in rescale.split(","))
    except ValueError:
        raise HTTPException(400, "bad rescale")
    if lo >= hi:
        raise HTTPException(400, "bad rescale")
    return lo, hi


def _parse_blackmax(blackmax: str | None) -> int | None:
    if not blackmax:
        v = config.BLACKMAX
    else:
        try:
            v = int(blackmax)
        except ValueError:
            raise HTTPException(400, "bad blackmax")
    if v < 0:  # -1 (or any negative) explicitly disables the black -> transparent pass
        return None
    if v > 255:
        raise HTTPException(400, "bad blackmax")
    return v


def _guard(u: str):
    """URL policy (scheme/SSRF/allowlist); lru_cached, so it is cheap enough to
    run on every request, including one answered from the encoded-tile cache."""
    try:
        check_url(u)
    except ValueError as e:
        raise HTTPException(400, str(e))


async def _open(u: str):
    _guard(u)
    for attempt in (0, 1):
        try:
            return await pool.get(u)
        except CogChanged:
            if attempt:
                raise HTTPException(503, "COG keeps changing")
            _purge(u)  # file replaced: evict, then read the new version
        except CogRedirect as e:
            raise HTTPException(400, f"unusable COG URL: {e}")
        except CogNotFound:
            raise HTTPException(404, "COG not found")
        except CogFetch as e:
            raise HTTPException(502, f"upstream error: {e}")
        except CogError as e:
            raise HTTPException(404, f"not a readable GeoTIFF: {e}")
        except UnsupportedCRS as e:
            raise HTTPException(400, str(e))


@app.get("/c2t/{z}/{x}/{y}.geojson", tags=["metadata"], summary="COG footprint ring",
         responses={200: {"description": "GeoJSON Feature polygon, lon/lat degrees "
                                         "(z/x/y are ignored: always the footprint)",
                          "content": {"application/json": {}}}, **_ERRORS})
async def footprint(z: int, x: int, y: int,
                    u: str = Query(..., description="COG url (http/https)")):
    """Remote COG footprint as a lon/lat geojson polygon (debug helper)."""
    cog = await _open(u)
    ring = cog.footprint_lonlat()
    return JSONResponse({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]},
                         "properties": {"crs": cog.crs_name, "url": u}})


@app.get("/c2t/{z}/{x}/{y}", tags=["tiles"], summary="One tile of the COG",
         response_class=Response,
         responses={200: {"description": "image in the negotiated format; a tile fully "
                                         "outside the COG is 200 + empty (transparent "
                                         "png/webp, solid jpeg for jpeg)",
                          "content": {m: {} for m in MIME.values()}}, **_ERRORS})
async def tile(z: int, x: int, y: int,
               u: str = Query(..., description="COG url (http/https) on an allowed host"),
               format: str = Query("png", description="png | jpeg | webp | auto - auto trusts "
                                                      "only the Accept header: webp iff image/webp is "
                                                      "listed explicitly with q>0, never on wildcards"),
               quality: int = Query(80, description="jpeg/webp quality, clamped to 1..95"),
               bidx: str | None = Query(None, description="1-based bands to render, e.g. 1,2,3"),
               rescale: str | None = Query(None, description="min,max in source units; required "
                                                             "for signed, 32-bit or float data"),
               blackmax: str | None = Query(None, description="pixels whose brightest band is at or "
                                                              "below this become no-coverage; default "
                                                              "20, -1 disables"),
               size: int = Query(512, description="tile edge in pixels, "
                                                  "128..1024, multiple of 16"),
               request: Request = None):
    """One tile of the COG: a square image in the requested format and size,
    addressed on the OSM/XYZ grid (z/x/y, y points down - not TMS)."""
    if not 0 <= z <= config.MAX_ZOOM:
        raise HTTPException(400, "bad zoom")
    n = 1 << z
    if not (0 <= x < n and 0 <= y < n):
        raise HTTPException(400, "tile index outside matrix")
    if format not in MIME and format != "auto":
        raise HTTPException(400, "unsupported format")
    if not 128 <= size <= 1024 or size % 16:
        raise HTTPException(400, "bad size, must be between 128 and 1024 and a multiple of 16")  # 16: jpeg macroblock / tiff block grid
    quality = max(1, min(95, quality))
    # format=auto depends only on the Accept header: webp if image/webp is
    # explicitly listed with q>0, else png. Wildcards (image/*, */*: JOSM,
    # plain HTTP clients) never select webp - JOSM cannot decode it.
    if format == "auto":
        format = "webp" if _accepts_webp(request.headers.get("accept", "")) else "png"
    key = f"{u}|{z}|{x}|{y}|{format}|{quality}|{size}|{bidx}|{rescale}|{blackmax}"
    _guard(u)  # also enforced on cache hits: a host taken out of the allowlist
    # must stop being served, even from the encoded-tile cache
    # at most one 1-byte If-Range probe per VALIDATE_TTL and per COG: it drops
    # cached tiles and the open reader when the file was replaced under the URL
    await _revalidate(u)
    blob = _tiles_cache.get(key)
    if blob is None:
        # single-flight: concurrent requests for the same tile render it once
        lock = _inflight.setdefault(key, asyncio.Lock())
        try:
            async with lock:
                blob = _tiles_cache.get(key)
                if blob is None:
                    blob = await _render_tile(u, z, x, y, format, quality, size,
                                              bidx, rescale, blackmax)
                    if config.TILE_CACHE:
                        _tiles_cache[key] = blob
                        while len(_tiles_cache) > config.TILE_CACHE:
                            _tiles_cache.popitem(last=False)
        finally:
            _inflight.pop(key, None)
    return Response(blob, media_type=MIME[format], headers=CACHE)


async def _revalidate(u: str):
    """Never raises: a replaced COG evicts its own caches, upstream trouble is
    ignored so already-encoded tiles keep being served (stale-while-error)."""
    try:
        await asyncio.to_thread(revalidate, u)
    except CogChanged:
        _purge(u)
    except (CogNotFound, CogFetch):
        pass


async def _render_tile(u: str, z: int, x: int, y: int, format: str, quality: int,
                       size: int, bidx: str | None, rescale: str | None,
                       blackmax: str | None) -> bytes:
    """Render, reopening the COG once if a read reveals a replaced file."""
    try:
        return await _render(u, z, x, y, format, quality, size, bidx, rescale, blackmax)
    except CogChanged:
        _purge(u)
        try:
            return await _render(u, z, x, y, format, quality, size, bidx, rescale, blackmax)
        except CogChanged as e:
            raise HTTPException(503, f"COG keeps changing: {e}")


async def _render(u: str, z: int, x: int, y: int, format: str, quality: int,
                  size: int, bidx: str | None, rescale: str | None,
                  blackmax: str | None) -> bytes:
    cog = await _open(u)
    try:
        rgb, valid = await pool.render(cog, z, x, y, size,
                                       _parse_bidx(bidx), _parse_rescale(rescale),
                                       _parse_blackmax(blackmax))
    except EmptyTile:
        return empty_tile(format, size, config.EMPTY_TILE_BG)
    except CogChanged:
        raise  # caller reopens the COG on the new version
    except CogRedirect as e:
        raise HTTPException(400, f"unusable COG URL: {e}")
    except (UnsupportedCRS, MissingRescale) as e:
        raise HTTPException(400, str(e))
    except CogNotFound:
        raise HTTPException(502, "COG vanished")
    except CogFetch as e:
        raise HTTPException(502, f"upstream error: {e}")
    except CogError as e:  # the file opened, its pixels did not: refuse with 404
        raise HTTPException(404, f"unreadable COG: {e}")
    return _encode(rgb, valid, format, quality)


def _accepts_webp(header: str) -> bool:
    """Strict token match, q=0 skipped: image/* and */* deliberately do NOT
    count - JOSM and plain HTTP clients send them but cannot decode webp."""
    for tok in header.split(","):
        parts = tok.strip().split(";")
        if parts[0].lower() == "image/webp":
            q = next((p.strip()[2:] for p in parts[1:] if p.strip().startswith("q=")), "1")
            try:
                if float(q) > 0:
                    return True
            except ValueError:
                pass  # malformed q: the media type still matched
    return False


def _encode(rgb: np.ndarray, valid: np.ndarray, fmt: str, quality: int) -> bytes:
    buf = BytesIO()
    if fmt == "jpeg":  # no alpha: invalid pixels get the configured solid fill
        bg = np.array(config.EMPTY_TILE_BG, dtype=np.uint8)
        Image.fromarray(np.where(valid[..., None], rgb, bg)).save(buf, "JPEG", quality=quality)
    else:
        rgba = np.ascontiguousarray(np.dstack([rgb, np.where(valid, 255, 0).astype(np.uint8)]))
        if fmt == "png":
            Image.fromarray(rgba).save(buf, "PNG", compress_level=config.PNG_COMPRESS)
        else:
            # exact=True keeps RGB where alpha=0 (Pillow would blend a background);
            # method 0: 4 ms, method 4 buys 1 % bytes for 3x the time
            kw = dict(lossless=True, exact=True) if config.WEBP_LOSSLESS \
                else dict(quality=quality, method=0, exact=True)
            Image.fromarray(rgba).save(buf, "WEBP", **kw)
    return buf.getvalue()


@app.get("/c2t.tms.json", tags=["metadata"], summary="TileJSON of the COG",
         responses={200: {"description": "TileJSON 2.2.0: lon/lat bbox and the zoom range "
                                         "where the COG holds its native resolution; "
                                         "minzoom sits lower when the COG's missing "
                                         "overviews can be derived (cog.synth)",
                          "content": {"application/json": {}}}, **_ERRORS})
async def tilejson(u: str = Query(..., description="COG url (http/https)")):
    """Returns the TileJSON describing the remote transformed COG."""
    cog = await _open(u)
    bbox = web_bbox_lonlat(cog.web_bbox())
    gsd = cog.psx * cog.unit_m
    last = cog.own_last  # the file's pyramid: a pooled reader may already derive more
    # the derived pyramid (CogFile.synth_to) halves the COG's smallest level until
    # one is at most SYNTH_MIN_PX wide, but only while that level is small enough
    # for the single contiguous read to stay cheap. minzoom follows the level the
    # service really renders from, derived levels included: past it the warp
    # upscales, exactly as it does beyond any pyramid
    floor = (synth_floor_px(last.width) if config.SYNTH_MAX_PX
             and last.width * last.height <= config.SYNTH_MAX_PX else last.width)
    synth = floor < last.width
    coarse_gsd = gsd * cog.levels[0].width / max(floor, 1)
    res0 = 2 * math.pi * crs.R_WEB / 256.0
    # provenance of the source file, free of charge: every value here is what
    # opening the COG already learned (no extra upstream byte)
    info = {"size": cog.reader.size(),
            "compression": compression_name(cog.levels[0].compression)}
    if synth:  # why minzoom sits lower than the file's own pyramid would give
        info["synth"] = True
    lm = last_modified(u)  # direct sightings only: behind a redirect nothing is anchored
    if lm:
        info["last_modified"] = lm
    return JSONResponse({
        "tilejson": "2.2.0", "name": "cog2tms", "scheme": "xyz",
        # minzoom/maxzoom are computed on a 256 px grid: pin the size so the
        # advertised zooms mean what TileJSON says
        "tiles": [f"/c2t/{{z}}/{{x}}/{{y}}?u={u}&size={_TMS_SIZE}"],
        "bbox": [round(v, 6) for v in bbox],
        "minzoom": max(0, min(config.MAX_ZOOM, math.floor(math.log2(res0 / max(coarse_gsd, 1e-9))))),
        "maxzoom": max(0, min(config.MAX_ZOOM, math.ceil(math.log2(res0 / gsd)))),
        "cog": info,
    })


@app.get("/", include_in_schema=False)  # the web UI is a viewer, not part of the api
async def index():
    return FileResponse(_STATIC / "maplibre.html", media_type="text/html")


@app.get("/healthz", tags=["ops"], summary="liveness probe")
async def healthz():
    return {"status": "ok"}


@app.get("/robots.txt", tags=["ops"], response_class=FileResponse,
         summary="robots policy for this service",
         responses={200: {"description": "disallows /c2t", "content": {"text/plain": {}}}})
async def robots():
    return FileResponse(_STATIC / "robots.txt", media_type="text/plain")
