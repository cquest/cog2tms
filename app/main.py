import asyncio
import math
from collections import OrderedDict
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from . import config, crs
from . import __version__ as version
from .cogpool import ReaderPool
from .crs import UnsupportedCRS
from .empty import empty_tile
from .geotiff import CogError
from .httprange import (CogChanged, CogFetch, CogNotFound, CogRedirect, forget,
                        revalidate)
from .security import check_url
from .tiles import EmptyTile, MissingRescale, web_bbox_lonlat

app = FastAPI(title="cog2tms", version=version)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"])


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
CACHE = {"Cache-Control": "public, max-age=86400"}
_tiles_cache: OrderedDict[str, bytes] = OrderedDict()
_inflight: dict[str, asyncio.Lock] = {}


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


async def _open(u: str):
    try:
        check_url(u)
    except ValueError as e:
        raise HTTPException(400, str(e))
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
        except requests.RequestException as e:
            raise HTTPException(502, f"upstream error: {e.__class__.__name__}")
        except CogError as e:
            raise HTTPException(404, f"not a readable GeoTIFF: {e}")
        except UnsupportedCRS as e:
            raise HTTPException(400, str(e))


@app.get("/c2t/{z}/{x}/{y}.geojson")
async def footprint(z: int, x: int, y: int, u: str = Query(...)):
    """COG footprint as a lon/lat polygon (debug helper)."""
    cog = await _open(u)
    ring = cog.footprint_lonlat()
    return JSONResponse({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]},
                         "properties": {"crs": cog.crs_name, "url": u}})


@app.get("/c2t/{z}/{x}/{y}")
async def tile(z: int, x: int, y: int, u: str = Query(...), format: str = "png",
               quality: int = 80, bidx: str | None = None, rescale: str | None = None,
               blackmax: str | None = None, size: int = 256, request: Request = None):
    if not 0 <= z <= config.MAX_ZOOM:
        raise HTTPException(400, "bad zoom")
    n = 1 << z
    if not (0 <= x < n and 0 <= y < n):
        raise HTTPException(400, "tile index outside matrix")
    if format not in MIME and format != "auto":
        raise HTTPException(400, "unsupported format")
    if not 1 <= size <= 1024:
        raise HTTPException(400, "bad size")
    quality = max(1, min(95, quality))
    # format=auto depends only on the Accept header: webp if image/webp is
    # explicitly listed with q>0, else png. Wildcards (image/*, */*: JOSM,
    # plain HTTP clients) never select webp - JOSM cannot decode it.
    if format == "auto":
        format = "webp" if _accepts_webp(request.headers.get("accept", "")) else "png"
    key = f"{u}|{z}|{x}|{y}|{format}|{quality}|{size}|{bidx}|{rescale}|{blackmax}"
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
    except (CogNotFound, CogFetch, requests.RequestException):
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
    except (CogFetch, requests.RequestException) as e:
        raise HTTPException(502, f"upstream error: {e}")
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


@app.get("/c2t.tms.json")
async def tilejson(u: str = Query(...)):
    cog = await _open(u)
    bbox = web_bbox_lonlat(cog.web_bbox())
    gsd = cog.psx * cog.unit_m
    coarse_gsd = gsd * cog.levels[0].width / cog.levels[-1].width
    res0 = 2 * math.pi * crs.R_WEB / 256.0
    return JSONResponse({
        "tilejson": "2.2.0", "name": "cog2tms", "scheme": "xyz",
        "tiles": ["/c2t/{z}/{x}/{y}?u=" + u],
        "bbox": [round(v, 6) for v in bbox],
        "minzoom": max(0, min(config.MAX_ZOOM, math.floor(math.log2(res0 / max(coarse_gsd, 1e-9))))),
        "maxzoom": max(0, min(config.MAX_ZOOM, math.ceil(math.log2(res0 / gsd)))),
    })


@app.get("/")
async def index():
    return FileResponse(_STATIC / "maplibre.html", media_type="text/html")


@app.get("/maplibre")
async def maplibre():
    return RedirectResponse("/")


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/robots.txt")
async def robots():
    return FileResponse(_STATIC / "robots.txt", media_type="text/plain")
