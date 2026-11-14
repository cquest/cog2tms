# cog2tms — "COG → TMS tiles" service in Python

COG URL in, **EPSG:3857 XYZ (OSM) tiles** out; pixels are fetched from the remote
COG by **HTTP byte-range requests only** (never the whole file).
**Language convention: all code, comments, docstrings, tests, commit messages and
docs are in English** — exception: `README_fr.md` is a maintained French mirror of
`README.md`; edit the two together, never let them drift.
Code style: **minimal, idiomatic, no over-engineering**.

## API contract

```
GET /                                    web UI: MapLibre preview (/maplibre redirects to /)
GET /static/…                            UI assets (vendored maplibre-gl, maplibre_app.js)
GET /c2t/{z}/{x}/{y}?u=<cog url>         tile: format=png|jpeg|webp|auto, quality, size,
                                         bidx, rescale, blackmax
GET /c2t/{z}/{x}/{y}.geojson?u=…         COG footprint ring (debug)
GET /c2t.tms.json?u=…                    TileJSON (bbox, minzoom/maxzoom)
GET /healthz
GET /robots.txt                          "User-agent: * Disallow: /c2t" + X-Robots-Tag
                                         (nofollow, noimageindex) on every response
```

Errors: `400` (zoom/x/y outside matrix, format, size, URL/scheme, bidx/rescale/
blackmax, 16-bit data without `rescale`, unsupported CRS, redirect chain refused
or looping), `404` (unreachable COG, not a GeoTIFF), `502` (upstream error).

## Locked decisions

| Topic | Decision |
|---|---|
| Engine | Pure-Python COG reader over `requests` byte-range; images with Pillow+numpy; JPEG-in-TIFF blocks decoded by **simplejpeg** (identical pixels, releases the GIL) with a Pillow fallback; warp/resample by **`cv2.remap`** (opencv-python-headless — the one heavy wheel, accepted for the render win). Small surface, prebuilt wheels only, full control of what is fetched. |
| Y orientation | **OSM-style XYZ** (Y points down). Pass-through, no flip. A unit test pins the convention. |
| Out-of-bounds tiles | **200 + empty tile**: fully transparent PNG/WebP (alpha=0); JPEG has no alpha → solid fill, configurable `EMPTY_TILE_BG` (default `0,0,0`). |
| Output formats | **PNG, JPEG, WebP** + `format=auto` = the `Accept` header only: webp iff `image/webp` (q>0) is listed explicitly, else png; the source codec (JPEG-in-TIFF) plays **no** role. **Wildcards (`image/*`, `*/*`) never select webp** — JOSM and plain clients send them and cannot decode it. WebP = Pillow `exact=True` (without it, RGB is blended to a background where alpha<255) + `method=0`; `WEBP_LOSSLESS=1` → lossless+exact = bit-identical to PNG. The resolved format is part of the tile-cache key. PNG stays the default when `format` is omitted. |
| Size limit | **No 413 based on Content-Length** (multi-GB COGs are legitimate): validation via magic bytes `II*`/`MM*` + `MAX_ZOOM`. |

Stack: Python 3.12, FastAPI + uvicorn, requests (Range reads), numpy, Pillow,
simplejpeg, imagecodecs, opencv-python-headless; dev: pytest + pytest-cov + httpx.

## Supported COG features (`app/geotiff.py`)

- Classic TIFF + BigTIFF; tiled layout, planar contig, no rotation.
- Compressions: None, Deflate (+predictor 2), LZW, PackBits, **ZSTD (50000)**,
  **old-style JPEG**: blocks lack DQT/DHT → splice IFD `JPEGTables` (minus
  trailing EOI) + block body (GDAL recipe), then simplejpeg (`colorspace="RGB"`,
  `"GRAY"` for 1-sample levels; Pillow fallback). LZW/PackBits go through
  **imagecodecs** when installed (it also reads the libtiff LZW variant our
  pure-Python decoder rejects); ZSTD requires imagecodecs — blocks are one plain
  frame or libtiff big-endian size-prefixed segments (`_zstd_decode`).
- Internal overviews (`NewSubfileType & 1`), **1-bit internal masks** (`& 4`) → `valid` mask.
- GeoTIFF tags: ModelTiepoint/PixelScale (no ModelTransformation),
  GeoKeyDirectory(+DoubleParams), GDAL_NODATA, ExtraSamples→alpha.
- CRS subset (`app/crs.py`, pure Python): 4326, 3857, 2154 (Lambert-93),
  CC42–CC48 (3942–3948), UTM 326xx/327xx, user-defined LCC/TM from projection
  keys (Snyder formulas, <1 cm vs PROJ CLI); anything else → 400.

## Warp pipeline (`app/tiles.py`)

Overview level chosen so the source window ≥ tile/2 px; the window is **not
clamped to the raster** — `_window` pads outside pixels with zeros + `valid=False`
so tiles straddling the COG edge go transparent instead of smearing the last
row/column → blocks read via `reader.read_many` (adjacent ranges merged within
`MERGE_GAP`, groups fetched in parallel) → decode inline, or in the shared
`_POOL` when the window is ≥ 1 Mpx **and** the codec releases the GIL → per-pixel
source coords (exact affine if the source is 3857, else **17×17 inverse-projection
control grid** bilinearly interpolated) → bilinear resample by `cv2.remap`
(`data*valid` and `valid` remapped with `BORDER_CONSTANT`, then divided: out-of-
window/out-of-footprint samples contribute 0 to both, so invalid pixels divide to
exactly 0 — no clamp, no branch) → encode PNG/JPEG/WebP (`main._encode`).
`cv2.setNumThreads(1)` at import: internal threading gains nothing at 256² and
would oversubscribe across uvicorn workers.

## Performance levers (headline numbers in the README table)

- **Bulk head prefetch** (`HEAD_PREFETCH`): one 64 KB `Range:` read absorbs all
  IFD/tag parsing; IFDs at EOF get one chunk around the IFD pointer instead.
- `TileOffsets`/`TileByteCounts` **lazy**: small arrays load once per level, big
  ones per window as a row-major contiguous slice (`_Level.index`).
- One `read_many` batch per IFD; window blocks in parallel groups, or one
  **multi-range GET** (`multipart/byteranges`) — rejection remembered per host.
- **Encoded-tile LRU** (`TILE_CACHE`) + per-key `asyncio.Lock` single-flight.
- **Block decode in `_POOL`** (`DECODE_THREADS`, gated by `_PAR_MIN_PX` = 1 Mpx):
  works for GIL-free codecs (simplejpeg, zlib, imagecodecs LZW); Pillow,
  imagecodecs PackBits (9 µs/128² block) and ZSTD (53 µs) stay inline but are
  fast enough.
- **Warp = `cv2.remap`** of premultiplied `data*valid` + `valid` + renorm
  (≈1.4–2 ms per 256² tile vs 16–18 ms for the numpy gather; a dependency-free
  numpy rewrite only reaches ≈11 ms — this is why the opencv wheel is kept).

## Cache coherence (COG replaced under the same URL)

- `httprange._versions`: `url -> (ETag|Last-Modified, total size)` anchored from
  the first 206/HEAD and kept across reader reopens.
- Later single-range GETs carry `If-Range` (direct hosts only): a replaced file
  makes the server answer `200 + whole file` → `CogChanged`, raised **before the
  body is read**. Never sent on multi-range GETs; for servers that ignore
  `If-Range`, ETag/size drift between responses also raises `CogChanged`.
- Cached tiles are revalidated by one 1-byte `If-Range` GET at most every
  `VALIDATE_TTL` s; probe failures are ignored — cached bytes keep being served
  (stale-while-error).
- On `CogChanged`: `_purge(url)` = `pool.invalidate` + `httprange.forget` + drop
  every tile-cache key starting with `url|`, then re-render once; two consecutive
  render failures → 503.
- Behind a redirect, `If-Range`/ETag anchoring is **skipped** (`_direct` false):
  validators of one host must never be presented to another (alternating mirrors
  would cause endless `CogChanged` → 503). Size drift is still checked and the
  probe re-walks the chain. Not caught: in-place replacement behind a stable
  target with identical total size.

## Redirects answered by the remote server

- `httprange._follow` walks 30x itself (`allow_redirects=False` everywhere,
  `MAX_REDIRECTS`=5, loop detection): every `Location` is resolved with `urljoin`
  and passed through `check_url` **before** connecting — a remote server cannot
  send us to a private/reserved host or outside `ALLOWED_HOSTS`. Refusal/loop/
  hop-limit → `CogRedirect` → 400; the refused host is never contacted.
- One walk per reader, not per range: reads go straight to the settled host
  (`_target`); the 404/403 negative cache stays keyed by the requested URL;
  a 3xx body is closed unread.

## Layout

```
cog2tms/
├── AGENTS.md
├── pyproject.toml         # fastapi, uvicorn[standard], requests, pillow, numpy, simplejpeg, imagecodecs, opencv-python-headless ; dev: pytest, pytest-cov, httpx
├── app/
│   ├── main.py            # routes + error mapping + format negotiation + TileJSON + geojson + CORS
│   ├── cogpool.py         # LRU/TTL pool of open CogFile + render semaphore
│   ├── geotiff.py         # TIFF/BigTIFF parser, decoders (simplejpeg, imagecodecs, pure-Python/Pillow fallbacks), window reader
│   ├── httprange.py       # RangeReader (requests) / LocalReader, range merging,
│   │                      # retries, negative cache, ETag/If-Range version anchors,
│   │                      # own 30x walking with check_url on every Location
│   ├── tiles.py           # XYZ math, overview pick, warp grid, bilinear sampler
│   ├── crs.py             # pure-Python CRS math + minimal EPSG table
│   ├── empty.py           # memoized empty PNG/JPEG/WebP tiles
│   ├── config.py          # settings from env
│   ├── security.py        # URL validation (scheme + anti-SSRF + allowlist)
│   └── static/            # web UI: maplibre.html + maplibre_app.js + vendored maplibre-gl
│                          # (js/css) + josm.svg + robots.txt (served at /robots.txt)
├── tests/
│   ├── conftest.py        # pure-Python COG writer + Range server with byte counter
│   ├── test_tile.py       # offline unit tests
│   ├── test_ranges.py     # local byte-range proof (bytes served, Range-only)
│   ├── test_multirange.py # multi-range support + per-host fallback learning
│   ├── test_retries.py    # fault injection: 429/5xx retries, negative cache
│   ├── test_versions.py   # coherence when the COG is replaced under one URL
│   ├── test_redirects.py  # own 30x walking + per-hop check_url + target moves
│   ├── test_codecs.py     # JPEG-in-TIFF fixtures, codec parity, ZSTD, threaded decode
│   ├── test_format.py     # webp output + format=auto Accept negotiation
│   └── test_network.py    # integration on remote COGs (`network` marker) + relay proxy proof
├── Dockerfile             # python:3.12-slim → pip install . → uvicorn --workers 4
└── README.md
```

## App settings (env)

Full table with defaults in the README. Gotchas: CORS is `*` on all routes;
`ALLOW_PRIVATE_NET=1` exists for dev/tests only; `VALIDATE_TTL=-1` disables
version checks; `MULTIRANGE=0` forces single ranges.

## Test COG files (remote, marked `network`)

```python
COG_SMALL  = "https://data.cquest.org/temp/cog4josm/COG_BayonneSudEst.tif"   # 875 MB
COG_BIG    = "https://data.cquest.org/temp/cog4josm/Anglet2025_COG.tif"      # 3.8 GB (BigTIFF + masks)
COG_CENTER = (-1.421406, 43.465922)   # west of Bayonne
```

| File | Size | Format | CRS | Notes |
|---|---|---|---|---|
| COG_BayonneSudEst.tif | 875 MB | classic TIFF, JPEG, 11 levels | EPSG:3943 (CC43) | reference; 128×128 blocks |
| Anglet2025_COG.tif | 3.8 GB | **BigTIFF**, JPEG, 10 levels + **1-bit masks** | EPSG:3943 | very-large-file stress test |
| COG_Bayonne{NordEst,NordOuest,SudOuest}.tif | 1.5–1.7 GB | classic TIFF | EPSG:3943 | 4-tile footprint seam |

Verified: server advertises `Accept-Ranges: bytes`; opaque RGB without NoData on Bayonne;
4.8 cm/px GSD → useful zoom ~17–19. Tiles over `COG_CENTER`: `13/4063/2995`,
`16/32509/23964`, `18/130034/95856` (bright; exact geometric center is dark water).
Anglet center tile: `13/4061/2994`, `17/64982/47917`.

## Tests

### Unit (offline, default: `pytest`)

Fixtures written **in pure Python by conftest** (tiled TIFF + internal 2×
overview + GeoTIFF tags): EPSG:4326 and EPSG:3857 COGs with identical content,
JPEG-in-TIFF, single-band JPEG, LZW/PackBits/ZSTD variants — served by a local
Range-capable HTTP server that counts bytes. Pinned behaviors (do not break):

- center tile PNG; **XYZ convention** (a northern-half red square lands in the
  tile's upper half); **reprojection parity** 4326 vs 3857 (MSE < 100);
- **edge tiles**: transparent >2 tile-px outside the footprint (both CRS paths),
  never clamped edge values;
- **blackmax**: split exact (`max|band| ≤ blackmax` → transparent, `>` kept),
  colors untouched; out-of-bounds → transparent PNG / solid JPEG;
- **webp + auto**: lossy webp pixels within 48 of the png reference (saturated
  fixture colors are webp's worst case); lossless+exact bit-identical to png
  (with blackmax alpha); empty webp tile transparent; the Accept matrix (browser,
  JOSM wildcards, `q=0`, garbage, none) yields webp only on an explicit
  `image/webp;q>0` — the source codec never interferes; cache keyed by the
  resolved format;
- 400 paths (zoom, x/y, format, scheme, bidx, rescale, blackmax), 404 missing
  COG, `size`/`bidx`, TileJSON/GeoJSON/healthz; web UI (`GET /` = MapLibre page
  + vendored assets, `/maplibre` redirects to `/`);
- **byte-range proof**: served bytes ≪ file size, every upstream request has
  `Range:`; multi-range fallback learned per host;
- **codecs**: JPEG-in-TIFF renders like the deflate fixture and stays north-up;
  simplejpeg vs Pillow parity (≤ 2 levels); imagecodecs LZW/PackBits blocks
  byte-exact, including the **libtiff LZW variant** (our pure-Python decoder
  rejects it — pinned); **ZSTD** byte-identical to deflate, plain frame +
  size-prefixed segment blocks, no imagecodecs → CogError; pooled decode
  byte-identical to inline; `_threaded_decode` gating (GIL-free codec + ≥1 Mpx);
- **fault injection**: 429/5xx retries (Retry-After honored, bounded), 404/403
  negative cache;
- **version coherence** (mutable in-memory COG server): `CogChanged` on
  ETag/If-Range/size drift, whole-file 200 body never read, 1-byte probe gated
  by `VALIDATE_TTL`, purge + re-render byte-identical to the new file,
  stale-while-error when the probe fails;
- **redirects** (302 front on the fixtures): 2-hop chain with relative
  `Location` byte-identical to direct; refused hop (host/IP/scheme) never
  contacted; loop + hop limits; one walk per reader; alternating mirror ETags →
  200, never 503; moved entry caught by the probe.

### Integration (`pytest -m network`, excluded by default)

- z18 Bayonne → 200 non-empty; z13 overview; mid-Atlantic → transparent PNG and
  `EMPTY_TILE_BG` JPEG; `format=auto`: Chrome Accept → webp (source codec
  irrelevant), `*/*` → png, empty tile → transparent webp; webp z18 < half the
  png bytes; edge tile `13/4064/2995` → no opaque pixel >2 tile-px outside the
  web bbox; TileJSON bbox ≈ footprint; GeoJSON ring closed; COG_BIG z17 → 200
  with coverage (mask applied);
- **remote byte-range**: counting relay proxy in front of COG_SMALL →
  transferred < 1 % of file, no upstream request without `Range:`.

## Status

Roadmap 1–4 done: skeleton + PNG tiles + pool; JPEG/bidx/rescale/geojson/
TileJSON/healthz/CORS; anti-SSRF + fixtures + network tests; Docker + README +
public-COG verification. WebP and `format=auto` done on top.

## Follow-ups

- **On hold (probed + planned**, `~/.opencode/plan/cog2tms-passthrough.md`):
  byte-passthrough of JPEG-in-TIFF blocks (tier B rewrap, tier A raw copy) and
  `quality=src` via DQT estimation. Rejected for `auto`: webp is smaller than a
  source-quality jpeg re-encode. JPEG-XL *output* also on hold
  (`imagecodecs.jpegxl_*` decodes; Pillow cannot write JXL).
- Close to free: LERC/CCITT/JPEG-XL *decode* (new `comp` dispatch branches +
  fixtures); average-area resampling option (`resampling=`); rescale percentile
  auto; host allowlist tests; BigTIFF overview masks on more samples; seam test
  across the 4 Bayonne COGs.
- Known limits: the pure-Python LZW fallback reads only its own variant (libtiff
  output needs imagecodecs); `PNG_COMPRESS` retune (level 0: 6 ms vs 15 ms,
  +13 % bytes) is the last non-dependency perf lever.
