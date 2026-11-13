# cog2tms — "COG → TMS tiles" service in Python

Online service that takes the URL of a **COG (Cloud Optimized GeoTIFF)** file as input and
serves **image tiles following the TMS standard** (XYZ scheme), **reprojected to EPSG:3857**.
Tile data is fetched from the remote COG via **HTTP byte-range requests** (never the full file).

**Language convention: all code, comments, docstrings, tests, commit messages and docs are in English.**

## API contract

```
GET /                                       # web UI: COG url -> tile urls + MapLibre preview (/maplibre redirects to /)
GET /static/…                               # web UI assets (vendored MapLibre, maplibre_app.js)
GET /c2t/{z}/{x}/{y}?u=https://…/image.tif[&format=png|jpeg][&quality=80][&bidx=1,2,3][&rescale=min,max][&blackmax=0][&size=256]
GET /c2t/{z}/{x}/{y}.geojson?u=…      # COG footprint (debug)
GET /c2t.tms.json?u=…                 # TileJSON: bounds, COG minzoom/maxzoom
GET /healthz
```

Errors: `400` (z/x/y outside matrix, invalid format/URL/scheme, bidx/rescale/blackmax,
16-bit data without `rescale`, unsupported CRS, redirect chain refused or looping),
`404` (unreachable COG, file is not a GeoTIFF), `502` (upstream timeout/error).

## Locked decisions

| Topic | Decision |
|---|---|
| Engine | Pure-Python COG reader over `requests` byte-range; images with Pillow+numpy; JPEG-in-TIFF blocks decoded by **simplejpeg** (identical pixels, releases the GIL) with a Pillow fallback; warp/resample by **`cv2.remap`** (opencv-python-headless — the one heavy wheel, accepted for the render win). Small surface, prebuilt wheels only, full control of what is fetched. |
| Y orientation | **OSM-style XYZ** (Y points down). Pass-through, no flip. A unit test pins the convention. |
| Out-of-bounds tiles | **200 + empty tile**: fully transparent PNG (alpha=0); JPEG has no alpha → solid fill, configurable `EMPTY_TILE_BG` (default `0,0,0`). |
| MVP formats | **PNG and JPEG** (WebP maybe later). |
| Size limit | **No 413 based on Content-Length** (multi-GB COGs are legitimate): validation via magic bytes `II*`/`MM*` + `MAX_ZOOM`. |

## Stack

Python 3.12, **FastAPI + uvicorn**, **requests** (Range reads), **numpy**,
**opencv-python-headless** (`cv2.remap` warp), **simplejpeg** (JPEG-in-TIFF decode;
Pillow takes over if not importable), **Pillow** (PNG/JPEG output, fallback JPEG
decode); pytest + httpx for dev.

### Supported COG features (`app/geotiff.py`)

- Classic TIFF and BigTIFF; tiled layout, planar contig, no rotation.
- Compressions: None, Deflate (+predictor 2), LZW, PackBits, **JPEG (old-style)**: blocks lack
  DQT/DHT → splice IFD `JPEGTables` (minus trailing EOI) + block body (GDAL recipe), then
  simplejpeg (`colorspace="RGB"`, or `"GRAY"` for 1-sample levels — Pillow used as fallback).
- Internal overviews (`NewSubfileType & 1`), **1-bit internal masks** (`& 4`) → `valid` mask.
- GeoTIFF tags: ModelTiepoint/PixelScale (no ModelTransformation), GeoKeyDirectory(+DoubleParams),
  GDAL_NODATA, ExtraSamples→alpha.
- CRS subset (`app/crs.py`, pure Python): 4326, 3857, 2154 (Lambert-93), CC42–CC48 (3942–3948),
  UTM 326xx/327xx, user-defined LCC/TM from GeoTIFF projection keys. Anything else → 400.
  LCC/TM formulas (Snyder) validated against PROJ CLI (<1 cm inside zones).

### Warp pipeline (`app/tiles.py`)

Overview level chosen so the source window ≥ tile/2 px (window **not clamped
to the raster** — `_window` pads outside pixels with zeros + `valid=False`, so
tiles straddling the COG edge go transparent instead of smearing the last
row/column) → read blocks via `reader.read_many` (adjacent ranges merged within
`MERGE_GAP`, groups fetched in parallel) → decode (inline, or in the shared
`_POOL` when the window is ≥ 1 Mpx of blocks **and** the codec releases the GIL)
→ per-pixel source coords (exact affine if source is 3857, else **17×17
inverse-projection control grid** bilinearly interpolated) → bilinear resample by
`cv2.remap` (float32 maps; `data*valid` and `valid` remapped with
`BORDER_CONSTANT`, then divided: out-of-window/out-of-footprint samples
contribute 0 to both, so invalid pixels divide to exactly 0 — no clamp, no
branch) → PNG RGBA (`compress_level=1`) / JPEG.
`cv2.setNumThreads(1)` at import: internal threading gains nothing at 256² and
would oversubscribe across uvicorn workers.

### Performance levers (headline numbers in the README table)

- **Bulk head prefetch** (`HEAD_PREFETCH`): one 64 KB `Range:` read absorbs all
  IFD/tag parsing; IFDs at EOF (unoptimized TIFF) get one chunk around the IFD
  pointer instead.
- `TileOffsets`/`TileByteCounts` **lazy**: small arrays load once per level; big
  ones (the L0 index of a large COG) are fetched per window as a row-major
  contiguous slice (`_Level.index`), a few hundred bytes.
- IFD tag values fetched as one `read_many` batch per IFD; window blocks in
  parallel groups, or one **multi-range GET** (`multipart/byteranges`) when the
  server honors it — rejection remembered **per host**, `MULTIRANGE=0` disables.
- **Encoded-tile LRU** (`TILE_CACHE`) + per-key `asyncio.Lock` single-flight.
- **Block decode** in `_POOL` threads (`DECODE_THREADS`) for big windows: works
  because simplejpeg/zlib release the GIL (Pillow does not), gated by
  `_PAR_MIN_PX` = 1 Mpx (handoff costs ≈ 0.3 ms).
- **Warp = `cv2.remap`** of premultiplied `data*valid` + `valid` + renorm
  (≈1.4–2 ms per 256² tile vs 16–18 ms for the numpy gather; pixels identical
  within 1 level). A dependency-free numpy rewrite only reaches ≈11 ms — this is
  why the opencv wheel is kept.

### Cache coherence (COG replaced under the same URL)

- `httprange._versions`: `url -> (ETag|Last-Modified, total size)` anchored from
  the first 206/HEAD and kept across reader reopens.
- Every later single-range GET carries `If-Range` (direct hosts only, see
  Redirects below); a replaced file makes the server answer `200 + whole file`
  → `CogChanged`, raised **before the body is read** (a 3.8 GB replacement is
  never downloaded). `If-Range` is not sent on multi-range GETs (those answer
  200+whole file regardless); for servers that ignore `If-Range`, ETag/size drift
  between responses also raises `CogChanged`.
- Cached tiles are revalidated by one 1-byte `If-Range` GET at most every
  `VALIDATE_TTL` s (real reads refresh the stamp; failures are ignored →
  cached bytes keep being served, stale-while-error).
- On `CogChanged`: `_purge(url)` = `pool.invalidate` + `httprange.forget` + drop
  every tile-cache key starting with `url|`, then the request re-renders once;
  two render failures in a row → 503.
- Behind a redirect, `If-Range`/ETag anchoring is **skipped** (`_direct` false):
  validators of one host must never be presented to another (alternating mirrors
  would produce endless `CogChanged` → 503). Size drift is still checked, and the
  probe re-walks the chain, catching a COG that moved servers. Not caught: an
  in-place replacement behind a stable target with identical total size.

### Redirects answered by the remote server

- `httprange._follow` walks 30x itself (`allow_redirects=False` everywhere,
  `MAX_REDIRECTS`=5, loop detection): every `Location` is resolved against the
  response URL (`urljoin`, relative hops work) and passed through `check_url`
  **before** connecting — a remote server cannot send us to a private/reserved
  host or outside `ALLOWED_HOSTS`. Refusal/loop/hop-limit → `CogRedirect` → 400;
  the refused host is never contacted.
- One walk per reader, not per range: reads then go straight to the settled host
  (`_target`); the 404/403 negative cache stays keyed by the requested URL.
  A 3xx body is closed unread.

## Layout

```
cog2tms/
├── AGENTS.md
├── pyproject.toml         # fastapi, uvicorn[standard], requests, pillow, simplejpeg, numpy, opencv-python-headless ; dev: pytest, httpx
├── app/
│   ├── main.py            # routes + error mapping + TileJSON + geojson + CORS
│   ├── cogpool.py         # LRU/TTL pool of open CogFile + render semaphore
│   ├── geotiff.py         # TIFF/BigTIFF parser, decoders (simplejpeg+Pillow, block pool), window reader
│   ├── httprange.py       # RangeReader (requests) / LocalReader, range merging,
│   │                      # retries, negative cache, ETag/If-Range version anchors,
│   │                      # own 30x walking with check_url on every Location
│   ├── tiles.py           # XYZ math, overview pick, warp grid, bilinear sampler
│   ├── crs.py             # pure-Python CRS math + minimal EPSG table
│   ├── empty.py           # memoized empty PNG/JPEG tiles
│   ├── config.py          # settings from env
│   ├── security.py        # URL validation (scheme + anti-SSRF + allowlist)
│   └── static/            # web UI: maplibre.html + maplibre_app.js + vendored maplibre-gl (js/css)
├── tests/
│   ├── conftest.py        # pure-Python COG writer + Range server with byte counter
│   ├── test_tile.py       # offline unit tests
│   ├── test_ranges.py     # local byte-range proof (bytes served, Range-only)
│   ├── test_multirange.py # multi-range support + per-host fallback learning
│   ├── test_retries.py    # fault injection: 429/5xx retries, negative cache
│   ├── test_versions.py   # coherence when the COG is replaced under one URL
│   ├── test_redirects.py  # own 30x walking + per-hop check_url + target moves
│   ├── test_codecs.py     # JPEG-in-TIFF fixtures, simplejpeg/Pillow parity, threaded decode
│   └── test_network.py    # integration on remote COGs (`network` marker) + relay proxy proof
├── Dockerfile             # python:3.12-slim → pip install . → uvicorn --workers 4
└── README.md
```

Code style: **minimal code and comments**, idiomatic, no over-engineering.

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

Fixtures are written **in pure Python by conftest** (tiled TIFF + internal 2×
overview + GeoTIFF tags): EPSG:4326 and EPSG:3857 COGs with identical content, a
**JPEG-in-TIFF** COG and a **single-band JPEG** one, served by a local
Range-capable HTTP server that counts bytes. Pinned behaviors (do not break):

- center tile PNG; **XYZ convention** (a northern-half red square must land in
  the tile's upper half); **reprojection parity** 4326 vs 3857 (MSE < 100);
- **edge tiles**: transparent >2 tile-px outside the footprint (both CRS paths),
  never clamped edge values;
- **blackmax**: split exact (`max|band| ≤ blackmax` → transparent, `>` kept),
  colors untouched; out-of-bounds → transparent PNG / solid JPEG;
- 400 paths (zoom, x/y, format, scheme, bidx, rescale, blackmax), 404 missing
  COG, `size`/`bidx`, TileJSON/GeoJSON/healthz; web UI (`GET /` = MapLibre page
  + vendored assets, `/maplibre` redirects to `/`);
- **byte-range proof**: served bytes ≪ file size, every upstream request has
  `Range:`; multi-range fallback learned per host;
- **codecs**: JPEG-in-TIFF renders like the deflate fixture and stays north-up;
  simplejpeg vs Pillow parity (≤ 2 levels); pooled decode byte-identical to
  inline; `_threaded_decode` gating (GIL-free codec + ≥1 Mpx window);
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
  `EMPTY_TILE_BG` JPEG; edge tile `13/4064/2995` → no opaque pixel >2 tile-px
  outside the web bbox; TileJSON bbox ≈ footprint; GeoJSON ring closed;
  COG_BIG z17 → 200 with coverage (mask applied);
- **remote byte-range**: counting relay proxy in front of COG_SMALL →
  transferred < 1 % of file, no upstream request without `Range:`.

## Roadmap status

| # | Content | Status |
|---|---|---|
| 1 | Skeleton, config, PNG tiles, CogFile pool, empty tiles | done |
| 2 | JPEG/quality, bidx/rescale, `.geojson` + TileJSON + healthz, error mapping, CORS | done |
| 3 | URL validation/anti-SSRF, semaphores, COG fixtures + Range proxy + parity + `network` tests | done |
| 4 | Dockerfile, README, verification against public COGs | done |

Possible follow-ups: same output compression as input,
WebP, average-area resampling option (`resampling=`), rescale
percentile auto, host allowlist tests, BigTIFF overview masks on more samples, seam test
across the 4 Bayonne COGs. Decoding: **imagecodecs** would add the other TIFF codecs
(ZSTD/LERC/CCITT/JPEG-XL) and beat our pure-Python ones, but costs a ~30 MB wheel,
and its LZW output is not decoded by our reader (`bad LZW code`) — variant mismatch
to investigate first. `PNG_COMPRESS` retune (level 0: 6 ms vs 15 ms, +13 % bytes)
is the last non-dependency lever. Known open security gaps (documented, not
implemented): **DNS rebinding / TOCTOU** (`check_url` resolves, then the
connection resolves again — closing it needs pinning the validated IP, delicate
for HTTPS certificate checks) and the `ipaddress.is_global` table being
CPython-version dependent (multicast, NAT64 `64:ff9b::/96` slip through).
