# cog2tms — "COG → TMS tiles" service in Python

Online service that takes the URL of a **COG (Cloud Optimized GeoTIFF)** file as input and
serves **image tiles following the TMS standard** (XYZ scheme), **reprojected to EPSG:3857**.
Tile data is fetched from the remote COG via **HTTP byte-range requests** (never the full file).

**Language convention: all code, comments, docstrings, tests, commit messages and docs are in English.**

## API contract

```
GET /c2t/{z}/{x}/{y}?u=https://…/image.tif[&format=png|jpeg][&quality=80][&bidx=1,2,3][&rescale=min,max][&size=256]
GET /c2t/{z}/{x}/{y}.geojson?u=…      # COG footprint (debug)
GET /c2t.tms.json?u=…                 # TileJSON: bounds, COG minzoom/maxzoom
GET /healthz
```

Errors: `400` (z/x/y outside matrix, invalid format/URL/scheme, 16-bit data without `rescale`,
unsupported CRS, redirect chain refused or looping), `404` (unreachable COG, file is not a
GeoTIFF), `502` (upstream timeout/error).

## Locked decisions

| Topic | Decision |
|---|---|
| Engine | **No GDAL / no rio-tiler.** Pure-Python COG reader over `requests` byte-range; images with Pillow+numpy. Small surface, no native wheel, full control of what is fetched. |
| Y orientation | **OSM-style XYZ** (Y points down). Pass-through, no flip. A unit test pins the convention. |
| Out-of-bounds tiles | **200 + empty tile**: fully transparent PNG (alpha=0); JPEG has no alpha → solid fill, configurable `EMPTY_TILE_BG` (default `0,0,0`). |
| MVP formats | **PNG and JPEG** (WebP maybe later). |
| Size limit | **No 413 based on Content-Length** (multi-GB COGs are legitimate): validation via magic bytes `II*`/`MM*` + `MAX_ZOOM`. |

## Stack

- Python 3.12, **FastAPI + uvicorn**, **requests** (Range reads), **numpy** (warp/resample),
  **Pillow** (JPEG-in-TIFF decode, PNG/JPEG output); pytest + httpx for dev.
- No GDAL env vars needed anymore.

### Supported COG features (`app/geotiff.py`)

- Classic TIFF **and BigTIFF** (Anglet2025 is BigTIFF); tiled layout, planar contig, no rotation.
- Compressions: None, Deflate (+predictor 2), LZW, PackBits, **JPEG (old-style)**: blocks lack
  DQT/DHT → splice IFD `JPEGTables` (minus trailing EOI) + block body, then Pillow (GDAL recipe).
- Internal overviews (`NewSubfileType & 1`), **1-bit internal masks** (`& 4`) → `valid` mask.
- GeoTIFF tags: ModelTiepoint/PixelScale (no ModelTransformation), GeoKeyDirectory(+DoubleParams),
  GDAL_NODATA, ExtraSamples→alpha.
- CRS subset (`app/crs.py`, pure Python): 4326, 3857, 2154 (Lambert-93), CC42–CC48 (3942–3948),
  UTM 326xx/327xx, user-defined LCC/TM from GeoTIFF projection keys. Anything else → 400.
  LCC/TM formulas (Snyder) validated against PROJ CLI (<1 cm inside zones).

### Warp pipeline (`app/tiles.py`)

Overview level chosen so the source window ≥ tile/2 px → read blocks via
`reader.read_many` (adjacent ranges merged within `MERGE_GAP`, groups fetched
in parallel) → per-pixel source coords (exact affine if source is 3857, else
**17×17 inverse-projection control grid** bilinearly interpolated) → bilinear
gather with mask-aware renormalization (float32) → PNG RGBA (`compress_level=1`) / JPEG.

### Performance levers (measured on Bayonne COG)

- **Bulk head prefetch** (`RangeReader.prefetch`, `HEAD_PREFETCH`): one 64 KB
  `Range:` read absorbs all IFD/tag parsing → open = 0.05 s / **1 req** (was
  0.39 s / 105 req / 8.4 MB). IFDs at EOF (unoptimized TIFF) get one chunk
  around the IFD pointer instead. `RangeReader` keeps ≤2 prefetched chunks.
- `TileOffsets`/`TileByteCounts` **lazy**: arrays ≤ 1 MB (`_SLICE_AT`) load once
  per level; bigger ones (L0 index ≈ 6 MB) are fetched per window as a
  row-major contiguous slice (`_Level.index`), a few hundred bytes.
- Out-of-line IFD tag values + next-IFD pointer fetched as **one `read_many`
  batch per IFD**; window blocks fetched in parallel groups, or in a **single
  multi-range GET** (`multipart/byteranges`, RFC 7233) when the server honors
  it — body is only consumed on a real multipart 206, and rejection is
  remembered **per host** (`_host_multi`) when the server answers 200+whole-file
  (S3) or first-range-only (nginx), so no reader ever probes such a host again;
  `MULTIRANGE=0` forces single ranges.
- In-process **encoded-tile LRU** (`TILE_CACHE`) + per-key `asyncio.Lock`
  single-flight: repeat of a tile ≈ 2 ms.

### Cache coherence (COG replaced under the same URL)

- `httprange._versions`: `url -> (ETag|Last-Modified, total size)` anchored from
  the first 206/HEAD of each COG and kept across reader reopens.
- Every later single-range GET carries `If-Range` (direct hosts only, see
  Redirects below); a replaced file makes the server answer `200 + whole file`
  → `CogChanged` (subclass of `CogFetch`),
  raised **before the body is read** (`_raw` streams once a validator is known),
  so a 3.8 GB replacement is never downloaded. `If-Range` is deliberately not
  sent on multi-range GETs (those already answer 200+whole file regardless).
- Fallbacks for servers that ignore `If-Range`: ETag/Last-Modified drift between
  responses, or a changed `Content-Range` total, also raise `CogChanged`.
- Cached tiles are revalidated by one 1-byte `If-Range` GET (`revalidate`), at
  most every `VALIDATE_TTL` s (`VALIDATE_TIMEOUT` timeouts, no retry, its own
  keep-alive session). Real reads refresh the stamp, so a busy COG is probed
  rarely. A probe needs an anchor (a validator, or at least a known size behind
  a redirect); a URL never read at all is only checked through its reads.
- On `CogChanged`: `_purge(url)` = `pool.invalidate` + `httprange.forget` + drop
  every tile-cache key starting with `url|`; the request then re-renders once
  (`_render_tile` retry, `_open` reopen retry) — ~ the cost of a first tile.
  Two render failures in a row → 503. Probe failures (404/5xx/timeout) are
  ignored: cached bytes keep being served (stale-while-error).

### Redirects answered by the remote server

- `httprange._follow` walks 301/302/303/307/308 itself (`allow_redirects=False`
  everywhere, `MAX_REDIRECTS`=5, loop detection on the URLs seen): every
  `Location` is resolved against the response URL (`urljoin`, so relative
  hops work) and passed through `check_url` **before** connecting — a remote
  server cannot send us to a private/reserved host or outside `ALLOWED_HOSTS`.
  Refusal/loop/hop-limit → `CogRedirect(CogFetch)` → 400 "unusable COG URL";
  the refused host is never contacted.
- One walk per reader, not per range: `RangeReader` asks the requested URL until
  it has settled, then reads go straight to that host (`_target`); `_host_multi`
  (and the 404/403 negative cache, still keyed by the requested URL) follow the
  settled host. A 3xx body is closed unread.
- Coherence behind a redirect: `If-Range`/ETag anchoring is skipped
  (`_direct` false) because validators of one host must never be presented to
  another — alternating mirrors of the same bytes no longer produce an endless
  `CogChanged` → 503. Size drift is still checked, and `httprange._resolved`
  (requested URL → settled host) lets the 1-byte probe re-walk the chain: a COG
  that moved to another server is caught within `VALIDATE_TTL`. Not caught: an
  in-place replacement behind a stable target with an identical total size.

## Layout

```
cog2tms/
├── AGENTS.md
├── pyproject.toml         # fastapi, uvicorn[standard], requests, pillow, numpy ; dev: pytest, httpx
├── app/
│   ├── main.py            # routes + error mapping + TileJSON + geojson + CORS
│   ├── cogpool.py         # LRU/TTL pool of open CogFile + render semaphore
│   ├── geotiff.py         # TIFF/BigTIFF parser, decoders, window reader
│   ├── httprange.py       # RangeReader (requests) / LocalReader, range merging,
│   │                      # retries, negative cache, ETag/If-Range version anchors,
│   │                      # own 30x walking with check_url on every Location
│   ├── tiles.py           # XYZ math, overview pick, warp grid, bilinear sampler
│   ├── crs.py             # pure-Python CRS math + minimal EPSG table
│   ├── empty.py           # memoized empty PNG/JPEG tiles
│   ├── config.py          # settings from env
│   └── security.py        # URL validation (scheme + anti-SSRF + allowlist)
├── tests/
│   ├── conftest.py        # pure-Python COG writer + Range server with byte counter
│   ├── test_tile.py       # offline unit tests
│   ├── test_ranges.py     # local byte-range proof (bytes served, Range-only)
│   ├── test_multirange.py # multi-range support + per-host fallback learning
│   ├── test_retries.py    # fault injection: 429/5xx retries, negative cache
│   ├── test_versions.py   # coherence when the COG is replaced under one URL
│   ├── test_redirects.py  # own 30x walking + per-hop check_url + target moves
│   └── test_network.py    # integration on remote COGs (`network` marker) + relay proxy proof
├── Dockerfile             # python:3.12-slim → pip install . → uvicorn --workers 4
└── README.md
```

Code style: **minimal code and comments**, idiomatic, no over-engineering.

## App settings (env)

`MAX_ZOOM` (24), `EMPTY_TILE_BG`, `READER_TTL` (600s), `POOL_SIZE` (64),
`MAX_CONCURRENT_READS` (8), `BLOCK_CACHE` (1500 blocks/COG), `MERGE_GAP` (4 KB),
`TILE_CACHE` (1024 encoded tiles, LRU, 0 disables), `PNG_COMPRESS` (1),
`HEAD_PREFETCH` (65536 B bulk head read at open), `MULTIRANGE` (1, multi-range GETs),
`HTTP_CONNECT_TIMEOUT` (10), `HTTP_READ_TIMEOUT` (30), `HTTP_RETRIES` (2, on
429/5xx with backoff + Retry-After), `HTTP_BACKOFF` (0.3), `NEGATIVE_TTL`
(30 s, remote 404/403 cached per URL), `VALIDATE_TTL` (30 s between If-Range
version checks, -1 disables), `VALIDATE_TIMEOUT` (5 s for that probe),
`MAX_REDIRECTS` (5 30x hops, each `Location` re-validated; 0 refuses any),
`ALLOWED_HOSTS` (optional), `ALLOW_PRIVATE_NET` (dev/tests only). CORS `*` on all routes.

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
Fixtures are written **in pure Python by conftest** (tiled TIFF + DEFLATE + internal 2×
overview + GeoTIFF tags): one EPSG:4326 COG, one EPSG:3857 COG (same content), served by a
local Range-capable HTTP server that counts bytes. Coverage: center tile PNG; **XYZ
convention pinned** (red square drawn in the northern half must render in the tile's upper
half); JPEG + quality; out-of-bounds → transparent PNG / solid JPEG; 400 paths (zoom, x/y,
format, scheme, bidx, rescale); 404 missing COG; `size`/`bidx`; TileJSON/GeoJSON/healthz;
**reprojection parity** 4326 vs 3857 (MSE < 100); local byte-range proof (served ≪ file
size, every request has `Range:`); multi-range fallback learned per host; fault-injection
servers for transparent 429/5xx retries (Retry-After honored, bounded attempts) and the
404/403 negative cache; **version coherence** with a mutable in-memory COG server
(ETag/If-Range/size drift → `CogChanged`, whole-file 200 body never read, 1-byte probe
gated by `VALIDATE_TTL`, tile + reader purged and re-render byte-identical to the new
file, stale-while-error when the probe fails or hangs); **redirects** with a 302
front in front of the fixture servers (2-hop chain with a relative `Location`
renders byte-identical to direct access; hop to a non-allowlisted host, a private
IP or another scheme → `CogRedirect` and the refused server never contacted; loop
and `MAX_REDIRECTS` limit; one hop per reader but not per range; alternating
mirror ETags → 200, never 503; entry moved to another server → probe catches it,
tile re-rendered on the new target).

### Integration (`pytest -m network`, excluded by default)
- z18 Bayonne → 200 non-empty (verified mean); z13 overview; mid-Atlantic → transparent PNG
  and `EMPTY_TILE_BG` JPEG; TileJSON bbox ≈ known footprint; GeoJSON ring closed;
  COG_BIG z17 → 200 with coverage (mask applied);
- **remote byte-range**: counting relay proxy in front of COG_SMALL, z18 render →
  transferred < 1 % of file, no upstream request without `Range:`.

Verified end-to-end (uvicorn + curl, post-optimisation): open (TileJSON) ≈ 0.07 s;
first tile ≈ 0.1–0.15 s (any zoom); warm-block render ≈ 20 ms; cached tile ≈ 2 ms;
COG_BIG z17 ≈ 0.13 s. Parity vs rio-tiler reference tiles: corr
0.93–0.98, means within ~1 unit (differences = average vs bilinear resampling only).
Coherence on data.cquest.org (nginx, sends ETag + honors If-Range): stale validator
detected by the 1-byte probe, whole 875 MB 200 answer never read, tile re-rendered
in ≈ 0.15 s; probe cost ≈ 40 ms at most once per `VALIDATE_TTL` per COG (cached tile
stays ≈ 2 ms in between). Behind a 302 entry (fixture proof): first tile ≈ 0.09 s with
**one** hop for the whole render, cached tile ≈ 3.5 ms, entry moved to another server
→ new tile in ≈ 0.08 s (probe + reopen), never a 200 whole-file body, mirrors with
different ETags serve 200 instead of the previous 503.

## Roadmap status

| # | Content | Status |
|---|---|---|
| 1 | Skeleton, config, PNG tiles, CogFile pool, empty tiles | done |
| 2 | JPEG/quality, bidx/rescale, `.geojson` + TileJSON + healthz, error mapping, CORS | done |
| 3 | URL validation/anti-SSRF, semaphores, COG fixtures + Range proxy + parity + `network` tests | done |
| 4 | Dockerfile, README, verification against public COGs | done (image not built locally: daemon without permission here, but `pip install .` + import verified) |

Possible follow-ups: WebP, average-area resampling option (`resampling=`), rescale
percentile auto, host allowlist tests, BigTIFF overview masks on more samples, seam test
across the 4 Bayonne COGs. Known open security gaps (documented, not implemented):
**DNS rebinding / TOCTOU** (`check_url` resolves, then the connection resolves again —
closing it needs pinning the validated IP, delicate for HTTPS certificate checks) and
the `ipaddress.is_global` table being CPython-version dependent (multicast, NAT64
`64:ff9b::/96` slip through).
