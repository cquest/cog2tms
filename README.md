# cog2tms

Serve map tiles as TMS from a Cloud Optimized GeoTIFF (COG).

Give it the URL of a COG, it reads the file remotely over HTTP byte-range requests
(never the whole file — the demo files are 0.9–3.8 GB) and returns tiles as 256px
XYZ (OSM scheme) tiles reprojected to EPSG:3857 (Web Mercator) — whatever CRS the
COG uses (e.g. Lambert conformal conic French zones, UTM, WGS84).

The COG is parsed by a small pure-Python TIFF reader
(`app/geotiff.py`), accessed with `requests` + `Range:` headers, and pixels are
warped/resampled by `cv2.remap` (opencv-python-headless) with numpy/Pillow for
the rest. Supported: classic TIFF and BigTIFF, tiled layout,
internal overviews, compressions None/Deflate/LZW/PackBits/JPEG (JPEG-in-TIFF
with JPEGTables splicing, decoded by simplejpeg — 3× faster than Pillow, same
pixels — with a Pillow fallback), 1-bit internal alpha masks, GeoTIFF tags, and a
subset of CRSs (EPSG:4326, 3857, Lambert-93, Lambert CC42–CC46, UTM N/S zones,
plus user-defined LCC/TM projections); other CRSs return `400`.

## API

```
GET /c2t/{z}/{x}/{y}?u=<COG url>[&format=png|jpeg][&quality=80][&size=256]
                      [&bidx=1,2,3][&rescale=min,max]
GET /c2t/{z}/{x}/{y}.geojson?u=<COG url>   # COG footprint (debug)
GET /c2t.tms.json?u=<COG url>              # TileJSON (bbox, minzoom/maxzoom)
GET /healthz
```

- Tile scheme: XYZ (OSM/Leaflet numbering, y-down), output PNG (RGBA, alpha 0
  outside coverage) or JPEG (`quality`, solid `EMPTY_TILE_BG` outside coverage).
- Tiles outside the COG footprint return `200` with an empty (transparent) tile.
- `400` bad zoom / tile index / format / size / URL (also 16-bit data without
  `rescale`, unsupported CRS, redirect chain refused because its `Location` is
  not allowed); `404` COG unreachable or not a GeoTIFF; `502` upstream error.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

uvicorn app.main:app --reload --port 3857
```

Fetch a tile (example COGs are public drone orthophotos from the French coast,
in EPSG:3943 Lambert CC43):

```bash
curl -o tile.png \
  "http://127.0.0.1:3857/c2t/18/130034/95856?u=https://data.cquest.org/temp/cog4josm/COG_BayonneSudEst.tif"
```

Use it in JOSM (Preferences → Map paint settings → WMS/TMS sources → New WMS/TMS
source, type `TMS`) with this URL:

```
tms:http://127.0.0.1:3857/c2t/{zoom}/{x}/{y}?u=https://data.cquest.org/temp/cog4josm/COG_BayonneSudEst.tif
```

(JOSM's `{y}` placeholder uses the same top-down numbering as this service, so
no flip is needed; `{-y}` would be required only for true TMS y-up servers.)

## Tests

```bash
pytest              # offline: fixture COGs written in-process and served locally
pytest -m network   # integration on the public cog4josm COGs
```

The offline suite proves with a counting HTTP server that a render transfers
only a fraction of the file and that every upstream request carries a `Range:`
header; the network suite repeats this against the 875 MB COG (< 1 % transferred).

## Performance notes

Measured on the Bayonne COG (875 MB, EPSG:3943) from a workstation:

| Operation | Before | After |
|---|---|---|
| open COG (TIFF/GeoTIFF headers) | 0.39 s, 105 req, 8.4 MB | **0.05 s, 1 req, 64 KB** |
| first tile, fresh process (open + render) | 0.88 s | ≈0.15–0.3 s |
| first tile at max zoom (L0 block index) | ≈ +0.5 s, 6.3 MB of metadata | no extra cost |
| warm render (blocks cached) | ≈50–80 ms | ≈6 ms |
| repeat of the same tile (tile cache) | — | ≈2 ms |

What got it there: a single bulk prefetch of the COG head zone absorbs all
IFD/tag parsing (COGs pack metadata near the head; classic TIFFs with IFDs at
EOF get one chunk around the IFD pointer instead); `TileOffsets`/`TileByteCounts`
are never read wholesale — small arrays load once per level, huge ones (the L0
index of a big COG is ~6 MB for ~800 k blocks) are fetched as a few-hundred-byte
slice per window; one batched + parallel `read_many` per IFD and per window;
parallel range fetches; multi-range GETs
(`multipart/byteranges`, with automatic single-range fallback for servers like
S3 that answer 200 + the whole object); PNG `compress_level=1`;
and an in-process LRU tile cache with single-flight locks (identical concurrent
requests render once).

Warp and resampling was the dominant CPU step (≈16–18 ms of a ≈20 ms warm
render): the bilinear gather is now `cv2.remap` (opencv-python-headless, the one
heavy dependency — 61 MB wheel, but the headless wheels need no system
libraries) with the mask carried as a second channel — data and validity are
remapped premultiplied and renormalized, so out-of-window and out-of-footprint
samples contribute nothing and edge tiles stay transparent instead of smearing
the footprint's last row/column. **16–18 ms → 1.4–2 ms per tile**, pixels
identical to the numpy sampler within 1 level; `cv2.remap` releases the GIL and
is pinned to one internal thread (`cv2.setNumThreads(1)`) so it doesn't
oversubscribe across uvicorn workers.

Block decompression (a few % of a render, since overviews keep the number of
blocks per tile small): JPEG-in-TIFF blocks are decoded by `simplejpeg` at
**74 µs per 128×128 block instead of 210 µs with Pillow**, with identical pixels
(the `JPEGTables` splice is unchanged); simplejpeg also releases the GIL, which
Pillow does not, so the blocks of a large window (≥ 1 Mpx of source blocks, e.g.
512px-tiled COGs) are decoded in `DECODE_THREADS` threads — 13.4 ms → 5.6 ms for
the 8 blocks of an Anglet z19 tile, and cold renders gain 0–20% depending on how
much work a tile is. Pillow-only installs (no wheel for the platform) simply keep
decoding inline.

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `MAX_ZOOM` | `24` | highest zoom level served |
| `READER_TTL` | `600` | seconds an opened COG stays cached |
| `POOL_SIZE` | `64` | max number of concurrently opened COGs |
| `MAX_CONCURRENT_READS` | `8` | max simultaneous tile renders |
| `BLOCK_CACHE` | `1500` | decoded COG blocks kept per open COG |
| `DECODE_THREADS` | `4` | threads decoding the blocks of a large window (`1` decodes inline; JPEG needs simplejpeg) |
| `TILE_CACHE` | `1024` | encoded tiles kept in memory, LRU (`0` disables) |
| `PNG_COMPRESS` | `1` | PNG deflate level (1 is ~4× faster than 6, ~13% bigger) |
| `MERGE_GAP` | `4096` | byte-ranges closer than this share one HTTP request |
| `MULTIRANGE` | `1` | fetch several ranges in one GET (multipart/byteranges); rejection is remembered per host, never probed again |
| `HEAD_PREFETCH` | `65536` | bulk head read (bytes) that absorbs IFD/tag parsing at open (`0` disables) |
| `HTTP_CONNECT_TIMEOUT` / `HTTP_READ_TIMEOUT` | `10` / `30` | upstream timeouts (s) |
| `HTTP_RETRIES` / `HTTP_BACKOFF` | `2` / `0.3` | transparent retries on 429/5xx (Retry-After honored, exponential backoff); `0` disables |
| `NEGATIVE_TTL` | `30` | seconds a remote 404/403 short-circuits further requests for that COG; `0` disables |
| `VALIDATE_TTL` | `30` | seconds between two `If-Range` checks of a COG version (`-1` disables checks) |
| `VALIDATE_TIMEOUT` | `5` | timeout (s) of that version check, which sits in front of a tile response |
| `MAX_REDIRECTS` | `5` | 30x hops followed for a COG URL (each hop re-validated); `0` refuses any redirect |
| `EMPTY_TILE_BG` | `0,0,0` | solid RGB used for empty JPEG tiles |
| `ALLOWED_HOSTS` | — | if set, comma-separated COG host allowlist |

Security: only `http(s)` COG URLs are accepted, targets resolving to private or
reserved IPs are rejected (anti-SSRF). Set `ALLOW_PRIVATE_NET=1` for tests/dev.

Redirects: `MAX_REDIRECTS` 30x hops are followed for the COG URL, and every
`Location` is re-validated with the same rules (scheme, private IPs,
`ALLOWED_HOSTS`) before we connect to it — an upstream server cannot point the
service at an internal host. The chain is walked once per open COG, not per
range read; a refused or looping chain is a `400`, the refused host is never
contacted.

Resilience: upstream GETs are retried transparently on 429/5xx (bounded, with
exponential backoff; `Retry-After` honored); a remote 404/403 is remembered per
COG URL for `NEGATIVE_TTL` so gone/forbidden files are not re-fetched on every
tile request; already-cached tiles keep being served while upstream is down.

Coherence (COG replaced under the same URL): the `ETag`/`Last-Modified` and the
total size of each remote COG are remembered, and every later `Range` request
carries `If-Range`. A server whose validator no longer matches answers 200 +
the whole file — detected immediately, body never read (a 3.8 GB file is never
downloaded). Cached tiles are also revalidated with one 1-byte `If-Range` GET at
most every `VALIDATE_TTL` seconds. On any drift (ETag change, `If-Range`
mismatch, size change) the reader, its decoded blocks and all cached tiles of
that URL are dropped and the tile is rendered again on the new version — in
about the time of a first tile render. A COG served behind a redirect keeps the
same guarantees except for one case: our own validators are not exchanged with
the target host (mirrors answering different `ETag`s for the same file would
otherwise look like a replacement forever), so replacement in place is tracked
by size only, while the version check re-walks the redirect chain and catches a
COG that moved to another server. Note that downstream caches (browser,
CDN) keep tiles for the `max-age` advertised below; lower it if your COGs are
updated in place.

## Docker

```bash
docker build -t cog2tms .
docker run -p 3857:3857 cog2tms   # http://localhost:3857
```

## Status

MVP complete: PNG/JPEG tiles, overview selection, warp to 3857, empty tiles,
TileJSON/GeoJSON footprints, URL validation, offline + network test suites.
See [AGENTS.md](AGENTS.md) for the design notes and roadmap.
