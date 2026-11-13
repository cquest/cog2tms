# cog2tms

Serve map tiles as TMS from a Cloud Optimized GeoTIFF (COG).

Give it the URL of a COG, it reads the file remotely over HTTP byte-range requests
(never the whole file — the demo files are 0.9–3.8 GB) and returns tiles as 256px
XYZ (OSM scheme) tiles reprojected to EPSG:3857 (Web Mercator) — whatever CRS the
COG uses (e.g. Lambert conformal conic French zones, UTM, WGS84).

The COG is parsed by a small pure-Python TIFF reader (`app/geotiff.py`), fetched
with `requests` + `Range:` headers, and pixels are warped/resampled by `cv2.remap`
(opencv-python-headless, the only heavy dependency). Supported: classic TIFF and
BigTIFF, tiled layout, internal overviews, compressions None/Deflate/LZW/PackBits/
JPEG (JPEG-in-TIFF decoded by simplejpeg, Pillow fallback), 1-bit internal alpha
masks, GeoTIFF tags, and a subset of CRSs (EPSG:4326, 3857, Lambert-93, Lambert
CC42–CC46, UTM N/S, plus user-defined LCC/TM); other CRSs return `400`.

## API

```
GET /                                        # web UI (form + MapLibre preview, vector basemap)
GET /static/…                                # web UI assets (vendored MapLibre, maplibre_app.js)
GET /c2t/{z}/{x}/{y}?u=<COG url>[&format=png|jpeg][&quality=80][&size=256]
                      [&bidx=1,2,3][&rescale=min,max][&blackmax=0]
GET /c2t/{z}/{x}/{y}.geojson?u=<COG url>   # COG footprint (debug)
GET /c2t.tms.json?u=<COG url>              # TileJSON (bbox, minzoom/maxzoom)
GET /healthz
```

- Tile scheme: XYZ (OSM/Leaflet numbering, y-down), output PNG (RGBA, alpha 0
  outside coverage) or JPEG (`quality`, solid `EMPTY_TILE_BG` outside coverage).
- `blackmax=0..255`: pixels whose brightest band is ≤ this value (near-black,
  typically the unmasked fringes at orthophoto edges) are treated as no-coverage.
  Default `20` (env `BLACKMAX`); `-1` returns every pixel as rendered.
- Tiles outside the COG footprint return `200` with an empty (transparent) tile.
- `400` bad zoom / tile index / format / size / URL (also 16-bit data without
  `rescale`, unsupported CRS, refused redirect chain); `404` COG unreachable or
  not a GeoTIFF; `502` upstream error.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

uvicorn app.main:app --reload --port 3857
```

Fetch a tile (the example COG is a public drone orthophoto, EPSG:3943):

```bash
curl -o tile.png \
  "http://127.0.0.1:3857/c2t/18/130034/95856?u=https://data.cquest.org/temp/cog4josm/COG_BayonneSudEst.tif"
```

In JOSM (Preferences → Map paint settings → WMS/TMS sources → New WMS/TMS source,
type `TMS`), add:

```
tms:http://127.0.0.1:3857/c2t/{zoom}/{x}/{y}?u=https://data.cquest.org/temp/cog4josm/COG_BayonneSudEst.tif
```

(JOSM's `{y}` is already top-down like this service — no `{-y}` flip needed.)

## Web UI

Open `http://127.0.0.1:3857/` in a browser: paste a COG URL and the page shows
a preview map and ready-to-paste URLs.

## Tests

```bash
pytest              # offline: fixture COGs written in-process and served locally
pytest -m network   # integration on the public cog4josm COGs
```

## Performance

Measured on the Bayonne COG (875 MB, EPSG:3943) from a workstation:

| Operation | Measured |
|---|---|
| open COG (TIFF/GeoTIFF headers) | 0.05 s, 1 req, 64 KB |
| first tile, fresh process (open + render) | ≈0.15–0.3 s |
| first tile at max zoom (L0 block index) | no extra cost |
| warm render (blocks cached) | ≈6 ms |
| repeat of the same tile (tile cache) | ≈2 ms |

The levers behind those numbers (bulk head prefetch, lazy block indexes,
batched/parallel and multi-range fetches, `cv2.remap` warp, threaded simplejpeg
block decode, LRU tile cache with single-flight locks) are detailed in
[AGENTS.md](AGENTS.md).

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
| `BLACKMAX` | `20` | default near-black transparency threshold (`-1` disables; per-request `&blackmax=`) |
| `ALLOWED_HOSTS` | — | if set, comma-separated COG host allowlist |

Security, redirects, resilience and cache-coherence behavior in one line each
(full mechanics in [AGENTS.md](AGENTS.md)):

- **Anti-SSRF**: only `http(s)` COG URLs; targets resolving to private or reserved
  IPs are rejected (dev/tests: `ALLOW_PRIVATE_NET=1`). Every `Location` of a 30x
  chain is re-validated with the same rules before connecting.
- **Resilience**: 429/5xx retried transparently (bounded, `Retry-After` honored);
  a remote 404/403 remembered per URL for `NEGATIVE_TTL`; already-cached tiles
  keep being served while upstream is down.
- **Coherence**: the `ETag`/`Last-Modified`/size of each COG are remembered and
  later requests carry `If-Range`; any drift drops the reader and all cached
  tiles of that URL and the tile re-renders on the new version — a replacement's
  whole-file 200 answer is never downloaded.

## Docker

```bash
docker build -t cog2tms .
docker run -p 3857:3857 cog2tms   # http://localhost:3857
```

## Status

MVP complete: PNG/JPEG tiles, overview selection, warp to 3857, empty tiles,
TileJSON/GeoJSON footprints, URL validation, offline + network test suites.
See [AGENTS.md](AGENTS.md) for the design notes and roadmap.
