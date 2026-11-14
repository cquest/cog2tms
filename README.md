# cog2tms

[![AI:W WRITE](https://img.shields.io/badge/AI-WRITE-grey?style=flat&logoSize=auto&labelColor=13171f)](https://0rwx.com)
[![ci](https://github.com/cquest/cog2tms/actions/workflows/ci.yml/badge.svg)](https://github.com/cquest/cog2tms/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/cquest/cog2tms/graph/badge.svg)](https://codecov.io/gh/cquest/cog2tms)
![license: WTFPL](https://img.shields.io/badge/license-WTFPL-green)
![python: 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)

[Français](README_fr.md)

Serve map tiles as TMS from a Cloud Optimized GeoTIFF (COG).

Give it the URL of a COG, it reads the file remotely and returns Z/X/Y
(OSM scheme) tiles reprojected to EPSG:3857 (Web Mercator).

## API

```
GET /                                        # web UI (form + MapLibre preview, vector basemap)
GET /static/…                                # web UI assets (vendored MapLibre, maplibre_app.js)
GET /c2t/{z}/{x}/{y}?u=<COG url>[&format=png|jpeg|webp|auto][&quality=80][&size=256]
                      [&bidx=1,2,3][&rescale=min,max][&blackmax=0]
GET /c2t/{z}/{x}/{y}.geojson?u=<COG url>   # COG footprint (debug)
GET /c2t.tms.json?u=<COG url>              # TileJSON (bbox, minzoom/maxzoom)
GET /healthz
GET /robots.txt
```

- Tile scheme: XYZ (OSM/Leaflet numbering, y-down). Output PNG (RGBA), JPEG
  (solid `EMPTY_TILE_BG` outside coverage) or WebP (≈13× smaller than PNG on
  ortho tiles; `WEBP_LOSSLESS=1` for lossless WebP).
- `format=auto` follows only the `Accept` header: webp when `image/webp` is
  listed explicitly (q>0), else png. Wildcards (`image/*`, `*/*`) never select
  webp — JOSM sends them and cannot decode it.
- `blackmax=0..255`: pixels whose brightest band is ≤ this value (near-black,
  typically the unmasked fringes at orthophoto edges) become no-coverage.
  Default `20`; `-1` disables.
- Tiles outside the COG footprint: `200` with an empty (transparent) tile.
- Errors: `400` bad zoom / index / format / size / URL, 16-bit data without
  `rescale`, unsupported CRS, refused redirect chain; `404` COG unreachable or
  not a GeoTIFF; `502` upstream error.
- Anti-scraping: `robots.txt` disallows the tile API (`/c2t`) and every
  response carries `X-Robots-Tag: nofollow, noimageindex` — tile urls are
  images that robots.txt alone would not cover. (Deters well-behaved crawlers
  only, not scrapers.)

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload --port 3857
```

The web UI at `http://127.0.0.1:3857/`: paste a COG URL, get a MapLibre preview
and ready-to-paste tile URLs. Or straight from the shell (the example COG is a
public drone orthophoto, EPSG:3943):

```bash
curl -o tile.png \
  "http://127.0.0.1:3857/c2t/18/130034/95856?u=https://data.cquest.org/temp/cog4josm/COG_BayonneSudEst.tif"
```

In JOSM (Map paint settings → WMS/TMS sources → new `TMS` source) — JOSM's `{y}`
is already top-down, no flip needed:

```
tms:http://127.0.0.1:3857/c2t/{zoom}/{x}/{y}?u=https://data.cquest.org/temp/cog4josm/COG_BayonneSudEst.tif
```

## Tests

```bash
pytest              # offline: fixture COGs written in-process and served locally
pytest -m network   # integration on the public cog4josm COGs
pytest --cov        # branch coverage of app/ (threshold enforced, see pyproject)
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
| z18 tile encoding | PNG 103 KB, WebP 7.9 KB (≈13× smaller, ≈2 ms), JPEG 10 KB |

Levers (bulk head prefetch, lazy block indexes, merged/parallel/multi-range
fetches, `cv2.remap` warp, threaded decode, LRU tile cache with single-flight
locks): detailed in [AGENTS.md](AGENTS.md).

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `MAX_ZOOM` | `24` | highest zoom level served |
| `READER_TTL` | `600` | seconds an opened COG stays cached |
| `POOL_SIZE` | `64` | max concurrently opened COGs |
| `MAX_CONCURRENT_READS` | `8` | max simultaneous tile renders |
| `BLOCK_CACHE` | `1500` | decoded COG blocks kept per open COG |
| `DECODE_THREADS` | `4` | threads decoding a large window's blocks (`1` decodes inline) |
| `TILE_CACHE` | `1024` | encoded tiles kept in memory, LRU (`0` disables) |
| `PNG_COMPRESS` | `1` | PNG deflate level (1 is ~4× faster than 6, ~13% bigger) |
| `WEBP_LOSSLESS` | `0` | `1`: lossless+exact WebP (bit-identical to PNG, ~2× the lossy size) |
| `MERGE_GAP` | `4096` | byte-ranges closer than this share one HTTP request |
| `MULTIRANGE` | `1` | several ranges per GET (multipart/byteranges); rejection remembered per host |
| `HEAD_PREFETCH` | `65536` | bulk head read (bytes) absorbing IFD/tag parsing at open (`0` disables) |
| `HTTP_CONNECT_TIMEOUT` / `HTTP_READ_TIMEOUT` | `10` / `30` | upstream timeouts (s) |
| `HTTP_RETRIES` / `HTTP_BACKOFF` | `2` / `0.3` | transparent retries on 429/5xx (`0` disables) |
| `NEGATIVE_TTL` | `30` | a remote 404/403 short-circuits that COG for this long (`0` disables) |
| `VALIDATE_TTL` | `30` | seconds between two `If-Range` version checks (`-1` disables) |
| `VALIDATE_TIMEOUT` | `5` | timeout (s) of that version check |
| `MAX_REDIRECTS` | `5` | 30x hops followed, each re-validated (`0` refuses redirects) |
| `EMPTY_TILE_BG` | `0,0,0` | solid RGB used for empty JPEG tiles |
| `BLACKMAX` | `20` | default near-black transparency threshold (`-1` disables) |
| `ALLOWED_HOSTS` | — | if set, comma-separated COG host allowlist |

- **Anti-SSRF**: only `http(s)` COG URLs; targets resolving to private or
  reserved IPs are rejected (dev/tests: `ALLOW_PRIVATE_NET=1`); every `Location`
  of a 30x chain is re-validated before connecting.
- **Resilience**: 429/5xx retried transparently (`Retry-After` honored); remote
  404/403 remembered per URL; cached tiles keep being served while upstream is
  down.
- **Coherence**: each COG's `ETag`/`Last-Modified`/size is anchored and later
  requests carry `If-Range`; any drift drops that COG's reader and cached tiles
  — a replacement's whole-file 200 answer is never downloaded.

## Docker

```bash
docker build -t cog2tms .
docker run -p 3857:3857 cog2tms   # http://localhost:3857
```

Design notes, supported-CRS list and roadmap: [AGENTS.md](AGENTS.md).

## License

WTFPL v2 — see [LICENSE](LICENSE). Vendored third-party files keep their own
terms: MapLibre GL JS (BSD-3-Clause, unmodified), the JOSM icon (public domain).
