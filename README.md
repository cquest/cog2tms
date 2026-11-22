# cog2tms

[![AI:W WRITE](https://img.shields.io/badge/AI-WRITE-grey?style=flat&logoSize=auto&labelColor=13171f)](https://0rwx.com)
[![ci](https://github.com/cquest/cog2tms/actions/workflows/ci.yml/badge.svg)](https://github.com/cquest/cog2tms/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/cquest/cog2tms/graph/badge.svg)](https://codecov.io/gh/cquest/cog2tms)
![license: WTFPL](https://img.shields.io/badge/license-WTFPL-green)
![python: 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)

[Français](README_fr.md)

Give it the URL of a Cloud Optimized GeoTIFF: it reads the file remotely (HTTP
byte-ranges only, never the whole file) and serves Z/X/Y tiles (OSM scheme)
reprojected to EPSG:3857 (Web Mercator).

## API

```
GET /                                       # web UI: MapLibre preview (vector basemap) + tile URLs
GET /c2t/{z}/{x}/{y}?u=<COG url>[&format=png|jpeg|webp|auto][&quality=80][&size=512]
                      [&bidx=1,2,3][&rescale=min,max][&blackmax=0]
GET /c2t/{z}/{x}/{y}.geojson?u=<COG url>   # COG footprint (debug)
GET /c2t.tms.json?u=<COG url>              # TileJSON (bbox, minzoom/maxzoom + "cog"
                                           # block: codec, size, Last-Modified of the COG)
GET /healthz   GET /robots.txt   GET /static/…   # health, robots policy, UI assets
GET /openapi.json   GET /docs   GET /redoc       # OpenAPI 3.1 + docs (swagger-ui from a CDN)
# every route above also answers HEAD: same headers, no body
```

- XYZ (OSM/Leaflet numbering, y-down). Formats: PNG (RGBA), JPEG, WebP
  (≈12× smaller than PNG on ortho tiles). `format=auto` trusts only the
  `Accept` header: webp iff `image/webp` is listed explicitly (q>0), never on
  wildcards (`*/*` — JOSM sends them and cannot decode webp); default png.
- Default tile size `512`, `size=128..1024` restricted to multiples of `16`
  (jpeg macroblock / TIFF block grid)
- Tiles outside the COG footprint: `200` + empty tile — transparent PNG/WebP,
  solid `EMPTY_TILE_BG` for JPEG.
- `blackmax=0..255`: near-black pixels (unmasked orthophoto fringes) become
  no-coverage; default `20`, `-1` disables.
- NODATA becomes transparency
- **CRS**: 4326, 3857, Lambert-93, CC42–48, UTM, user-defined LCC/TM —
  pure-Python and deterministic.
  PROJ's (national grids not shipped). Without it, and for unknown codes, `400`.
- Errors: `400` zoom / index / format / size / URL, signed/32-bit/float data
  without `rescale`, unsupported CRS, refused redirect chain; `404` COG
  unreachable, not a GeoTIFF, unusable sample format; `502` upstream error.

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload --port 3857
```

The web UI at `http://127.0.0.1:3857/` takes a COG URL and shows a map
preview plus ready-to-paste tile URLs. From the shell (public example
orthophoto, EPSG:3943):

```bash
curl -o tile.png \
  "http://127.0.0.1:3857/c2t/18/130034/95856?u=https://data.cquest.org/temp/cog4josm/COG_BayonneSudEst.tif"
```

In JOSM (Map paint settings → WMS/TMS sources → new `TMS` source — its `{y}`
is already top-down):

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
| z18 tile encoding (256 px) | PNG 134 KB, WebP 11.4 KB (≈12× smaller, ≈2 ms), JPEG 13.5 KB |

The levers behind these numbers (head prefetch, lazy block indexes,
merged/parallel/multi-range fetches, `cv2.remap` warp, threaded decode, tile
LRU with single-flight locks): [AGENTS.md](AGENTS.md).

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
| `HEAD_CACHE` | — | sqlite file persisting those head spans across restarts (COGs with ETag/Last-Modified only, rows bounded 4096 + 30-day TTL); empty disables |
| `HTTP_CONNECT_TIMEOUT` / `HTTP_READ_TIMEOUT` | `10` / `30` | upstream timeouts (s) |
| `HTTP_RETRIES` / `HTTP_BACKOFF` | `2` / `0.3` | transparent retries on 429/5xx (`0` disables) |
| `HTTP2` | `1` | HTTP/2 to hosts offering it (ALPN); `0` pins HTTP/1.1 |
| `NEGATIVE_TTL` | `30` | a remote 404/403 short-circuits that COG for this long (`0` disables) |
| `VALIDATE_TTL` | `30` | seconds between two `If-Range` version checks (`-1` disables) |
| `VALIDATE_TIMEOUT` | `5` | timeout (s) of that version check |
| `MAX_REDIRECTS` | `5` | 30x hops followed, each re-validated (`0` refuses redirects) |
| `EMPTY_TILE_BG` | `0,0,0` | solid RGB used for empty JPEG tiles |
| `BLACKMAX` | `20` | default near-black transparency threshold (`-1` disables) |
| `SHARPEN` | `0.3` | unsharp mask on overview tiles (full-resolution tiles untouched; `0` disables) |
| `ALLOWED_HOSTS` | — | if set, comma-separated COG host allowlist |

- **Anti-SSRF**: http(s) only; private/reserved IPs rejected (dev/tests:
  `ALLOW_PRIVATE_NET=1`); every redirect `Location` re-validated before
  connecting; the URL policy (allowlist included) applies to every request,
  cache hits included.
- **Resilience**: 429/5xx retried transparently (`Retry-After` honored); remote
  404/403 remembered; cached tiles keep being served while upstream is down.
- **Coherence**: each COG's `ETag`/`Last-Modified`/size is anchored and later
  requests carry `If-Range`; any drift drops the reader and its cached tiles —
  a replacement's whole-file 200 answer is never downloaded. With
  `HEAD_CACHE` set, head spans survive a restart only for COGs whose
  responses carry such a validator: reopening re-anchors `If-Range` from the
  stored validator, so a file replaced while the service was down is caught
  by the very first block read (and the stale rows purged).

## Docker

```bash
docker build -t cog2tms .
docker run -p 3857:3857 cog2tms   # http://localhost:3857
# head spans kept in a volume across restarts (COGs with ETag/Last-Modified):
docker run -p 3857:3857 -v cog2tms-cache:/var/cache/cog2tms \
  -e HEAD_CACHE=/var/cache/cog2tms/heads.sqlite3 cog2tms
```
## License

WTFPL v2 — see [LICENSE](LICENSE). Vendored third-party files keep their own
terms: MapLibre GL JS (BSD-3-Clause, unmodified), the JOSM icon (public domain).
