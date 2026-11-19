import os

MAX_ZOOM = int(os.getenv("MAX_ZOOM", "24"))
READER_TTL = float(os.getenv("READER_TTL", "600"))
POOL_SIZE = int(os.getenv("POOL_SIZE", "64"))
MAX_CONCURRENT_READS = int(os.getenv("MAX_CONCURRENT_READS", "8"))
EMPTY_TILE_BG = tuple(int(v) for v in os.getenv("EMPTY_TILE_BG", "0,0,0").split(","))

# remote server timeouts
HTTP_CONNECT_TIMEOUT = float(os.getenv("HTTP_CONNECT_TIMEOUT", "10"))
HTTP_READ_TIMEOUT = float(os.getenv("HTTP_READ_TIMEOUT", "30"))

# HTTP/2 (ALPN) for remote COG reads; 0 pins HTTP/1.1
HTTP2 = os.getenv("HTTP2", "1") == "1"

# transparent retries on transient upstream failures (429 + 5xx, honoring
# Retry-After, exponential backoff); 0 disables
HTTP_RETRIES = int(os.getenv("HTTP_RETRIES", "2"))
HTTP_BACKOFF = float(os.getenv("HTTP_BACKOFF", "0.3"))

# a remote 404/403 short-circuits further requests for that COG during this
# window (s); 0 disables
NEGATIVE_TTL = float(os.getenv("NEGATIVE_TTL", "30"))

# cached tiles are revalidated (one-byte If-Range GET) at most every N seconds;
# a replaced COG then drops its stale tiles and reader. -1 disables probing
VALIDATE_TTL = float(os.getenv("VALIDATE_TTL", "30"))

# a revalidation sits in front of a tile response: keep its timeouts short (s)
VALIDATE_TIMEOUT = float(os.getenv("VALIDATE_TIMEOUT", "5"))

# redirects are followed by us, re-validating every Location (scheme, private
# IPs, ALLOWED_HOSTS); 0 refuses any redirect
MAX_REDIRECTS = int(os.getenv("MAX_REDIRECTS", "5"))

# adjacent byte-ranges closer than this are coalesced into one request
MERGE_GAP = int(os.getenv("MERGE_GAP", "4096"))

# decoded blocks kept in memory per open COG
BLOCK_CACHE = int(os.getenv("BLOCK_CACHE", "1500"))

# threads used to decode the blocks of a large window, when the decoder releases
# the GIL (JPEG needs simplejpeg, Deflate uses zlib); 1 decodes everything inline
DECODE_THREADS = int(os.getenv("DECODE_THREADS", "4"))

# encoded tiles kept in memory (LRU, 0 disables); a tile is immutable per params
TILE_CACHE = int(os.getenv("TILE_CACHE", "1024"))

# PNG deflate level: 1 is ~4x faster than default 6 for ~13% bigger files
PNG_COMPRESS = int(os.getenv("PNG_COMPRESS", "1"))
WEBP_LOSSLESS = os.getenv("WEBP_LOSSLESS", "0") == "1"

# fetch several ranges per GET (RFC 7233 multipart/byteranges); auto-disabled
# per reader when a server rejects or ignores it (e.g. S3, nginx)
MULTIRANGE = os.getenv("MULTIRANGE", "1") == "1"

# pixels whose brightest band is <= this are treated as no-coverage (near-black
# fringes of unmasked orthophotos); -1 disables, overridable per request with
# &blackmax=
BLACKMAX = int(os.getenv("BLACKMAX", "20"))

# unsharp-mask strength (0-1) compensating overview averaging: applied only to
# tiles built from an overview level, never at full resolution, and never next
# to no-coverage pixels (halo guard); 0 disables
SHARPEN = float(os.getenv("SHARPEN", "0.3"))

# bulk read of the COG head zone (IFDs, GeoTIFF tags) to avoid many tiny
# round-trips at open; 0 disables
HEAD_PREFETCH = int(os.getenv("HEAD_PREFETCH", "65536"))

# sqlite file persisting prefetched head spans across restarts (only for COGs
# whose responses carry ETag/Last-Modified); empty disables
HEAD_CACHE = os.getenv("HEAD_CACHE", "")
