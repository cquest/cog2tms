"""Multi-range GET (multipart/byteranges, RFC 7233) and fallbacks when the
server mishandles it: S3 answers 200 + whole object, nginx answers the first
range only. The client must never consume a whole-file answer."""
import http.server
import socketserver
import threading

from app.httprange import RangeReader

DATA = bytes((i * 7 + i * i) & 0xFF for i in range(60000))
BOUND = "XbOuNdArYxY"

# far apart: 3 groups after MERGE_GAP coalescing -> multi-range candidate
SPANS = [(0, 8), (1000, 100), (20000, 500), (59000, 999)]
EXPECTED = [DATA[o:o + n] for o, n in SPANS]


class _Handler(http.server.BaseHTTPRequestHandler):
    mode = "support"  # support | s3 | firstonly
    log = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        rng = self.headers.get("Range", "")
        type(self).log.append(rng)
        size = len(DATA)
        ranges = []
        if rng.startswith("bytes="):
            ranges = [tuple(int(v) for v in s.strip().split("-"))
                      for s in rng[6:].split(",")]
        if len(ranges) > 1 and self.mode == "s3":  # what S3 does: full object
            self.send_response(200)
            self.send_header("Content-Length", str(size))
            self.end_headers()
            try:
                self.wfile.write(DATA)
            except OSError:
                pass
            return
        if len(ranges) > 1 and self.mode == "firstonly":  # what nginx does
            a, b = ranges[0]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
            self.send_header("Content-Length", str(b - a + 1))
            self.end_headers()
            self.wfile.write(DATA[a:b + 1])
            return
        if len(ranges) > 1:  # RFC-conformant multipart (quoted boundary)
            body = b""
            for a, b in ranges:
                body += (b"--" + BOUND.encode()
                         + f"\r\nContent-Type: application/octet-stream\r\nContent-Range: bytes {a}-{b}/{size}\r\n\r\n".encode()
                         + DATA[a:b + 1] + b"\r\n")
            body += b"--" + BOUND.encode() + b"--\r\n"
            self.send_response(206)
            self.send_header("Content-Type", f'multipart/byteranges; boundary="{BOUND}"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        a, b = ranges[0] if ranges else (0, size - 1)
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
        self.send_header("Content-Length", str(b - a + 1))
        self.end_headers()
        self.wfile.write(DATA[a:b + 1])


def _client(mode):
    handler = type("H", (_Handler,), {"mode": mode, "log": []})
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    reader = RangeReader(f"http://127.0.0.1:{srv.server_address[1]}/f")
    return srv, handler, reader


def test_multirange_one_get():
    srv, h, r = _client("support")
    try:
        assert r.read_many(SPANS) == EXPECTED
        assert len(h.log) == 1 and h.log[0].count(",") == 2  # 3 merged groups
        assert r.size() == len(DATA)  # total seen from parts' Content-Range
    finally:
        srv.shutdown()
        srv.server_close()


def test_fallback_s3_sends_nothing_more_multirange():
    srv, h, r = _client("s3")
    try:
        assert r.read_many(SPANS) == EXPECTED  # served by single-range GETs
        assert r._multi_ok is False
        assert r.read_many(SPANS) == EXPECTED
        assert all("," not in x for x in h.log[1:])  # never retried multi
    finally:
        srv.shutdown()
        srv.server_close()


def test_fallback_first_range_only():
    srv, h, r = _client("firstonly")
    try:
        assert r.read_many(SPANS) == EXPECTED
        assert r._multi_ok is False
        assert h.log[0].count(",") == 2  # attempt was multi-range
        assert all("," not in x for x in h.log[1:])
    finally:
        srv.shutdown()
        srv.server_close()


def test_fallback_learned_per_host():
    """Once a host has rejected multi-ranges, brand-new readers for that host
    (e.g. after the reader pool TTL expired) must not probe again."""
    srv, h, r = _client("s3")
    try:
        assert r.read_many(SPANS) == EXPECTED  # one probe, then fallback
        r2 = RangeReader(r.url)
        assert r2._multi_ok is False  # learned at construction
        assert r2.read_many(SPANS) == EXPECTED
        assert all("," not in x for x in h.log[1:])  # only the first probe
    finally:
        srv.shutdown()
        srv.server_close()
