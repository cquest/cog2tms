"""Bounded transparent retries (httprange's own loop: 429 + 5xx, Retry-After)
and negative caching of remote 404/403: fault-injection servers, no real network."""
import socketserver
import threading
import time

import http.server
import pytest

from app import config
from app.httprange import CogFetch, CogNotFound, RangeReader, _negative

DATA = b"abcdefghij" * 1000


class _Flaky(http.server.BaseHTTPRequestHandler):
    policy: list[int] = []  # statuses served per request; then plain 206
    retry_after: str | None = None
    log: list[str] = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        type(self).log.append(self.path)
        pol = type(self).policy
        status = pol.pop(0) if pol else 206
        rng = self.headers.get("Range", "bytes=0-")
        a, b = (int(v) for v in rng[6:].split(",")[0].split("-"))
        b = min(b, len(DATA) - 1)
        if status == 206:
            body = DATA[a:b + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {a}-{b}/{len(DATA)}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(status)
            if self.retry_after:
                self.send_header("Retry-After", self.retry_after)
            self.send_header("Content-Length", "0")
            self.end_headers()


def _serve(policy=(), retry_after=None):
    handler = type("H", (_Flaky,), {"policy": list(policy),
                                   "retry_after": retry_after, "log": []})
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, handler, f"http://127.0.0.1:{srv.server_address[1]}/f"


@pytest.fixture(autouse=True)
def _no_negative_leftover():
    _negative.clear()
    yield
    _negative.clear()


def test_transient_5xx_retried_transparently():
    srv, h, url = _serve([500, 503])
    try:
        r = RangeReader(url)
        assert r.read(0, 10) == DATA[:10]
        assert len(h.log) == 3  # 2 retries, then success
    finally:
        srv.shutdown()
        srv.server_close()


def test_429_retry_after_honored():
    srv, h, url = _serve([429], retry_after="1")
    try:
        r = RangeReader(url)
        t0 = time.monotonic()
        assert r.read(0, 10) == DATA[:10]
        assert time.monotonic() - t0 >= 0.9  # waited ~Retry-After, not backoff
        assert len(h.log) == 2
    finally:
        srv.shutdown()
        srv.server_close()


def test_persistent_5xx_bounded():
    srv, h, url = _serve([503] * 20)
    try:
        r = RangeReader(url)
        with pytest.raises(CogFetch):  # bounded: attempts then upstream error
            r.read(0, 10)
        assert len(h.log) == config.HTTP_RETRIES + 1  # attempt + bounded retries
    finally:
        srv.shutdown()
        srv.server_close()


def test_negative_cache_404():
    srv, h, url = _serve([404] * 20)
    try:
        r = RangeReader(url)
        with pytest.raises(CogNotFound):
            r.read(0, 10)
        with pytest.raises(CogNotFound):
            r.read(0, 10)  # short-circuited: no upstream request
        r2 = RangeReader(url)  # fresh reader (e.g. pool reopen) as well
        with pytest.raises(CogNotFound):
            r2.read(0, 10)
        assert len(h.log) == 1
    finally:
        srv.shutdown()
        srv.server_close()


def test_negative_cache_403():
    srv, h, url = _serve([403] * 20)
    try:
        r = RangeReader(url)
        with pytest.raises(CogFetch):
            r.read(0, 10)
        with pytest.raises(CogFetch):
            r.read(0, 10)
        assert len(h.log) == 1
    finally:
        srv.shutdown()
        srv.server_close()


def test_negative_expiry():
    config_ttl = config.NEGATIVE_TTL
    config.NEGATIVE_TTL = 0.3
    try:
        srv, h, url = _serve([404, 404])
        try:
            r = RangeReader(url)
            with pytest.raises(CogNotFound):
                r.read(0, 10)
            with pytest.raises(CogNotFound):
                r.read(0, 10)  # cached, no request
            assert len(h.log) == 1
            time.sleep(0.35)
            with pytest.raises(CogNotFound):
                r.read(0, 10)  # expired -> probes again
            assert len(h.log) == 2
        finally:
            srv.shutdown()
            srv.server_close()
    finally:
        config.NEGATIVE_TTL = config_ttl
