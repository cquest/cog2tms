"""Strict coherence when a remote COG is replaced under the same URL: ETag /
If-Range / Content-Range drift evicts readers, decoded blocks and tiles."""
import http.server
import socketserver
import threading
import time

import pytest
from fastapi.testclient import TestClient

from app import config
from app import httprange as hr
from app.httprange import CogChanged, RangeReader, revalidate
from .conftest import BBOX4326, H, W, fixture_tile, write_cog


class _Versioned(http.server.BaseHTTPRequestHandler):
    """Mutable file held in memory, with ETag + If-Range. Flags emulate a
    server that ignores If-Range, or one answering a huge whole-file 200."""
    blob = b""
    etag = '"v1"'
    honor_if_range = True
    big_200 = False
    fail_probe = False
    hang = 0.0
    log: list[str | None] = []
    status: list[int] = []

    def log_message(self, *a):
        pass

    def _send(self, status, body, extra=()):
        type(self).status.append(status)
        self.send_response(status)
        self.send_header("ETag", type(self).etag)
        for k, v in extra:
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        cls = type(self)
        rng = self.headers.get("Range")
        cls.log.append(rng)
        if cls.hang and rng == "bytes=0-0" and self.headers.get("If-Range"):
            time.sleep(cls.hang)
        if cls.fail_probe and rng == "bytes=0-0":
            self._send(503, b"")
            return
        if not rng or not rng.startswith("bytes="):
            self._send(200, cls.blob)
            return
        a, b = rng[6:].split(",")[0].split("-")  # multi-range: serve first only
        a = int(a)
        b = len(cls.blob) - 1 if b == "" else min(len(cls.blob) - 1, int(b))
        known = self.headers.get("If-Range")
        if known and cls.honor_if_range and known != cls.etag:
            # replaced file: answer the whole file, not the range
            cls.status.append(200)
            self.send_response(200)
            self.send_header("ETag", cls.etag)
            body = b"x" * 4_000_000 if cls.big_200 else cls.blob
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if not cls.big_200:  # poison body: a client reading it is a bug
                self.wfile.write(cls.blob)
            return
        self._send(206, cls.blob[a:b + 1],
                   [("Content-Range", f"bytes {a}-{b}/{len(cls.blob)}")])


def _serve(blob, etag='"v1"', **flags):
    handler = type("H", (_Versioned,), dict({"blob": blob, "etag": etag, "log": [],
                                             "status": []}, **flags))
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, handler, f"http://127.0.0.1:{srv.server_address[1]}/c.tif"


def _down(srv):
    srv.shutdown()
    srv.server_close()


@pytest.fixture(autouse=True)
def _clean_state():
    from app.main import clear_caches, pool
    pool.clear()
    clear_caches()
    for d in (hr._negative, hr._versions, hr._checked, hr._host_multi, hr._resolved):
        d.clear()
    yield
    pool.clear()
    clear_caches()
    for d in (hr._negative, hr._versions, hr._checked, hr._host_multi, hr._resolved):
        d.clear()
    if hr._vsession is not None:
        hr._vsession.close()
        hr._vsession = None


def _cog(tmp_path, name, bbox=BBOX4326) -> bytes:
    w, s, e, n = bbox
    write_cog(tmp_path / name, w, n, (e - w) / W, (n - s) / H, geographic=True)
    return (tmp_path / name).read_bytes()


def test_if_range_mismatch_detected_without_download():
    srv, h, url = _serve(b"a" * 1000)
    try:
        r = RangeReader(url)
        assert r.read(0, 100) == b"a" * 100
        assert hr._versions[url] == ('"v1"', 1000)
        h.blob, h.etag, h.big_200 = b"b" * 5000, '"v2"', True
        with pytest.raises(CogChanged):
            r.read(500, 100)
        assert r._size == 1000 and not r._bufs  # whole-file body never read
    finally:
        _down(srv)


def test_etag_drift_detected():
    srv, h, url = _serve(b"a" * 1000, honor_if_range=False)
    try:
        r = RangeReader(url)
        r.read(0, 100)
        h.blob, h.etag = b"b" * 1000, '"v2"'
        with pytest.raises(CogChanged):
            r.read(200, 100)
    finally:
        _down(srv)


def test_size_change_detected_with_stale_etag():
    srv, h, url = _serve(b"a" * 1000, honor_if_range=False)
    try:
        r = RangeReader(url)
        r.read(0, 100)
        h.blob = b"a" * 4000  # server keeps an old ETag: size betrays it
        with pytest.raises(CogChanged):
            r.read(200, 100)
    finally:
        _down(srv)


def test_revalidate_probe_is_cheap_and_gated():
    srv, h, url = _serve(b"a" * 1000)
    try:
        RangeReader(url).read(0, 10)
        hr._checked[url] = 0.0
        revalidate(url)  # unchanged: one 1-byte range
        assert h.log[-1] == "bytes=0-0"
        revalidate(url)  # just looked: no extra request (VALIDATE_TTL gate)
        assert len(h.log) == 2
        h.blob, h.etag = b"c" * 12, '"v2"'
        hr._checked[url] = 0.0
        with pytest.raises(CogChanged):
            revalidate(url)
    finally:
        _down(srv)


def test_cached_tile_purged_when_file_replaced(tmp_path, monkeypatch):
    from app.main import _tiles_cache, app
    srv, h, url = _serve(_cog(tmp_path, "a.tif"))
    srv2, h2, url2 = _serve(_cog(tmp_path, "b.tif", (BBOX4326[0] + 0.01, *BBOX4326[1:])))
    monkeypatch.setattr(config, "VALIDATE_TTL", 0.0)  # probe every cached hit
    z, x, y = fixture_tile()
    path = f"/c2t/{z}/{x}/{y}?u={url}"
    try:
        with TestClient(app) as c:
            first = c.get(path).content
            assert c.get(path).content == first  # cache hit, same version
            h.blob, h.etag = h2.blob, '"v2"'
            fresh = c.get(path)
            assert fresh.content != first
            assert 200 in h.status  # the probe saw the If-Range mismatch
            # the same tile straight from the new file is byte-identical
            assert fresh.content == c.get(f"/c2t/{z}/{x}/{y}?u={url2}").content
            assert hr._versions[url][0] == '"v2"'  # reader re-anchored
            assert len(_tiles_cache) == 2  # one tile per URL, no stale version
    finally:
        _down(srv)
        _down(srv2)


def test_mid_render_change_reopens_cog(tmp_path, monkeypatch):
    """With probes off (VALIDATE_TTL < 0) the change must be caught by the
    render's own reads: reader evicted, tile rendered once more on the new file."""
    from app.main import app, pool
    srv, h, url = _serve(_cog(tmp_path, "a.tif"))
    srv2, h2, url2 = _serve(_cog(tmp_path, "b.tif", (BBOX4326[0] + 0.002, *BBOX4326[1:])))
    monkeypatch.setattr(config, "VALIDATE_TTL", -1.0)
    monkeypatch.setattr(config, "HEAD_PREFETCH", 0)  # force real reads per tile
    z, x, y = fixture_tile()
    z2, x2, y2 = fixture_tile(z + 1)
    try:
        with TestClient(app) as c:
            assert c.get(f"/c2t/{z}/{x}/{y}?u={url}").status_code == 200
            h.blob, h.etag = h2.blob, '"v2"'
            pool._items[url][1]._cache.clear()  # force reads of the replaced file
            tile = c.get(f"/c2t/{z2}/{x2}/{y2}?u={url}")
            assert tile.status_code == 200
            assert 200 in h.status  # a read hit the replaced file
            ref = c.get(f"/c2t/{z2}/{x2}/{y2}?u={url2}").content
            assert tile.content == ref
            assert hr._versions[url][0] == '"v2"'
    finally:
        _down(srv)
        _down(srv2)


def test_warm_reader_change_detected_by_probe(tmp_path, monkeypatch):
    """A pooled reader with all its blocks cached needs no network read: the
    rate-limited probe is what notices the file was replaced."""
    from app.main import app
    srv, h, url = _serve(_cog(tmp_path, "a.tif"))
    srv2, h2, url2 = _serve(_cog(tmp_path, "b.tif", (BBOX4326[0] + 0.002, *BBOX4326[1:])))
    monkeypatch.setattr(config, "VALIDATE_TTL", 0.0)
    z, x, y = fixture_tile()
    z2, x2, y2 = fixture_tile(z + 1)
    try:
        with TestClient(app) as c:
            assert c.get(f"/c2t/{z}/{x}/{y}?u={url}").status_code == 200
            h.blob, h.etag = h2.blob, '"v2"'
            tile = c.get(f"/c2t/{z2}/{x2}/{y2}?u={url}")
            assert tile.status_code == 200
            ref = c.get(f"/c2t/{z2}/{x2}/{y2}?u={url2}").content
            assert tile.content == ref
            assert hr._versions[url][0] == '"v2"'
    finally:
        _down(srv)
        _down(srv2)


def test_probe_failure_keeps_serving_cached_tile(tmp_path, monkeypatch):
    from app.main import app
    srv, h, url = _serve(_cog(tmp_path, "a.tif"))
    monkeypatch.setattr(config, "VALIDATE_TTL", 0.0)
    z, x, y = fixture_tile()
    path = f"/c2t/{z}/{x}/{y}?u={url}"
    try:
        with TestClient(app) as c:
            first = c.get(path).content
            h.fail_probe = True
            assert c.get(path).content == first  # stale-while-error
            assert c.get(path).status_code == 200
            assert hr._versions.get(url) is not None  # still anchored
    finally:
        _down(srv)


def test_slow_probe_does_not_stall_cached_tile(tmp_path, monkeypatch):
    """A revalidation sits in front of the response, so it must be quick."""
    from app.main import app
    srv, h, url = _serve(_cog(tmp_path, "a.tif"), hang=3.0)
    monkeypatch.setattr(config, "VALIDATE_TTL", 0.0)
    monkeypatch.setattr(config, "VALIDATE_TIMEOUT", 0.5)
    z, x, y = fixture_tile()
    path = f"/c2t/{z}/{x}/{y}?u={url}"
    try:
        with TestClient(app) as c:
            first = c.get(path).content  # render: probes are not sent yet
            t0 = time.perf_counter()
            assert c.get(path).content == first  # probe times out, tile served
            assert time.perf_counter() - t0 < 2.0
    finally:
        _down(srv)
