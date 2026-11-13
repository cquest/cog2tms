"""30x redirects are followed by us, re-validating every Location: an upstream
server cannot point us at a private host or a non-allowlisted one; mirrors with
different ETags never look like a replaced file; the hop is walked once per
reader, and the revalidation probe re-walks it to spot a moved COG."""
import http.server
import socketserver
import threading

import pytest
from fastapi.testclient import TestClient

from app import config, security
from app import httprange as hr
from app.httprange import CogRedirect, RangeReader
from .conftest import BBOX4326, fixture_tile
from .test_versions import _cog, _down, _serve, _Versioned


class _Front(_Versioned):
    """Redirects on `entry` only: other paths are served as a plain file, so a
    relative Location within the same authority can be exercised"""
    entry = "/entry.tif"
    location = ""

    def do_GET(self):
        cls = type(self)
        if self.path != cls.entry:
            return _Versioned.do_GET(self)
        cls.log.append(None)
        cls.status.append(302)
        self.send_response(302)
        self.send_header("Location", cls.location)
        self.send_header("Content-Length", "0")
        self.end_headers()


def _front(blob=b"", entry="/entry.tif", location="", base=_Front, **extra):
    handler = type("H", (base,), dict({"log": [], "status": [], "blob": blob,
                                       "entry": entry, "location": location}, **extra))
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, handler, f"http://127.0.0.1:{srv.server_address[1]}{entry}"


def hops(handler) -> int:
    """how many 302 this server handed out"""
    return handler.status.count(302)


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


def test_two_hops_render_like_direct_access(tmp_path):
    """entry -> alias on another host -> file at another path (relative Location)"""
    blob = _cog(tmp_path, "a.tif")
    srv, h, url = _serve(blob)
    srv1, h1, alias = _front(blob, entry="/a/entry.tif", location="/c.tif")
    srv2, h2, entry = _front(location=alias)
    z, x, y = fixture_tile()
    try:
        assert RangeReader(entry).read(0, 16) == blob[:16]
        assert hops(h2) == 1 and hops(h1) == 1 and len(h.log) == 0
        from app.main import app
        with TestClient(app) as c:
            tile = c.get(f"/c2t/{z}/{x}/{y}?u={entry}")
            assert tile.status_code == 200
            assert tile.content == c.get(f"/c2t/{z}/{x}/{y}?u={url}").content
    finally:
        _down(srv)
        _down(srv1)
        _down(srv2)


def test_hop_outside_allowlist_blocked_before_connecting(monkeypatch):
    """a 302 must not widen what ALLOWED_HOSTS accepts, and the refused host is
    never contacted: 'localhost' and '127.0.0.1' are different names here"""
    monkeypatch.setattr(security, "ALLOWED_HOSTS", ["localhost"])
    srv, h, url = _serve(b"a" * 1000)
    srv1, h1, entry = _front(location=url)
    entry = entry.replace("127.0.0.1", "localhost")  # allowed name, hops elsewhere
    try:
        with pytest.raises(CogRedirect, match="host not allowed"):
            RangeReader(entry).read(0, 10)
        assert h.log == []  # the target server was never reached
    finally:
        _down(srv)
        _down(srv1)


def test_hop_to_private_ip_rejected_even_if_url_was_allowed(monkeypatch):
    """the entry passes check_url only because tests set ALLOW_PRIVATE_NET=1;
    the hop is refused under the production policy"""
    monkeypatch.setattr(security, "ALLOW_PRIVATE_NET", False)
    srv, h, url = _serve(b"a" * 1000)
    srv1, h1, entry = _front(location=url)
    try:
        with pytest.raises(CogRedirect, match="private or reserved"):
            RangeReader(entry).read(0, 10)
        assert h.log == []
    finally:
        _down(srv)
        _down(srv1)


def test_hop_to_other_scheme_rejected():
    srv, h, url = _serve(b"a" * 1000)
    srv1, h1, entry = _front(location="ftp://127.0.0.1:9/x.tif")
    try:
        with pytest.raises(CogRedirect, match="only http"):
            RangeReader(entry).read(0, 10)
        assert h.log == []
    finally:
        _down(srv)
        _down(srv1)


def test_redirect_loop_rejected():
    srv, h, entry = _front()
    h.location = entry  # points at itself
    try:
        with pytest.raises(CogRedirect, match="redirect loop"):
            RangeReader(entry).read(0, 10)
    finally:
        _down(srv)


def test_hop_limit_stops_the_chain(monkeypatch):
    monkeypatch.setattr(config, "MAX_REDIRECTS", 2)
    srvs = [_front(location="http://127.0.0.1:1/x.tif")]
    for _ in range(3):
        srvs.append(_front(location=srvs[-1][2]))
    try:
        with pytest.raises(CogRedirect, match="too many redirects"):
            RangeReader(srvs[-1][2]).read(0, 10)
        assert [hops(h_) for _, h_, _ in srvs] == [0, 1, 1, 1]  # MAX+1 hops walked
    finally:
        for s, _, _ in srvs:
            _down(s)


def test_max_redirects_zero_refuses_any_hop(monkeypatch):
    monkeypatch.setattr(config, "MAX_REDIRECTS", 0)
    srv, h, url = _serve(b"a" * 1000)
    srv1, h1, entry = _front(location=url)
    try:
        with pytest.raises(CogRedirect, match="too many redirects"):
            RangeReader(entry).read(0, 10)
        assert hops(h1) == 1 and h.log == []  # the hop itself was not followed
    finally:
        _down(srv)
        _down(srv1)


def test_hop_walked_once_per_reader_not_per_range(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "VALIDATE_TTL", -1.0)
    monkeypatch.setattr(config, "HEAD_PREFETCH", 0)  # force real reads per render
    blob = _cog(tmp_path, "a.tif")
    srv, h, url = _serve(blob)
    srv1, h1, entry = _front(location=url)
    z, x, y = fixture_tile()
    path = f"/c2t/{z}/{x}/{y}?u={entry}"
    try:
        from app.main import app, clear_caches, pool
        with TestClient(app) as c:
            first = c.get(path)
            assert first.status_code == 200
            assert hops(h1) == 1 and len(h.log) > 1  # one hop, many ranges
            assert c.get(path).content == first.content  # tile cache: no read
            assert hops(h1) == 1
            pool.clear()
            clear_caches()
            assert c.get(path).content == first.content  # reader reopen
            assert hops(h1) == 2
    finally:
        _down(srv)
        _down(srv1)


def test_alternating_mirrors_never_look_replaced(tmp_path, monkeypatch):
    """two servers, same bytes, different ETags (mirrors / CDN re-encode): we
    never present one host's validator to another, so no false CogChanged"""
    monkeypatch.setattr(config, "VALIDATE_TTL", 0.0)  # probe every cached hit
    blob = _cog(tmp_path, "a.tif")
    srv, h, url = _serve(blob, etag='"mirror-a"')
    srv2, h2, url2 = _serve(blob, etag='"mirror-b"')

    class Alternate(_Front):
        def do_GET(self):
            if self.path != type(self).entry:
                return _Versioned.do_GET(self)
            type(self).n += 1
            type(self).location = url2 if type(self).n % 2 else url
            _Front.do_GET(self)

    srv1, h1, entry = _front(base=Alternate, n=0)
    z, x, y = fixture_tile()
    path = f"/c2t/{z}/{x}/{y}?u={entry}"
    try:
        from app.main import app
        with TestClient(app) as c:
            first = c.get(path)
            assert first.status_code == 200
            assert hr._versions[entry] == (None, len(blob))  # no ETag anchored
            fresh = c.get(path)  # the entry now answers the other mirror
            assert fresh.status_code == 200  # not 503
            assert fresh.content == first.content
            assert len(h2.log) >= 1  # mirror B really served the ranges
            assert hr._versions[entry] == (None, len(blob))
    finally:
        _down(srv)
        _down(srv2)
        _down(srv1)


def test_target_moved_mid_reader_purged_by_probe_and_rerendered(tmp_path, monkeypatch):
    """another server takes over behind the same entry, with a file of the same
    size: ETag/size drift see nothing, only the re-walked hop betrays it"""
    monkeypatch.setattr(config, "VALIDATE_TTL", 0.0)
    monkeypatch.setattr(config, "HEAD_PREFETCH", 0)  # force reads per render
    blob = _cog(tmp_path, "a.tif")
    other = _cog(tmp_path, "b.tif", (BBOX4326[0] + 0.01, *BBOX4326[1:]))
    assert len(blob) == len(other)  # size drift alone would see nothing
    srv, h, url = _serve(blob)
    srv2, h2, url2 = _serve(other)
    srv1, h1, entry = _front(location=url)
    z, x, y = fixture_tile()
    z2, x2, y2 = fixture_tile(z + 1)
    try:
        from app.main import app
        with TestClient(app) as c:
            assert c.get(f"/c2t/{z}/{x}/{y}?u={entry}").status_code == 200
            assert hr._resolved[entry] == url
            h1.location = url2  # the entry now points at another server
            moved = c.get(f"/c2t/{z2}/{x2}/{y2}?u={entry}")
            assert moved.status_code == 200
            assert moved.content == c.get(f"/c2t/{z2}/{x2}/{y2}?u={url2}").content
            assert hr._resolved[entry] == url2  # re-anchored on the new target
    finally:
        _down(srv)
        _down(srv2)
        _down(srv1)
