"""SQLite persistence of COG head spans across restarts (HEAD_CACHE): a span
is saved only when responses carry a change detector, adopted with its
validator so If-Range keeps working, and purged when the file was replaced
or the entry moved - stale head bytes never mix with a new file's blocks."""
import http.server
import pathlib
import socketserver
import sqlite3
import threading

import numpy as np
import pytest

from app import config, headcache, tiles
from app import httprange as hr
from app.geotiff import CogFile
from app.httprange import CogChanged, RangeReader
from .conftest import BBOX4326, H, W, fixture_tile, write_cog


class _Hosted(http.server.BaseHTTPRequestHandler):
    """In-memory COG with ETag + If-Range, logging (Range, If-Range) and
    counting served bytes. etag=None plays the validator-less host; `front`
    turns the entry into a 302 (the file moved to another host)."""
    blob = b""
    etag = '"v1"'
    front = None
    log: list[tuple] = []
    served = 0

    def log_message(self, *a):
        pass

    def _send(self, status, body, extra=()):
        type(self).served += len(body)
        self.send_response(status)
        if type(self).etag:
            self.send_header("ETag", type(self).etag)
        for k, v in extra:
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):  # aborted whole-file 200
            pass

    def do_GET(self):
        cls = type(self)
        rng = self.headers.get("Range")
        cls.log.append((rng, self.headers.get("If-Range")))
        if cls.front:
            self.send_response(302)
            self.send_header("Location", cls.front)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if not rng or not rng.startswith("bytes="):
            self._send(200, cls.blob)
            return
        a, b = rng[6:].split(",")[0].split("-")  # multi-range: serve first
        a = int(a)
        b = len(cls.blob) - 1 if b == "" else min(len(cls.blob) - 1, int(b))
        known = self.headers.get("If-Range")
        if known and known != cls.etag:  # replaced file: 200 + whole file
            self._send(200, cls.blob)
            return
        self._send(206, cls.blob[a:b + 1],
                   [("Content-Range", f"bytes {a}-{b}/{len(cls.blob)}")])


def _serve(**flags):
    handler = type("H", (_Hosted,), dict(flags))
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, handler, f"http://127.0.0.1:{srv.server_address[1]}/c.tif"


def _down(srv):
    srv.shutdown()
    srv.server_close()


def _rows(url):
    db = sqlite3.connect(config.HEAD_CACHE)
    try:
        return db.execute("SELECT off, validator, total FROM heads"
                          " WHERE url = ?", (url,)).fetchall()
    except sqlite3.OperationalError:  # no table yet: nothing was ever saved
        return []
    finally:
        db.close()


def _restart():
    """What a service restart looks like to a fresh reader: no in-memory
    version knowledge left, only what a previous process persisted."""
    from app.main import clear_caches, pool
    pool.clear()
    clear_caches()
    for d in (hr._negative, hr._versions, hr._checked, hr._host_multi,
              hr._resolved):
        d.clear()


@pytest.fixture(autouse=True)
def _state(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "HEAD_CACHE", str(tmp_path / "heads.db"))
    _restart()
    yield
    _restart()


def _cog(tmp_path, dx=0.0) -> bytes:
    w, s, e, n = (v + dx for v in BBOX4326)
    write_cog(tmp_path / "cog.tif", w, n, (e - w) / W, (n - s) / H, geographic=True)
    return (tmp_path / "cog.tif").read_bytes()


def test_span_saved_with_validator_only(tmp_path):
    blob = _cog(tmp_path)
    assert len(blob) > config.HEAD_PREFETCH  # else there is no head span to save
    srv, h, url = _serve(blob=blob)
    try:
        CogFile(RangeReader(url)).close()
        rows = _rows(url)  # conftest IFDs sit near EOF: head span + IFD zone span
        assert [r[1:] for r in rows] == [('"v1"', len(blob))] * len(rows)
        assert len(rows) == 2 and 0 in [r[0] for r in rows]
    finally:
        _down(srv)
    srv, h, url = _serve(blob=blob, etag=None)  # no ETag: nothing re-identifiable
    try:
        CogFile(RangeReader(url)).close()
        assert _rows(url) == []
    finally:
        _down(srv)


def test_restart_reopens_without_the_head_fetch(tmp_path):
    blob = _cog(tmp_path)
    srv, h, url = _serve(blob=blob)
    try:
        cog = CogFile(RangeReader(url))
        rgb1, valid1 = tiles.render_tile(cog, *fixture_tile(), 256)
        cog.close()
        first = h.served
        assert any(r == f"bytes=0-{config.HEAD_PREFETCH - 1}" for r, _ in h.log)

        _restart()
        h.log, h.served = [], 0
        cog = CogFile(RangeReader(url))
        rgb2, valid2 = tiles.render_tile(cog, *fixture_tile(), 256)

        # no prefetch-sized request anymore: spans came from sqlite
        sizes = [int(b) - int(a) + 1 for a, b in
                 (r[6:].split(",")[0].split("-") for r, _ in h.log if r)]
        assert sizes and max(sizes) < config.HEAD_PREFETCH
        assert any(ir == '"v1"' for _, ir in h.log)  # If-Range anchored from row 1
        assert hr._versions[url][0] == '"v1"'
        assert h.served + config.HEAD_PREFETCH <= first
        assert np.array_equal(rgb1, rgb2) and (valid1 == valid2).all()
        assert _rows(url)
        cog.close()
    finally:
        _down(srv)


def test_file_replaced_while_down_purges_the_row(tmp_path):
    srv, h, url = _serve(blob=_cog(tmp_path))
    try:
        CogFile(RangeReader(url)).close()
        assert _rows(url)

        _restart()
        h.blob, h.etag = _cog(tmp_path, dx=0.001), '"v2"'  # replaced under one url
        with pytest.raises(CogChanged):
            tiles.render_tile(CogFile(RangeReader(url)), *fixture_tile(), 256)
        hr.forget(url)  # what _purge does on CogChanged
        assert _rows(url) == []

        rgb, valid = tiles.render_tile(CogFile(RangeReader(url)),
                                       *fixture_tile(), 256)
        assert valid.any()
        rows = _rows(url)  # re-saved on the new file, with the new validator
        assert len(rows) == 2 and {r[1] for r in rows} == {'"v2"'}
    finally:
        _down(srv)


def test_broken_cache_file_is_a_miss_not_an_outage(tmp_path):
    blob = _cog(tmp_path)
    pathlib.Path(config.HEAD_CACHE).write_bytes(b"definitely not sqlite")
    srv, h, url = _serve(blob=blob)
    try:
        cog = CogFile(RangeReader(url))
        rgb, valid = tiles.render_tile(cog, *fixture_tile(), 256)  # plain network path
        assert valid.any() and rgb.any()
        assert any(r == f"bytes=0-{config.HEAD_PREFETCH - 1}" for r, _ in h.log)
        cog.close()
    finally:
        _down(srv)


def test_row_count_bounded_whatever_the_url_flood(monkeypatch):
    monkeypatch.setattr(headcache, "_MAX_ROWS", 2)
    for i in range(5):
        headcache.save(f"http://h/f{i}.tif", 0, '"v"', 10, b"x" * 10)
    db = sqlite3.connect(config.HEAD_CACHE)
    n = db.execute("SELECT COUNT(*) FROM heads").fetchone()[0]
    left = {r[0] for r in db.execute("SELECT url FROM heads")}
    db.close()
    assert n == 2 and left == {"http://h/f3.tif", "http://h/f4.tif"}  # newest kept


def test_moved_entry_never_mixes_cold_spans(tmp_path):
    blob = _cog(tmp_path)
    srv_old, h, url = _serve(blob=blob)  # the entry, direct as long as it answers
    srv_new, hn, new_url = _serve(blob=blob)  # where the file really lives now
    try:
        CogFile(RangeReader(url)).close()
        assert _rows(url)

        _restart()
        h.front = new_url  # the entry now redirects: the file moved
        with pytest.raises(CogChanged):
            tiles.render_tile(CogFile(RangeReader(url)), *fixture_tile(), 256)
        assert _rows(url) == []  # cold spans belonged to the old direct answer

        # a fresh reader follows the move honestly - and behind a redirect
        # nothing is persisted: one host's validators are never anchored there
        rgb, valid = tiles.render_tile(CogFile(RangeReader(url)),
                                       *fixture_tile(), 256)
        assert valid.any()
        assert _rows(url) == [] and _rows(new_url) == []
    finally:
        _down(srv_old)
        _down(srv_new)
