"""Byte-range proof: a tile render must fetch only a fraction of the file and
every upstream request must carry a Range header."""
import http.server
import socketserver
import threading
import time

import pytest

from app.httprange import CogFetch, RangeReader
from .conftest import COG4326, COGJPT, fixture_tile, tile_url

FIXTURES = (COG4326, COGJPT)  # plain deflate, and JPEG whose DQT sits in tag 347


def test_render_reads_only_needed_bytes(client, server, counter, fixtures_dir):
    """The COG is 200 KB and the render reads a few blocks. (Not pinned on the
    JPEG fixtures: they are ~27 KB, below the 64 KB head prefetch, so a single
    honest range there is the whole file - test_all_upstream_requests_use_range
    covers them instead.)"""
    file_size = (fixtures_dir / COG4326).stat().st_size
    z, x, y = fixture_tile()
    r = client.get(tile_url(server, COG4326, z, x, y))
    assert r.status_code == 200
    served = counter.served
    assert 0 < served < file_size, f"served {served} of {file_size} bytes"
    # sanity: the headers alone already move some bytes; the tile blocks are a
    # small minority of the file (overview path, not the full raster)
    assert served < file_size * 0.5


@pytest.mark.parametrize("name", FIXTURES)
def test_all_upstream_requests_use_range(client, server, counter, name):
    z, x, y = fixture_tile()
    client.get(tile_url(server, name, z, x, y, format="jpeg"))
    assert counter.requests, "server saw no request"
    bad = [(p, r) for p, r in counter.requests if not (r and r.startswith("bytes="))]
    assert not bad, f"requests without Range: {bad}"


def test_range_ignoring_answer_is_cut_short():
    """A server that answers a range request with the whole file must be stopped
    by our byte cap: a few KB arrive, not the file, and the failure is reported
    at once. No Accept-Encoding either: a transparently inflated body would
    break the Content-Length/Content-Range bookkeeping we trust."""
    sent: list[int] = []
    hdrs: dict[str, str] = {}

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            hdrs.update(dict(self.headers))
            self.send_response(200)
            self.send_header("Content-Length", str(100_000))
            self.end_headers()
            try:  # drip the file out: an honest client stops reading early
                for _ in range(100):
                    self.wfile.write(b"x" * 1000)
                    self.wfile.flush()
                    sent.append(1)
                    time.sleep(0.01)
            except OSError:
                pass

    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    reader = RangeReader(f"http://127.0.0.1:{srv.server_address[1]}/big.tif")
    t0 = time.monotonic()
    try:
        with pytest.raises(CogFetch):
            reader.read(1000, 500)
    finally:
        srv.shutdown()
        srv.server_close()
    assert time.monotonic() - t0 < 0.9  # the full drip would take 1 s
    assert len(sent) < 20  # we stopped reading at the cap, the rest never moved
    assert "accept-encoding" not in {k.lower() for k in hdrs}
