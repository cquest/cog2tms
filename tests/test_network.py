"""Integration tests on public COGs (pytest -m network)."""
import io
import threading

import pytest
import requests
from PIL import Image
import numpy as np

pytestmark = pytest.mark.network

COG_SMALL = "https://data.cquest.org/temp/cog4josm/COG_BayonneSudEst.tif"
COG_BIG = "https://data.cquest.org/temp/cog4josm/Anglet2025_COG.tif"
COG_CENTER = (-1.421406, 43.465922)


def _t(client, path):
    return client.get(path)


def test_native_tile(client):
    r = _t(client, f"/c2t/18/130034/95856?u={COG_SMALL}")
    assert r.status_code == 200
    a = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"))
    assert a.mean() > 50  # verified bright tile (vegetation)


def test_overview_tile_fast(client):
    r = _t(client, f"/c2t/13/4063/2995?u={COG_SMALL}")
    assert r.status_code == 200
    a = np.asarray(Image.open(io.BytesIO(r.content)))
    assert (a[..., 3] > 0).sum() > 10000


def test_out_of_bounds(client):
    r = _t(client, f"/c2t/13/4000/2990?u={COG_SMALL}")
    assert r.status_code == 200
    assert np.asarray(Image.open(io.BytesIO(r.content)))[..., 3].max() == 0
    r = _t(client, f"/c2t/13/4000/2990?u={COG_SMALL}&format=jpeg")
    assert (np.asarray(Image.open(io.BytesIO(r.content))) == 0).all()


def test_edge_tile_transparent(client):
    """z13 tile 4064/2995 straddles the east edge of the (rotated) footprint:
    no pixel more than 2 tile-px outside the web bbox may be opaque — the old
    bug painted whole columns with repeats of the footprint edge column."""
    from app.crs import WEBMERCATOR
    from app.tiles import tile_web_bounds
    a = np.asarray(Image.open(io.BytesIO(
        _t(client, f"/c2t/13/4064/2995?u={COG_SMALL}").content)))
    v = a[..., 3] > 0
    assert v.sum() > 20000  # most of the footprint half still renders
    w, s, e, n = _t(client, f"/c2t.tms.json?u={COG_SMALL}").json()["bbox"]
    bx0, by0 = WEBMERCATOR.to_xy(w, s)
    bx1, by1 = WEBMERCATOR.to_xy(e, n)
    tx0, ty0, tx1, ty1 = tile_web_bounds(13, 4064, 2995)
    cell = (tx1 - tx0) / 256
    u = (np.arange(256) + 0.5) / 256
    dx = np.maximum.reduce([bx0 - (tx0 + (tx1 - tx0) * u),
                            (tx0 + (tx1 - tx0) * u) - bx1,
                            np.zeros(256)])[None, :]
    dy = np.maximum.reduce([by0 - (ty1 + (ty0 - ty1) * u),
                            (ty1 + (ty0 - ty1) * u) - by1,
                            np.zeros(256)])[:, None]
    assert not (v & ((dx > 2 * cell) | (dy > 2 * cell))).any()


def test_tilejson(client):
    tj = _t(client, f"/c2t.tms.json?u={COG_SMALL}").json()
    w, s, e, n = tj["bbox"]
    assert abs(w - -1.461891) < 0.01 and abs(e - -1.380911) < 0.01
    assert abs(s - 43.443089) < 0.01 and abs(n - 43.488634) < 0.01
    assert 0 <= tj["minzoom"] < tj["maxzoom"]


def test_geojson_footprint(client):
    g = _t(client, f"/c2t/13/4063/2995.geojson?u={COG_SMALL}").json()
    assert g["properties"]["crs"] == "EPSG:3943"
    ring = g["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1]
    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    assert -1.47 < min(lons) and max(lons) < -1.37 and 43.43 < min(lats) and max(lats) < 43.50


def test_cog_big_smoke(client):
    """3.8 GB BigTIFF + internal mask: z17 must answer quickly."""
    r = _t(client, f"/c2t/17/64982/47917?u={COG_BIG}")
    assert r.status_code == 200
    a = np.asarray(Image.open(io.BytesIO(r.content)))
    assert (a[..., 3] > 0).sum() > 10000


# --- remote byte-range proof through a byte-counting relay proxy -----------

class _CountingRelay:
    """Local HTTP relay in front of a remote COG: forwards Range headers,
    counts relayed bytes, records whether every request was ranged."""

    def __init__(self, remote):
        self.remote = remote
        self.served = 0
        self.ranged = []

    def __enter__(self):
        import http.server
        import socketserver

        relay = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                hdr = {}
                if self.headers.get("Range"):
                    hdr["Range"] = self.headers["Range"]
                relay.ranged.append(bool(hdr))
                r = requests.get(relay.remote, headers=hdr, timeout=60)
                body = r.content
                relay.served += len(body)
                self.send_response(r.status_code)
                self.send_header("Content-Length", str(len(body)))
                cr = r.headers.get("Content-Range")
                if cr:
                    self.send_header("Content-Range", cr)
                self.end_headers()
                self.wfile.write(body)

        self.httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/cog.tif"
        return self

    def __exit__(self, *a):
        self.httpd.shutdown()
        self.httpd.server_close()


def test_remote_byte_range(client):
    with _CountingRelay(COG_SMALL) as proxy:
        r = _t(client, f"/c2t/18/130034/95856?u={proxy.url}")
        assert r.status_code == 200
        assert proxy.served < 20 * 1024 * 1024, f"too many bytes: {proxy.served}"
        assert all(proxy.ranged), "a request reached upstream without Range"
    file_size = requests.head(COG_SMALL, timeout=30).headers["Content-Length"]
    assert proxy.served * 100 < int(file_size)  # < 1 % of the file
