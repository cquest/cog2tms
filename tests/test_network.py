"""Integration tests on public COGs (pytest -m network)."""
import io
import threading

import httpx
import pytest
from PIL import Image
import numpy as np

pytestmark = pytest.mark.network

COG_SMALL = "https://data.cquest.org/temp/cog4josm/COG_BayonneSudEst.tif"
COG_BIG = "https://data.cquest.org/temp/cog4josm/Anglet2025_COG.tif"
COG_MERGED = "https://data.cquest.org/temp/cog4josm/COG_Bayonne_2025b.tif"
COG_WEBP = "https://data.cquest.org/temp/cog4josm/Test_webp.tif"
COG_JXL = "https://data.cquest.org/temp/cog4josm/Test_jxl.tif"
COG_CENTER = (-1.421406, 43.465922)


def _t(client, path):
    return client.get(path)


def test_web_ui(client):
    assert _t(client, "/").status_code == 200
    assert _t(client, "/static/maplibre_app.js").status_code == 200


def test_native_tile(client):
    r = _t(client, f"/c2t/18/130034/95856?u={COG_SMALL}")
    assert r.status_code == 200
    a = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"))
    assert a.mean() > 50  # verified bright tile (vegetation)


def test_overview_tile_fast(client):
    # blackmax=-1: the z13 overview of this COG is dark water (mean < 20), it
    # would be fully transparent with the default threshold - this test is about overview rendering
    r = _t(client, f"/c2t/13/4063/2995?u={COG_SMALL}&blackmax=-1")
    assert r.status_code == 200
    a = np.asarray(Image.open(io.BytesIO(r.content)))
    assert (a[..., 3] > 0).sum() > 10000


def test_out_of_bounds(client):
    r = _t(client, f"/c2t/13/4000/2990?u={COG_SMALL}")
    assert r.status_code == 200
    assert np.asarray(Image.open(io.BytesIO(r.content)))[..., 3].max() == 0
    r = _t(client, f"/c2t/13/4000/2990?u={COG_SMALL}&format=jpeg")
    assert (np.asarray(Image.open(io.BytesIO(r.content))) == 0).all()


def test_auto_accept_negotiation_remote(client):
    """format=auto on the real COG follows only the Accept header, whatever the
    source codec: webp to a webp-capable browser (also for the empty tile),
    png to a */* client (JOSM)."""
    ch = {"Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8"}
    r = client.get(f"/c2t/18/130034/95856?u={COG_SMALL}&format=auto", headers=ch)
    assert r.status_code == 200 and r.headers["content-type"] == "image/webp"
    r = client.get(f"/c2t/18/130034/95856?u={COG_SMALL}&format=auto", headers={"Accept": "*/*"})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    r = client.get(f"/c2t/13/4000/2990?u={COG_SMALL}&format=auto", headers=ch)
    assert r.status_code == 200 and r.headers["content-type"] == "image/webp"
    assert np.asarray(Image.open(io.BytesIO(r.content)))[..., 3].max() == 0


def test_webp_smaller_than_png(client):
    """z18 ortho tile: explicit webp must be clearly smaller than the PNG."""
    png = _t(client, f"/c2t/18/130034/95856?u={COG_SMALL}")
    webp = _t(client, f"/c2t/18/130034/95856?u={COG_SMALL}&format=webp")
    assert webp.status_code == 200 and webp.headers["content-type"] == "image/webp"
    assert len(webp.content) * 2 < len(png.content)


def test_edge_tile_transparent(client):
    """z13 tile 4064/2995 straddles the east edge of the (rotated) footprint:
    no pixel more than 2 tile-px outside the web bbox may be opaque — the old
    bug painted whole columns with repeats of the footprint edge column."""
    from app.crs import WEBMERCATOR
    from app.tiles import tile_web_bounds
    a = np.asarray(Image.open(io.BytesIO(
        _t(client, f"/c2t/13/4064/2995?u={COG_SMALL}&blackmax=-1").content)))  # threshold off: test is about edges
    v = a[..., 3] > 0
    assert v.sum() > 20000  # most of the footprint half still renders
    npx = a.shape[0]  # pixel grid of the served tile (the default 512)
    wl, sl, el, nl = _t(client, f"/c2t.tms.json?u={COG_SMALL}").json()["bbox"]
    bx0, by0 = WEBMERCATOR.to_xy(wl, sl)
    bx1, by1 = WEBMERCATOR.to_xy(el, nl)
    tx0, ty0, tx1, ty1 = tile_web_bounds(13, 4064, 2995)
    cell = (tx1 - tx0) / npx
    u = (np.arange(npx) + 0.5) / npx
    dx = np.maximum.reduce([bx0 - (tx0 + (tx1 - tx0) * u),
                            (tx0 + (tx1 - tx0) * u) - bx1,
                            np.zeros(npx)])[None, :]
    dy = np.maximum.reduce([by0 - (ty1 + (ty0 - ty1) * u),
                            (ty1 + (ty0 - ty1) * u) - by1,
                            np.zeros(npx)])[:, None]
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


def test_cog_webp(client):
    """WebP-compressed COG (libtiff tag 50001, 512x512 VP8 blocks, 7 levels):
    full-res pixels match the JPEG COG of the same area within tolerance (two
    lossy encodings of the same imagery, measured MSE 8.4), and overview tiles
    render off the small pyramid."""
    r = _t(client, f"/c2t/18/130009/95839?u={COG_WEBP}")
    assert r.status_code == 200
    a = np.asarray(Image.open(io.BytesIO(r.content)))
    assert (a[..., 3] > 0).sum() > 40000  # the tile sits inside the footprint
    r = _t(client, f"/c2t/18/130009/95839?u={COG_SMALL}")
    b = np.asarray(Image.open(io.BytesIO(r.content)))
    both = (a[..., 3] > 0) & (b[..., 3] > 0)
    mse = np.mean((a[..., :3][both].astype(int) - b[..., :3][both].astype(int)) ** 2)
    assert mse < 40
    r = _t(client, f"/c2t/16/32502/23960?u={COG_WEBP}")  # straddles the footprint
    assert r.status_code == 200
    a = np.asarray(Image.open(io.BytesIO(r.content)))
    assert (a[..., 3] > 0).sum() > 25000  # measured 33.5k: 492x417 m of overlap


def test_cog_jxl(client):
    """JPEG XL compressed COG (libtiff tag 50002, one bare JXL codestream per
    block, 7 levels): full-res pixels match the JPEG COG of the same area
    within tolerance (measured MSE 9.4), overview tiles render off the small
    pyramid, and the TileJSON cog block names the codec and the file."""
    r = _t(client, f"/c2t/18/130009/95839?u={COG_JXL}")
    assert r.status_code == 200
    a = np.asarray(Image.open(io.BytesIO(r.content)))
    assert (a[..., 3] > 0).sum() > 200000  # the tile sits inside the footprint
    r = _t(client, f"/c2t/18/130009/95839?u={COG_SMALL}")
    b = np.asarray(Image.open(io.BytesIO(r.content)))
    both = (a[..., 3] > 0) & (b[..., 3] > 0)
    mse = np.mean((a[..., :3][both].astype(int) - b[..., :3][both].astype(int)) ** 2)
    assert mse < 40
    r = _t(client, f"/c2t/16/32502/23959?u={COG_JXL}")  # straddles the footprint
    assert r.status_code == 200
    a = np.asarray(Image.open(io.BytesIO(r.content)))
    assert (a[..., 3] > 0).sum() > 40000  # measured 47k
    tj = client.get(f"/c2t.tms.json?u={COG_JXL}").json()["cog"]
    assert tj["compression"] == "jpegxl" and tj["size"] > 8_000_000
    assert tj["last_modified"]  # the host carries one: it must show up


def test_cog_merged(client):
    """5.9 GB cog_merger union of the four Bayonne COGs: BigTIFF, overviews as
    SubIFDs (tag 330), absent blocks with byte count 0. Pinned behaviors: the
    overviews are found (fast overview tiles), merger holes are transparent,
    and they are distinguished from the sources' genuine black fringes."""
    r = _t(client, f"/c2t/13/4062/2994?u={COG_MERGED}")
    assert r.status_code == 200
    a = np.asarray(Image.open(io.BytesIO(r.content)))
    assert (a[..., 3] > 0).sum() > 10000  # rendered from SubIFD overviews
    # tile inside the union notch (outside all four sources): fully transparent
    r = _t(client, f"/c2t/17/65034/47949?u={COG_MERGED}")
    assert np.asarray(Image.open(io.BytesIO(r.content)))[..., 3].max() == 0
    # tile straddling the notch over genuine black data: transparent under the
    # default blackmax, opaque black with blackmax=-1 (hole != black fringe)
    r = _t(client, f"/c2t/18/130010/95895?u={COG_MERGED}")
    assert np.asarray(Image.open(io.BytesIO(r.content)))[..., 3].max() == 0
    r = _t(client, f"/c2t/18/130010/95895?u={COG_MERGED}&blackmax=-1")
    a = np.asarray(Image.open(io.BytesIO(r.content)))
    assert (a[..., 3] > 0).sum() > 10000 and a[..., :3][a[..., 3] > 0].max() <= 8


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
                r = httpx.get(relay.remote, headers=hdr, timeout=60)
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
    file_size = httpx.head(COG_SMALL, timeout=30).headers["Content-Length"]
    assert proxy.served * 100 < int(file_size)  # < 1 % of the file


def test_http2_parity(client, monkeypatch):
    """Negotiating HTTP/2 must not move a pixel — and the wire version must
    really be the one asked for, else this would silently compare h1 with h1."""
    from app import config, httprange
    from app.httprange import RangeReader
    from app.main import clear_caches, pool
    path = f"/c2t/13/4063/2995?u={COG_SMALL}&blackmax=-1"

    def render():
        pool.clear()
        clear_caches()
        for d in (httprange._versions, httprange._lastmod, httprange._checked,
                  httprange._host_multi):
            d.clear()
        tile = _t(client, path)
        assert tile.status_code == 200
        rd = RangeReader(COG_SMALL)
        try:
            rd.read(0, 16)
        finally:
            rd.close()
        return tile.content, rd.proto

    monkeypatch.setattr(config, "HTTP2", True)
    h2, proto_h2 = render()
    monkeypatch.setattr(config, "HTTP2", False)
    h1, proto_h1 = render()
    assert (proto_h2, proto_h1) == (3, 2)  # h2 negotiated, then pinned to 1.1
    assert h2 == h1
