"""Offline unit tests: fixture COGs served locally over HTTP byte-range."""
import io
import warnings

import numpy as np
import pytest
from PIL import Image

from .conftest import (BBOX4326, COG3857, COG4326, COGBADCRS, COGROT, COGWEBP1,
                       _web, fixture_tile, tile_url)


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_anti_scraping(client, server):
    r = client.get("/robots.txt")
    assert r.status_code == 200 and r.headers["content-type"] == "text/plain; charset=utf-8"
    assert "User-agent: *" in r.text and "Disallow: /c2t" in r.text
    # robots.txt alone would not cover tile urls (images): every response is
    # tagged so well-behaved crawlers keep out of the whole service
    z, x, y = fixture_tile()
    for path in ("/", "/healthz", tile_url(server, COG4326, z, x, y)):
        assert client.get(path).headers["x-robots-tag"] == "nofollow, noimageindex"


def test_head_requests(client, server, counter):
    """HEAD answers like the GET (same headers, same cache entry) with an empty
    body: FastAPI's @app.get() is GET-only, so probes used to get 405 Method Not
    Allowed, and no ASGI server strips the body for us."""
    u = f"{server.base}/{COG4326}"
    for path in ("/healthz", "/robots.txt", "/static/maplibre-gl.css",
                 f"/c2t.tms.json?u={u}", f"/c2t/0/0/0.geojson?u={u}"):
        r = client.head(path)
        # the file paths stream several chunks: all of them must be swallowed
        assert r.status_code == 200 and r.content == b""
    url = tile_url(server, COG4326, *fixture_tile())
    head, get = client.head(url), client.get(url)
    assert head.status_code == 200 and get.status_code == 200
    assert head.content == b""  # our middleware, not the server: uvicorn httptools
    # content-length still describes the GET payload the client would have got
    assert head.headers["content-length"] == str(len(get.content))
    for k in ("content-type", "cache-control", "x-robots-tag"):
        assert head.headers[k] == get.headers[k]
    # the HEAD paid for the render: a following GET leaves the encoded-tile cache
    # without fetching a single upstream byte
    served = counter.served
    assert client.get(url).status_code == 200 and counter.served == served


def test_options_methods(client, server):
    """A bare OPTIONS gets a conformant 405 + Allow; only a CORS preflight is
    answered, and it covers HEAD as well as GET. HEAD itself is not a declared
    route method (see test_head_requests): the Allow header says so anyway."""
    r = client.options(f"/c2t/1/1/1?u={server.base}/{COG4326}")
    assert r.status_code == 405
    assert {"GET", "HEAD"} <= set(r.headers["allow"].split(", "))
    pre = client.options(tile_url(server, COG4326, *fixture_tile()),
                         headers={"Origin": "https://example.org",
                                  "Access-Control-Request-Method": "GET"})
    assert pre.status_code == 200
    assert set(pre.headers["access-control-allow-methods"].split(", ")) == {"GET", "HEAD"}


def test_openapi_spec(client):
    """The generated document is what a client generator consumes: valid (no
    duplicate operationId - a documented HEAD twin per route would add one),
    honest about the image media types and our error contract, and it leaves the
    web UI out."""
    from app.main import app
    app.openapi_schema = None  # regenerate now, so the warnings below are real
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        spec = client.get("/openapi.json").json()
    assert not [w for w in seen if "Duplicate Operation" in str(w.message)]
    ids = [op["operationId"] for ops in spec["paths"].values() for op in ops.values()
           if "operationId" in op]
    assert len(ids) == len(set(ids)) > 0
    tile = spec["paths"]["/c2t/{z}/{x}/{y}"]["get"]
    assert set(tile["responses"]["200"]["content"]) == {"image/png", "image/jpeg", "image/webp"}
    assert {"400", "404", "502"} <= set(tile["responses"])
    assert tile["tags"] == ["tiles"]
    query = [p for p in tile["parameters"] if p["in"] == "query"]
    assert {p["name"] for p in query} == {"u", "format", "quality", "bidx", "rescale",
                                          "blackmax", "size"}
    assert all(p.get("description") for p in query)
    assert "/" not in spec["paths"]  # the web UI is not part of the documented api
    for page in ("/docs", "/redoc"):  # kept public: schema + browsable documentation
        assert client.get(page).status_code == 200


def test_web_ui(client):
    r = client.get("/")
    assert r.status_code == 200 and r.headers["content-type"] == "text/html; charset=utf-8"
    assert b"maplibre" in r.content and b"cog2tms" in r.content
    for asset in ("/static/maplibre-gl.js", "/static/maplibre-gl.css", "/static/maplibre_app.js",
                  "/static/josm.svg"):
        assert client.get(asset).status_code == 200
    assert client.get("/static/nope.js").status_code == 404


def test_upstream_user_agent():
    """Upstream COG servers must see "cog2tms/<version>" (a contact URL may
    follow), not the curl/requests default."""
    from app import __version__
    from app.httprange import USER_AGENT, RangeReader, _session
    assert USER_AGENT.startswith(f"cog2tms/{__version__}")
    RangeReader("https://invalid.invalid/x.tif")  # readers hold no session of their own
    assert _session().headers["User-Agent"] == USER_AGENT


def test_center_tile_png(client, server):
    z, x, y = fixture_tile()
    r = client.get(tile_url(server, COG4326, z, x, y))
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    im = Image.open(io.BytesIO(r.content))
    assert im.size == (256, 256)
    a = np.asarray(im)
    assert (a[..., 3] > 0).all()  # centre tile fully covered by the footprint


def test_xyz_convention(client, server):
    """OSM/XYZ: north up, no flip. The red square sits in the northern half of
    the fixture raster, so it must appear in the upper half of the tile."""
    z, x, y = fixture_tile()
    r = client.get(tile_url(server, COG4326, z, x, y))
    a = np.asarray(Image.open(io.BytesIO(r.content)))
    red = (a[..., 0] > 200) & (a[..., 1] < 60) & (a[..., 2] < 60)
    ys = np.nonzero(red.any(axis=1))[0]
    assert len(ys) > 10 and ys.mean() < 128  # red square above the tile mid-line


def test_jpeg(client, server):
    z, x, y = fixture_tile()
    r = client.get(tile_url(server, COG4326, z, x, y, format="jpeg", quality=60))
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    im = Image.open(io.BytesIO(r.content))
    assert im.size == (256, 256) and im.mode == "RGB"


def test_out_of_bounds(client, server):
    """Tile far from the footprint: transparent PNG, solid JPEG."""
    z, x, y = fixture_tile()
    r = client.get(tile_url(server, COG4326, z, x + 40, y))
    assert r.status_code == 200
    a = np.asarray(Image.open(io.BytesIO(r.content)))
    assert a.shape == (256, 256, 4) and a[..., 3].max() == 0
    r = client.get(tile_url(server, COG4326, z, x + 40, y, format="jpeg"))
    im = np.asarray(Image.open(io.BytesIO(r.content)))
    assert (im == np.array([0, 0, 0])).all()  # EMPTY_TILE_BG default


def test_edge_tile_transparent(client, server):
    """Tile straddling the east edge of the footprint: pixels outside the
    footprint must be transparent, not smeared repeats of the edge column."""
    from app.tiles import tile_web_bounds
    z, xc, yc = fixture_tile()
    x = xc + 1  # one tile east of centre: the footprint's east edge crosses it
    e = _web(BBOX4326[2], 0.0)[0]  # fixture east edge in EPSG:3857 m
    for cog in (COG4326, COG3857):
        a = np.asarray(Image.open(io.BytesIO(
            client.get(tile_url(server, cog, z, x, yc)).content)))
        v = a[..., 3] > 0
        assert 0.2 < v.mean() < 0.6, (cog, v.mean())
        tx0, _, tx1, _ = tile_web_bounds(z, x, yc)
        centers = tx0 + (tx1 - tx0) * (np.arange(256) + 0.5) / 256
        beyond_2px = centers > e + 2 * (tx1 - tx0) / 256
        assert not v[:, beyond_2px].any(), (cog, v[:, beyond_2px].sum())


def test_sampler_outside_window_transparent():
    """_sample must not clamp coordinates to the window: samples whose stencil
    lies fully outside contribute nothing (transparent), not edge pixels."""
    from app.tiles import _sample
    data = np.full((4, 4, 3), 100, np.uint8)
    valid = np.ones((4, 4), bool)
    fx, fy = np.meshgrid(np.arange(-2.0, 6.0), np.arange(-2.0, 6.0))
    rgb, v = _sample(data, valid, fx, fy)
    assert (v == (fx >= 0) & (fx <= 3) & (fy >= 0) & (fy <= 3)).all()
    assert (rgb[v] == 100).all() and not rgb[~v].any()


def test_blackmax_transparent(client, server):
    """blackmax: pixels near black on every band (unmasked ortho edges) go
    transparent inside fully covered tiles; other pixels and colors unchanged."""
    z, xc, yc = fixture_tile()
    x, y = xc - 1, yc - 1  # tile over the fixture's NW corner: the pattern is near-black there
    a0 = np.asarray(Image.open(io.BytesIO(
        client.get(tile_url(server, COG4326, z, x, y, blackmax=-1)).content)))  # explicit pass-through
    assert (a0[..., 3] > 0).all()  # tile fully inside the footprint
    a1 = np.asarray(Image.open(io.BytesIO(
        client.get(tile_url(server, COG4326, z, x, y, blackmax=40)).content)))
    assert (a0[..., :3] == a1[..., :3]).all()  # colors untouched, alpha only
    gone = (a0[..., 3] > 0) & (a1[..., 3] == 0)
    kept = a1[..., 3] > 0
    assert gone.sum() > 1000
    assert a1[..., :3][gone].max(axis=-1).max() <= 40
    assert a1[..., :3][kept].max(axis=-1).min() > 40


def test_blackmax_default(client, server):
    """Default threshold BLACKMAX=20: darker-than-default requests keep every
    pixel, the default drops the near-black ones and nothing else."""
    z, xc, yc = fixture_tile()
    x, y = xc - 2, yc - 1  # covers the fixture's near-black NW corner inside the footprint
    a_full = np.asarray(Image.open(io.BytesIO(
        client.get(tile_url(server, COG4326, z, x, y, blackmax=-1)).content)))
    a_def = np.asarray(Image.open(io.BytesIO(
        client.get(tile_url(server, COG4326, z, x, y)).content)))
    assert (a_full[..., :3] == a_def[..., :3]).all()
    gone = (a_full[..., 3] > 0) & (a_def[..., 3] == 0)
    assert gone.sum() > 1000
    assert a_def[..., :3][gone].max(axis=-1).max() <= 20
    assert a_def[..., :3][a_def[..., 3] > 0].max(axis=-1).min() > 20


def test_blackmax_validation(client, server):
    z, x, y = fixture_tile()
    u = f"{server.base}/{COG4326}"
    for bad in ("abc", "256"):
        r = client.get(f"/c2t/{z}/{x}/{y}", params={"u": u, "blackmax": bad})
        assert r.status_code == 400, bad
    assert client.get(f"/c2t/{z}/{x}/{y}", params={"u": u, "blackmax": "-1"}).status_code == 200


def test_bad_requests(client, server):
    z, x, y = fixture_tile()
    u = f"{server.base}/{COG4326}"
    cases = [
        ("/c2t/30/1/1", {"u": u}),                     # zoom > MAX_ZOOM
        (f"/c2t/{z}/99999/{y}", {"u": u}),             # x outside matrix
        (f"/c2t/{z}/{x}/-1", {"u": u}),                # negative y
        (f"/c2t/{z}/{x}/{y}", {"u": u, "format": "tiff"}),
        (f"/c2t/{z}/{x}/{y}", {"u": "ftp://example.com/x.tif"}),
        (f"/c2t/{z}/{x}/{y}", {"u": f"{server.base}/missing.tif"}),  # 404 upstream
        (f"/c2t/{z}/{x}/{y}", {"u": u, "bidx": "0"}),
        (f"/c2t/{z}/{x}/{y}", {"u": u, "rescale": "9,1"}),
    ]
    for path, q in cases:
        r = client.get(path, params=q)
        want = 404 if "missing" in str(q.get("u")) else 400
        assert r.status_code == want, (path, q, r.status_code)


def test_reprojection_parity(client, server):
    """Same footprint, two source CRSs (4326 vs hand-reprojected 3857 box) ->
    the rendered tiles must match closely."""
    z, x, y = fixture_tile()
    a = np.asarray(Image.open(io.BytesIO(
        client.get(tile_url(server, COG4326, z, x, y)).content)))[..., :3]
    b = np.asarray(Image.open(io.BytesIO(
        client.get(tile_url(server, COG3857, z, x, y)).content)))[..., :3]
    mse = ((a.astype(np.float64) - b) ** 2).mean()
    assert mse < 100, mse


def test_tms_json(client, server):
    r = client.get(f"/c2t.tms.json?u={server.base}/{COG4326}")
    assert r.status_code == 200
    tj = r.json()
    assert tj["scheme"] == "xyz" and tj["minzoom"] <= tj["maxzoom"]
    w, s, e, n = BBOX4326
    assert abs(tj["bbox"][0] - w) < 0.01 and abs(tj["bbox"][2] - e) < 0.01


def test_geojson(client, server):
    z, x, y = fixture_tile()
    r = client.get(f"/c2t/{z}/{x}/{y}.geojson?u={server.base}/{COG4326}")
    assert r.status_code == 200
    ring = r.json()["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1] and len(ring) > 80


def test_size_and_bidx(client, server):
    z, x, y = fixture_tile()
    r = client.get(tile_url(server, COG4326, z, x, y, size=100))
    assert Image.open(io.BytesIO(r.content)).size == (100, 100)
    r = client.get(tile_url(server, COG4326, z, x, y, bidx="1,1,1"))
    a = np.asarray(Image.open(io.BytesIO(r.content)))
    v = a[..., 3] > 0
    assert (a[..., 0][v] == a[..., 1][v]).all() and (a[..., 1][v] == a[..., 2][v]).all()


def test_cog_local_not_accepted(client, tmp_path):
    """The public API only accepts http(s), never local paths."""
    r = client.get("/c2t/1/0/0", params={"u": "/etc/hostname"})
    assert r.status_code == 400


def test_endpoint_error_paths(client, server):
    """geojson and TileJSON go through the same URL/open error mapping."""
    z, x, y = fixture_tile()
    cases = [
        ("ftp://example.invalid/x.tif", 400),                     # scheme
        (f"{server.base}/missing.tif", 404),                      # unreachable
        (f"{server.base}/junk.txt", 404),                         # not a GeoTIFF
        (f"{server.base}/{COGROT}", 404),                         # rotated raster
        (f"{server.base}/{COGWEBP1}", 404),                       # 1-band WebP level
        (f"{server.base}/{COGBADCRS}", 400),                      # unsupported CRS
    ]
    for path in (f"/c2t/{z}/{x}/{y}.geojson", "/c2t.tms.json"):
        for u, want in cases:
            r = client.get(path, params={"u": u})
            assert r.status_code == want, (path, u, r.status_code)


def test_allowed_hosts(client, server, monkeypatch):
    """A deployment allowlist rejects every other host before connecting."""
    from app import security as sec
    monkeypatch.setattr(sec, "ALLOWED_HOSTS", ["other.example"])
    sec.check_url.cache_clear()
    z, x, y = fixture_tile()
    u = f"{server.base}/{COG4326}"
    for path in (f"/c2t/{z}/{x}/{y}", "/c2t.tms.json"):
        r = client.get(path, params={"u": u})
        assert r.status_code == 400 and "not allowed" in r.json()["detail"]
    # the allowlisted host still works
    from urllib.parse import urlsplit
    monkeypatch.setattr(sec, "ALLOWED_HOSTS", [urlsplit(server.base).hostname])
    sec.check_url.cache_clear()
    assert client.get(f"/c2t/{z}/{x}/{y}", params={"u": u}).status_code == 200
    sec.check_url.cache_clear()  # leave no verdict anchored to a patched allowlist


def test_check_url_guardrails():
    """Direct unit coverage of the SSRF guardrails (cache cleared per case)."""
    from app import security as sec
    sec.check_url.cache_clear()
    for bad, msg in [(r"ftp://example.com/x.tif", r"only http\(s\)"),
                     ("http:///x.tif", "missing host"),
                     ("https://does-not-exist.invalid/x.tif", "cannot resolve")]:
        with pytest.raises(ValueError, match=msg):
            sec.check_url(bad)
        sec.check_url.cache_clear()
    # loopback fixtures are only reachable because ALLOW_PRIVATE_NET=1 is set
    # by conftest; with it off, the very same host is a refusal
    monkey_saved = sec.ALLOW_PRIVATE_NET
    sec.ALLOW_PRIVATE_NET = False
    try:
        sec.check_url.cache_clear()
        with pytest.raises(ValueError, match="private or reserved"):
            sec.check_url("http://127.0.0.1:1/x.tif")
    finally:
        sec.ALLOW_PRIVATE_NET = monkey_saved
        sec.check_url.cache_clear()
