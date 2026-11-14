"""WebP output and format=auto negotiation: the Accept header selects webp only
when image/webp is listed explicitly (image/* and */* never do - JOSM sends
them and cannot decode webp); the source codec plays no role."""
import io

import numpy as np
import pytest
from PIL import Image

from app import config
from app.main import clear_caches
from .conftest import COG4326, fixture_tile, tile_url

CHROME_ACCEPT = "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8"


def _pix(r):
    im = Image.open(io.BytesIO(r.content))
    im.load()
    return np.asarray(im)


def _auto(client, server, name, fmt="auto", accept=None, **q):
    z, x, y = fixture_tile()
    headers = {"Accept": accept} if accept is not None else {}
    return client.get(tile_url(server, name, z, x, y, format=fmt, **q), headers=headers)


def test_webp_tile(client, server):
    """format=webp returns a valid lossy webp: pixels within codec distance of
    the png reference. The fixture center tile is fully opaque, and Pillow
    drops the alpha channel of fully-opaque webp - harmless, browsers read it
    as opaque. Transparency is pinned separately."""
    z, x, y = fixture_tile()
    png = _pix(client.get(tile_url(server, COG4326, z, x, y)))
    r = client.get(tile_url(server, COG4326, z, x, y, format="webp"))
    assert r.status_code == 200 and r.headers["content-type"] == "image/webp"
    w = _pix(r)
    assert w.shape[:2] == (256, 256)
    # saturated fixture colors (pure red/0) are webp's worst case: probe max 40
    assert np.abs(w[..., :3].astype(int) - png[..., :3]).max() < 48


def test_webp_lossless_is_bit_exact(client, server, monkeypatch):
    """lossless+exact webp carries alpha through untouched and matches the png
    pixels to the bit."""
    monkeypatch.setattr(config, "WEBP_LOSSLESS", True)
    clear_caches()
    z, x, y = fixture_tile()
    a = _pix(client.get(tile_url(server, COG4326, z, x, y, format="webp", blackmax=200)))
    b = _pix(client.get(tile_url(server, COG4326, z, x, y, format="png", blackmax=200)))
    assert a.shape == (256, 256, 4) and a[..., 3].min() == 0  # blackmax -> real alpha
    assert (a == b).all()
    monkeypatch.setattr(config, "WEBP_LOSSLESS", False)
    clear_caches()


def test_empty_webp_tile(client, server):
    """Out-of-bounds webp tile: transparent (alpha=0), like the png empty tile."""
    z, x, y = fixture_tile()
    r = client.get(tile_url(server, COG4326, z, x + 40, y, format="webp"))
    assert r.status_code == 200 and r.headers["content-type"] == "image/webp"
    assert _pix(r)[..., 3].max() == 0


@pytest.mark.parametrize("accept,expected", [
    (CHROME_ACCEPT, "image/webp"),   # browsers list image/webp explicitly
    ("image/webp", "image/webp"),    # minimal honest client
    ("image/webp;q=0", "image/png"),  # q=0 must refuse
    ("image/webp;q=0.3", "image/webp"),
    ("*/*", "image/png"),            # JOSM / plain HTTP clients: never webp
    ("image/*", "image/png"),        # wildcards never either
    ("image/png", "image/png"),
    ("garbage", "image/png"),        # malformed header: honest fallback
    (None, "image/png"),             # no header at all
])
def test_auto_accept_negotiation(client, server, accept, expected):
    r = _auto(client, server, COG4326, accept=accept)
    assert r.status_code == 200 and r.headers["content-type"] == expected


def test_auto_out_of_bounds_takes_accept_codec(client, server):
    """Empty tiles follow the negotiated codec too: transparent webp to a
    webp-capable client, never a jpeg EMPTY_TILE_BG fill."""
    z, x, y = fixture_tile()
    r = client.get(tile_url(server, COG4326, z, x + 40, y, format="auto"),
                   headers={"Accept": CHROME_ACCEPT})
    assert r.status_code == 200 and r.headers["content-type"] == "image/webp"
    assert _pix(r)[..., 3].max() == 0


def test_auto_bad_format_still_400(client, server):
    assert client.get(tile_url(server, COG4326, *fixture_tile(), format="bogus")).status_code == 400


def test_auto_keys_cache_per_accept(client, server, counter):
    """The resolved format is part of the cache key: one Accept renders and
    caches once (identical bytes, no re-render), a different Accept is a
    different entry with its own bytes."""
    r1 = _auto(client, server, COG4326, accept=CHROME_ACCEPT)
    r2 = _auto(client, server, COG4326, accept=CHROME_ACCEPT)
    assert r2.content == r1.content and r1.headers["content-type"] == "image/webp"
    r3 = _auto(client, server, COG4326, accept="*/*")
    assert r3.headers["content-type"] == "image/png"
    assert r3.content != r1.content  # separate cache entry, re-encoded
