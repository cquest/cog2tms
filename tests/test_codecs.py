"""Block decoders: JPEG-in-TIFF blocks, simplejpeg/Pillow parity, threaded window decode."""
import threading

import numpy as np
import pytest

from app import geotiff as g
from app import tiles
from app.httprange import LocalReader
from .conftest import COG4326, COGJPEG, COGJPEG1, fixture_tile

POOL = g._POOL  # the pool as configured, to put it back inside tests


def _render(fixtures_dir, name):
    cog = g.CogFile(LocalReader(str(fixtures_dir / name)))
    try:
        return tiles.render_tile(cog, *fixture_tile(), 256)
    finally:
        cog.close()


def _rgb_valid(rgb, valid):
    """rgb with the invalid pixels zeroed, so both render paths are comparable."""
    return np.dstack([rgb, valid[..., None].astype(np.uint8) * 255])


def test_jpeg_fixture_renders(fixtures_dir):
    """A JPEG-in-TIFF COG renders like the deflate fixture (lossy, so close)."""
    ref, refv = _render(fixtures_dir, COG4326)
    rgb, valid = _render(fixtures_dir, COGJPEG)
    assert valid.sum() == refv.sum() > 0
    diff = np.abs(rgb.astype(int) - ref.astype(int))
    assert diff.mean() < 2 and diff.max() < 16  # measured: 0.3 / 3


def test_jpeg_fixture_north_up(fixtures_dir):
    """The fixture's red square sits north: it must be in the tile's upper half."""
    rgb, _ = _render(fixtures_dir, COGJPEG)
    red = (rgb[..., 0] > 200) & (rgb[..., 1] < 60) & (rgb[..., 2] < 60)
    ys = np.nonzero(red.any(axis=1))[0]
    assert len(ys) > 10 and ys.mean() < 128


def test_gray_jpeg_fixture(fixtures_dir):
    """Single-band JPEG blocks (GRAY): decoded once, replicated over 3 bands."""
    ref, _ = _render(fixtures_dir, COG4326)
    rgb, valid = _render(fixtures_dir, COGJPEG1)
    assert valid.all()
    assert (rgb[..., 0] == rgb[..., 1]).all() and (rgb[..., 1] == rgb[..., 2]).all()
    # the fixture's red channel is its gray pattern (measured max diff: 1)
    assert np.abs(rgb[..., 0].astype(int) - ref[..., 0].astype(int)).max() < 16


@pytest.mark.skipif(g._sjpeg is None, reason="simplejpeg is not installed")
def test_simplejpeg_matches_pillow(fixtures_dir, monkeypatch):
    """Swapping the JPEG decoder must not move the pixels (measured: identical)."""
    sj, sjv = _render(fixtures_dir, COGJPEG)
    gs, _ = _render(fixtures_dir, COGJPEG1)
    monkeypatch.setattr(g, "_sjpeg", None)
    pj, pjv = _render(fixtures_dir, COGJPEG)
    gp, _ = _render(fixtures_dir, COGJPEG1)
    assert (sjv == pjv).all()
    for a, b in ((sj, pj), (gs, gp)):
        assert np.abs(a.astype(int) - b.astype(int)).max() <= 2


@pytest.mark.skipif(g._POOL is None, reason="DECODE_THREADS=1")
def test_threaded_decode_matches_inline(fixtures_dir, monkeypatch):
    """Decoding blocks in threads gives byte-identical windows."""
    monkeypatch.setattr(g, "_PAR_MIN_PX", 0)  # the fixture windows are small
    for name in (COG4326, COGJPEG):  # deflate, and JPEG when simplejpeg is present
        monkeypatch.setattr(g, "_POOL", None)
        inline = _rgb_valid(*_render(fixtures_dir, name))
        monkeypatch.setattr(g, "_POOL", POOL)
        pooled = _rgb_valid(*_render(fixtures_dir, name))
        assert (inline == pooled).all(), name


def test_threaded_decode_needs_a_gil_free_codec_and_a_big_window():
    """Threads only pay off when the decoder releases the GIL and the window is big."""
    lvl = g._Level.__new__(g._Level)
    lvl.tilew = lvl.tileh = 512
    for comp, expect in ((1, False), (5, False), (32773, False), (8, True), (32946, True)):
        lvl.compression = comp
        assert g._threaded_decode(lvl, 4) is expect
    lvl.compression = 7
    assert g._threaded_decode(lvl, 4) is (g._sjpeg is not None)
    assert not g._threaded_decode(lvl, 1)  # a single block: nothing to overlap
    lvl.compression = 8
    lvl.tilew = lvl.tileh = 128
    assert not g._threaded_decode(lvl, 4)  # 64 kpx: inline decoding is cheaper


@pytest.mark.skipif(g._POOL is None, reason="DECODE_THREADS=1")
def test_blocks_decode_in_pool_threads(fixtures_dir, monkeypatch):
    """The blocks of one window really are decoded in the shared pool."""
    threads = set()
    orig = g.CogFile._decode

    def spy(self, lvl, raw):
        threads.add(threading.current_thread().name)
        return orig(self, lvl, raw)

    monkeypatch.setattr(g.CogFile, "_decode", spy)
    monkeypatch.setattr(g, "_PAR_MIN_PX", 0)  # the 4 x 128^2 fixture window is small
    _render(fixtures_dir, COGJPEG if g._sjpeg is not None else COG4326)
    assert any(t.startswith("cog-decode") for t in threads), threads
