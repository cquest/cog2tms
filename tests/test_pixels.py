"""Pixel semantics beyond plain RGB8: 16-bit + rescale, GDAL nodata, ExtraSamples
alpha, and the sliced fetch of big TileOffsets/TileByteCounts arrays."""
import numpy as np
import pytest

from app import geotiff as g
from app import tiles
from app.httprange import LocalReader
from .conftest import (COG16, COG4326, COGALPHA, COGHOLE, COGMASK, COGND, COGROT,
                       COGSUB, W, fixture_tile)


def _open(fixtures_dir, name):
    return g.CogFile(LocalReader(str(fixtures_dir / name)))


def _render(fixtures_dir, name, **kw):
    cog = _open(fixtures_dir, name)
    try:
        return tiles.render_tile(cog, *fixture_tile(), 256, **kw)
    finally:
        cog.close()


def test_16bit_requires_rescale(fixtures_dir):
    """16-bit data would clip to 0 without an explicit window: refuse it."""
    with pytest.raises(tiles.MissingRescale):
        _render(fixtures_dir, COG16)


def test_16bit_rescale_exact(fixtures_dir):
    """The fixture is the 8-bit pattern scaled by exactly 257: rescale=min,max
    restores it bit-for-bit (257*255 = 65535)."""
    rgb, valid = _render(fixtures_dir, COG16, rescale=(0.0, 65535.0))
    ref, _ = _render(fixtures_dir, COG4326)
    assert valid.sum() > 0
    assert (rgb[..., 0] == ref[..., 0]).all()  # gray == pattern red channel
    assert (rgb[..., 0] == rgb[..., 1]).all() and (rgb[..., 1] == rgb[..., 2]).all()
    cog = _open(fixtures_dir, COG16)
    try:
        data, _ = cog.window(0, 0, 0, 4, 4)
        assert data.dtype == np.uint16
    finally:
        cog.close()


def test_nodata_fixture(fixtures_dir):
    """GDAL_NODATA="0,0,0": only pixels matching on every band are no-coverage;
    a red pixel (two channels equal 0) must survive (GDAL parity)."""
    cog = _open(fixtures_dir, COGND)
    try:
        assert cog.nodata == [0.0, 0.0, 0.0]
        _, valid = cog.window(0, 0, 0, 4, 4)
        assert not valid[0, 0] and valid[3, 3]
        _, valid = cog.window(0, 200, 200, 202, 202)
        assert valid.all()  # the red square keeps its zero channels
    finally:
        cog.close()


def test_extra_samples_alpha(fixtures_dir):
    """ExtraSamples band: alpha 0 pixels become no-coverage, band 4 is dropped."""
    cog = _open(fixtures_dir, COGALPHA)
    try:
        assert cog.extra_alpha
        data, valid = cog.window(0, 200, 200, 203, 203)  # the transparent hole
        assert data.shape[2] == 3 and not valid.any()
        data, valid = cog.window(0, 100, 100, 103, 103)  # outside the hole
        assert valid.all()
    finally:
        cog.close()


def test_sliced_block_index(fixtures_dir, monkeypatch):
    """Index arrays above _SLICE_AT are fetched per window as tight slices
    instead of once whole: lower the threshold and demand identical pixels."""
    cog = _open(fixtures_dir, COG4326)
    try:  # normal path first: the fixture arrays (16 x 4 B) stay whole-loaded
        whole = tiles.render_tile(cog, *fixture_tile(), 256)
        assert cog.levels[0].offsets is not None
    finally:
        cog.close()
    monkeypatch.setattr(g, "_SLICE_AT", 16)
    cog = _open(fixtures_dir, COG4326)
    try:
        rgb, valid = tiles.render_tile(cog, *fixture_tile(), 256)
        assert cog.levels[0].offsets is None  # the window used the slice path
    finally:
        cog.close()
    wrgb, wvalid = whole
    assert (rgb == wrgb).all() and (valid == wvalid).all()


def test_internal_mask_fixture(fixtures_dir):
    """NewSubfileType & 4: the 1-bit mask levels the raster where it is 0
    (west third here). Pixels stay untouched, only coverage changes; the mask
    has full size, so it applies to level 0 only."""
    cog = _open(fixtures_dir, COGMASK)
    ref = _open(fixtures_dir, COG4326)
    try:
        assert len(cog.masks) == 1
        _, vm = cog.window(0, 10, 100, 14, 104)  # cols 10..13 < W/3: masked out
        _, vr = ref.window(0, 10, 100, 14, 104)
        assert not vm.any() and vr.all()
        dm, vm = cog.window(0, 300, 100, 304, 104)  # east of the mask edge: kept
        dr, _ = ref.window(0, 300, 100, 304, 104)
        assert vm.all() and (dm == dr).all()
        _, vm1 = cog.window(1, 5, 50, 9, 54)  # overview has no matching-size mask
        assert vm1.all()
    finally:
        cog.close()
        ref.close()


def test_modeltransformation_refused(fixtures_dir):
    """Rotated/skewed rasters (ModelTransformation instead of tiepoint+scale)
    are refused at open, not silently read as north-up."""
    with pytest.raises(g.CogError, match="rotated rasters not supported"):
        _open(fixtures_dir, COGROT)


def test_subifd_overviews(fixtures_dir):
    """Overviews can be published through SubIFDs (tag 330) with an empty IFD
    chain (cog_merger layout): same levels, render pixel-identical."""
    cog = _open(fixtures_dir, COGSUB)
    try:
        assert len(cog.levels) == 2 and not cog.masks
        assert cog.levels[1].width == W // 2  # the overview was found
        rgb, valid = tiles.render_tile(cog, *fixture_tile(), 256)
    finally:
        cog.close()
    ref = _open(fixtures_dir, COG4326)
    try:
        rrgb, rvalid = tiles.render_tile(ref, *fixture_tile(), 256)
    finally:
        ref.close()
    assert (rgb == rrgb).all() and (valid == rvalid).all()


def test_count_zero_blocks_are_holes(fixtures_dir):
    """cog_merger unions footprints by leaving absent blocks at byte count 0
    (here the east block column, with a deliberately bogus offset): never
    fetched, and kept no-coverage -> transparent, not a black smear."""
    cog = _open(fixtures_dir, COGHOLE)
    ref = _open(fixtures_dir, COG4326)
    try:
        d, v = cog.window(0, 3 * W // 4, 10, 3 * W // 4 + 32, 42)  # holed column
        assert not v.any() and not d.any()
        d2, v2 = cog.window(0, 0, 10, 3 * W // 4, 42)  # west of the holes
        dr, vr = ref.window(0, 0, 10, 3 * W // 4, 42)
        assert (d2 == dr).all() and v2.all() and vr.all()
        rgb, valid = tiles.render_tile(cog, *fixture_tile(), 256)
        rrgb, rvalid = tiles.render_tile(ref, *fixture_tile(), 256)
    finally:
        cog.close()
        ref.close()
    # the tile's east edge maps to the holed blocks (bogus past-EOF offset):
    # coverage vanishes there, never exceeds the reference's, and kept pixels
    # keep their exact colors (renormalization, not a black smear)
    assert not valid[:, -1].any() and valid[:, :8].all() and rvalid[:, -1].all()
    assert not (valid & ~rvalid).any()
    assert (rgb[valid] == rrgb[valid]).all()
