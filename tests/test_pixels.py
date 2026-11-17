"""Pixel semantics beyond plain RGB8: 16-bit + rescale, GDAL nodata, SampleFormat
(signed/float pixels, NaN no-data), ExtraSamples alpha, the sliced fetch of big
TileOffsets/TileByteCounts arrays, the sharpen gating, and the pixel-center 1:1
warp convention."""
import math

import cv2
import numpy as np
import pytest

from app import config
from app import geotiff as g
from app import tiles
from app.httprange import LocalReader
from .conftest import (COG12, COG16, COG4326, COGALPHA, COGCX, COGF32, COGFP2,
                       COGHOLE, COGI8, COGI16, COGMASK, COGND, COGNDJUNK, COGNDNEG,
                       COGPRED3, COGROT, COGSUB, W, fixture_tile, write_cog)


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


def test_nodata_out_of_range_renders(fixtures_dir):
    """GDAL_NODATA=-9999 declared on unsigned 8-bit samples: no pixel can hold
    that value, so it matches nothing and the tile renders - a reader must not
    raise (numpy 2 casts refuse, they do not wrap)."""
    cog = _open(fixtures_dir, COGNDNEG)
    try:
        assert cog.nodata == [-9999.0]
        data, valid = cog.window(0, 0, 0, 4, 4)
        assert data.dtype == np.uint8 and valid.all()
    finally:
        cog.close()


def test_int16_signed_nodata(fixtures_dir):
    """SampleFormat 2 (signed) with bps 16: the DEM's -9999 sentinel becomes
    no-coverage while the rest keeps its values, and rescale=0,255 restores the
    8-bit fixture pixel-for-pixel."""
    cog = _open(fixtures_dir, COGI16)
    try:
        assert cog.levels[0].dtype == "<i2"
        data, valid = cog.window(0, 0, 0, 4, 4)
        assert data.dtype == np.int16 and data[0, 0, 0] == -9999
        assert not valid[:, 0].any() and valid[:, 1:].all()
        with pytest.raises(tiles.MissingRescale):
            tiles.render_tile(cog, *fixture_tile(), 256)
        rgb, valid = tiles.render_tile(cog, *fixture_tile(16), 256, rescale=(0.0, 255.0))
    finally:
        cog.close()
    ref = _open(fixtures_dir, COG4326)
    try:
        rrgb, rvalid = tiles.render_tile(ref, *fixture_tile(16), 256)
    finally:
        ref.close()
    assert valid.all() and rvalid.all()
    # the gray bands carry the red channel of the RGB fixture (its blue is 0)
    assert (rgb[..., 0] == rrgb[..., 0]).all()


def test_int8_requires_rescale(fixtures_dir):
    """SampleFormat 2 with bps 8: signed bytes are not something Pillow can
    encode, so the rescale gate must catch them - not only multi-byte samples."""
    with pytest.raises(tiles.MissingRescale):
        _render(fixtures_dir, COGI8)


def test_float32_nan_nodata(fixtures_dir):
    """float32 samples (SampleFormat 3) with GDAL_NODATA="nan": the sentinel
    column is no-coverage. A NaN nodata can never be found by == (NaN != NaN),
    it needs an isnan test; and floats stay without a natural range, so the
    render still demands rescale=min,max."""
    cog = _open(fixtures_dir, COGF32)
    try:
        assert math.isnan(cog.nodata[0])
        data, valid = cog.window(0, 0, 0, 4, 4)
        assert data.dtype == np.float32 and math.isnan(data[0, 0, 0])
        assert not valid[:, 0].any() and valid[:, 1:].all()
        with pytest.raises(tiles.MissingRescale):
            tiles.render_tile(cog, *fixture_tile(), 256)
    finally:
        cog.close()


@pytest.mark.parametrize("name,why", [(COG12, r"BitsPerSample 12"),
                                      (COGCX, r"SampleFormat 4"),
                                      (COGPRED3, r"predictor 3"),
                                      (COGFP2, r"predictor 2 needs"),
                                      (COGNDJUNK, r"bad GDAL_NODATA")])
def test_unreadable_samples_refused_at_open(fixtures_dir, name, why):
    """Depths, encodings and sentinel declarations we cannot honour are refused
    when the file opens: bps 12 used to fall back to 8-bit silently, which
    renders wrong pixels, and a sentinel read as 0.0 would eat real coverage."""
    with pytest.raises(g.CogError, match=why):
        _open(fixtures_dir, name)


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


def test_sharpen_gates(fixtures_dir, monkeypatch):
    """Overview tiles get the unsharp mask (anti-halo: only pixels whose 3x3
    neighborhood is fully valid are touched, so no liseré along the footprint),
    and full-resolution tiles are never modified."""
    cog = _open(fixtures_dir, COG4326)
    try:
        z, x, y = fixture_tile(12)  # there the 2x overview is the never-upscale pick
        monkeypatch.setattr(config, "SHARPEN", 0.0)
        raw, v = tiles.render_tile(cog, z, x, y, 256)
        monkeypatch.setattr(config, "SHARPEN", 0.5)
        sharp, _ = tiles.render_tile(cog, z, x, y, 256)
        assert v.any() and (sharp != raw).any()
        inner = cv2.erode(v.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        assert (sharp[~inner] == raw[~inner]).all()  # untouched along the edges
        z, x, y = fixture_tile()  # z14: level 0 is the pick -> gate closed
        a, _ = tiles.render_tile(cog, z, x, y, 256)
        monkeypatch.setattr(config, "SHARPEN", 0.0)
        b, _ = tiles.render_tile(cog, z, x, y, 256)
        assert (a == b).all()
    finally:
        cog.close()


def test_pixel_center_sampling_1to1(tmp_path):
    """Raster georeferenced exactly 1:1 on the tile matrix (tile px == source
    px, footprint corner == tile corner): the render must reproduce the source
    pixels bit-for-bit. Pins both the sampling convention (remap coordinates
    are pixel centers, ox/oy are GeoTIFF corners -> no corner low-pass) and
    the never-upscale overview rule (a 2x overview window of 128 px is refused
    at 1:1, level 0 is used)."""
    z = 13
    x, y = 2200, 4400  # arbitrary tile of that matrix row/column
    tx0, ty0, tx1, ty1 = tiles.tile_web_bounds(z, x, y)
    psx = (tx1 - tx0) / 256.0
    write_cog(tmp_path / "align.tif", tx0, ty1, psx, psx, geographic=False)
    cog = g.CogFile(LocalReader(str(tmp_path / "align.tif")))
    try:
        src, valid = cog.window(0, 0, 0, 256, 256)
        rgb, v = tiles.render_tile(cog, z, x, y, 256)
        assert v.all()
        assert (rgb == src).all()
    finally:
        cog.close()
