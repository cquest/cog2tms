"""Pixel semantics beyond plain RGB8: 16-bit + rescale, GDAL nodata, SampleFormat
(signed/float pixels, NaN no-data), ExtraSamples alpha, the sliced fetch of big
TileOffsets/TileByteCounts arrays, the BigTIFF layout, the sharpen gating, and the
pixel-center 1:1 warp convention."""
import math

import cv2
import numpy as np
import pytest

from app import config
from app import geotiff as g
from app import tiles
from app.httprange import LocalReader
from .conftest import (COG12, COG16, COG1L, COG1LM, COG4326, COGALPHA, COGBIG, COGCX,
                       COGF32, COGFP2, COGHOLE, COGI8, COGI16, COGMASK, COGND,
                       COGNDJUNK, COGNDNEG, COGPRED3, COGROT, COGSUB, W, fixture_tile,
                       write_cog)


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


def test_bigtiff_fixture(fixtures_dir):
    """BigTIFF (version 43: 64-bit offsets and counts, LONG8 block index, values
    inline up to 8 bytes) - the layout gdal_translate switches to past 4 GB, and
    the one both 4 GB+ reference COGs use. Pinned offline: the file really is
    version 43, the reader parses it as such, and the same content renders
    bit-identical to the classic-TIFF twin (LONG8 index and mask included)."""
    assert (fixtures_dir / COGBIG).open("rb").read(4) == b"II+\x00"  # magic 43
    cog = _open(fixtures_dir, COGBIG)
    try:
        assert cog.abis == 8  # the IFD parser widened entries to 20 bytes
        assert len(cog.levels) == 2 and cog.levels[1].width == W // 2
        assert [(m.width, m.height) for m in cog.masks] == [(W, W)]
        rgb, valid = tiles.render_tile(cog, *fixture_tile(), 256)
    finally:
        cog.close()
    rrgb, rvalid = _render(fixtures_dir, COGMASK)
    assert (rgb == rrgb).all() and (valid == rvalid).all()


def test_synth_overviews_match_the_cogs_own_pyramid(fixtures_dir):
    """A COG written without internal overviews (level 0 alone) renders the same
    tiles as its twin that has them: CogFile.synth_to reads level 0 once - blocks
    are jointive, so a whole level is a single byte-range - and halves it by area
    average, each level from the previous one, down to the SYNTH_MIN_PX floor -
    256 px, one tile at the lowest zoom the tilejson then advertises. That is what
    the COG's own overview holds, and coverage stays exactly the same."""
    ref, one = _open(fixtures_dir, COG4326), _open(fixtures_dir, COG1L)
    try:
        assert [l.width for l in one.levels] == [W]  # no pyramid to start with
        for z in (12, 11, 10, 9, 8):
            a = tiles.render_tile(ref, *fixture_tile(z=z), 256)
            b = tiles.render_tile(one, *fixture_tile(z=z), 256)
            assert (a[1] == b[1]).all()  # coverage identical at every zoom
            diff = np.abs(a[0].astype(int) - b[0].astype(int))
            # the fixture's own overview is a resampled pattern, not an average, so
            # the two pyramids differ by rounding only (mean <= 0.07, max 2 measured)
            assert diff.mean() < 0.1 and diff.max() <= 2
            # one pass, all the way to the floor: 512 -> 256, then it stops -
            # a narrower level would be upscaled into a 256 px tile
            assert [s.width for s in one.synth] == [256]
    finally:
        ref.close()
        one.close()


def test_synth_stops_at_the_floor(fixtures_dir, monkeypatch):
    """SYNTH_MIN_PX is where the derived pyramid ends, and the level pick sees the
    whole chain at once (not one level per request). A COG whose own coarsest level
    already reaches the floor gets nothing derived: below it a level is upscaled
    into a tile, which buys bytes, not detail."""
    assert [g.synth_floor_px(w) for w in (512, 1024, 1042, 260, 256, 128)] == \
        [256, 256, 131, 130, 256, 128]  # halving rounds up, like _halve
    with monkeypatch.context() as m:  # the chain follows the knob
        m.setattr(config, "SYNTH_MIN_PX", 64)
        assert g.synth_floor_px(1042) == 33  # 1042 -> 521 -> 261 -> 131 -> 66 -> 33
    cog = _open(fixtures_dir, COG1L)
    try:
        assert cog.synth_to() and [s.width for s in cog.synth] == [256]
        assert [l.width for l in cog.levels] == [W, 256]  # visible to the pick
        assert cog.synth_to() and len(cog.synth) == 1  # idempotent
    finally:
        cog.close()
    twin = _open(fixtures_dir, COG4326)  # its own pyramid already reaches 256
    try:
        assert not twin.synth_to() and not twin.synth
    finally:
        twin.close()


def test_synth_is_refused_above_the_gate(fixtures_dir, monkeypatch):
    """SYNTH_MAX_PX bounds the one contiguous read the synthesis needs: level 0
    bigger than the gate is never walked (an unbounded pass over a huge raster
    would be an outage, not a slow tile). 0 turns the feature off."""
    cog = _open(fixtures_dir, COG1L)
    try:
        monkeypatch.setattr(config, "SYNTH_MAX_PX", 0)
        assert not cog.synth_to() and not cog.synth
        monkeypatch.setattr(config, "SYNTH_MAX_PX", W * W - 1)
        assert not cog.synth_to() and not cog.synth
        monkeypatch.setattr(config, "SYNTH_MAX_PX", W * W)
        assert cog.synth_to() and [s.width for s in cog.synth] == [256]
    finally:
        cog.close()


def test_halve_weights_the_average_by_coverage():
    """An invalid pixel feeds neither the value sum nor the count: a half-covered
    output pixel keeps the brightness of its valid half, where a plain average
    would darken it by the black of its no-data part (the merger-hole lesson).
    Odd edges average what is there instead of padding with black."""
    data, valid = np.zeros((2, 2, 3), np.uint8), np.zeros((2, 2), bool)
    data[0, 0] = 200
    valid[0, 0] = True
    out, ov = g._halve(data, valid)
    assert ov.tolist() == [[True]] and out[0, 0].tolist() == [200, 200, 200]
    out, ov = g._halve(np.zeros((2, 2, 3), np.uint8), np.zeros((2, 2), bool))
    assert not ov.any() and not out.any()  # no coverage stays no coverage
    data, valid = np.full((3, 3, 1), 7, np.uint8), np.ones((3, 3), bool)
    out, ov = g._halve(data, valid)  # odd size: 2x2 output, no black row/column
    assert out.shape == (2, 2, 1) and ov.all() and (out == 7).all()


def test_synth_levels_carry_the_mask(fixtures_dir):
    """The pyramid is built from window() of level 0, so the 1-bit internal mask is
    already applied when the levels are halved: coverage survives zooming out.
    (The fixture with its own overview has the mask on level 0 only - masks match
    by dims - so the derived path is the more faithful of the two, not the less.)"""
    masked, plain = _open(fixtures_dir, COG1LM), _open(fixtures_dir, COG1L)
    try:
        t = fixture_tile(z=12)
        a, b = tiles.render_tile(masked, *t, 256), tiles.render_tile(plain, *t, 256)
        assert 0 < a[1].sum() < b[1].sum()  # the masked third stays transparent
        assert not (a[1] & ~b[1]).any()  # ... and never the other way round
    finally:
        masked.close()
        plain.close()


def test_synth_levels_are_per_reader(fixtures_dir):
    """Derived levels live on the reader, so a purge + reopen (a COG replaced under
    one URL) cannot mix them with the new file's pixels."""
    a = _open(fixtures_dir, COG1L)
    b = _open(fixtures_dir, COG1L)
    try:
        assert a.synth_to() and a.synth
        assert not b.synth and [l.width for l in b.levels] == [W]
    finally:
        a.close()
        b.close()


def test_window_ceiling_is_a_clean_error(fixtures_dir, monkeypatch):
    """MAX_WINDOW_PX: before the clamp, a tile whose window reached the whole
    raster hit OpenCV's SHRT_MAX assertion in remap or a 272 GB allocation - both
    escape as a 500 and can take the worker down. Now a CogError (a 404 at the
    endpoint), and the clamp itself changes no pixel: an outside-window sample
    contributes 0 to value and coverage, exactly as the old padding did."""
    cog = _open(fixtures_dir, COG1L)
    try:
        monkeypatch.setattr(config, "MAX_WINDOW_PX", 1000)
        with pytest.raises(g.CogError, match="over MAX_WINDOW_PX"):
            tiles.render_tile(cog, *fixture_tile(), 256)
    finally:
        cog.close()


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
