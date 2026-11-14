"""Block decoders: JPEG-in-TIFF blocks, simplejpeg/Pillow parity, imagecodecs
LZW/PackBits/ZSTD, threaded window decode."""
import threading

import numpy as np
import pytest

from app import geotiff as g
from app import tiles
from app.httprange import LocalReader
from .conftest import (COG4326, COGLZW, COGLZWLIB, COGJPEG, COGJPEG1, COGPACK,
                       COGPRED, COGZSTD, fixture_tile)

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
    lzw_par = g._ic is not None  # imagecodecs LZW releases the GIL, ours is Python
    for comp, expect in ((1, False), (5, lzw_par), (32773, False), (8, True),
                         (32946, True), (50000, False)):
        lvl.compression = comp
        assert g._threaded_decode(lvl, 4) is expect
    lvl.compression = 7
    assert g._threaded_decode(lvl, 4) is (g._sjpeg is not None)
    assert not g._threaded_decode(lvl, 1)  # a single block: nothing to overlap
    lvl.compression = 8
    lvl.tilew = lvl.tileh = 128
    assert not g._threaded_decode(lvl, 4)  # 64 kpx: inline decoding is cheaper


@pytest.mark.skipif(g._ic is None, reason="imagecodecs is not installed")
def test_imagecodecs_lzw_and_packbits_blocks(monkeypatch):
    """LZW/PackBits blocks decode through imagecodecs; byte-identical results."""
    import imagecodecs as ic
    rng = np.random.default_rng(3)
    blk = np.ascontiguousarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))
    cog = g.CogFile.__new__(g.CogFile)
    blobs = {5: ic.lzw_encode(blk.tobytes()), 32773: ic.packbits_encode(blk.tobytes())}
    for comp, raw in blobs.items():
        a = cog._decode(_lvl(comp), raw)
        assert (a == blk).all(), comp
        if comp == 32773:  # TIFF PackBits is standard: our Python decoder too
            monkeypatch.setattr(g, "_ic", None)
            assert (cog._decode(_lvl(comp), raw) == blk).all()


@pytest.mark.skipif(g._ic is None, reason="imagecodecs is not installed")
def test_lzw_decodes_libtiff_variant():
    """GDAL/libtiff emit an LZW variant our pure-Python decoder rejects ("bad
    LZW code"); imagecodecs reads it - the reason LZW decoding goes through it."""
    import io
    import struct
    from PIL import Image
    yy, xx = np.mgrid[:64, :64]
    blk = np.dstack([(xx * 7) % 256, (yy * 13) % 256, np.full((64, 64), 90)]).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(blk).save(buf, "TIFF", compression="tiff_lzw")
    tiff = buf.getvalue()
    off = struct.unpack("<I", tiff[4:8])[0]
    tags = {}
    for i in range(struct.unpack("<H", tiff[off:off + 2])[0]):
        e = tiff[off + 2 + 12 * i: off + 14 + 12 * i]
        code, typ, cnt = struct.unpack("<HHI", e[:8])
        tags[code] = (typ, cnt, e[8:12])
    assert struct.unpack("<H", tags[259][2][:2])[0] == 5  # libtiff wrote LZW
    ptr = struct.unpack("<I", tags[273][2])[0]  # one strip: the whole block
    size = struct.unpack("<I", tags[279][2])[0]
    cog = g.CogFile.__new__(g.CogFile)
    assert (cog._decode(_lvl(5), tiff[ptr:ptr + size]) == blk).all()


@pytest.mark.skipif(g._ic is None, reason="imagecodecs is not installed")
def test_zstd_fixture_renders(fixtures_dir):
    """A ZSTD COG renders exactly like the deflate fixture (both lossless)."""
    ref = _rgb_valid(*_render(fixtures_dir, COG4326))
    zstd = _rgb_valid(*_render(fixtures_dir, COGZSTD))
    assert (zstd == ref).all()


@pytest.mark.skipif(g._ic is None, reason="imagecodecs is not installed")
def test_zstd_blocks(monkeypatch):
    """Plain zstd frame blocks decode byte-exact; without imagecodecs the block
    is refused (no pure-Python zstd decoder)."""
    import imagecodecs as ic
    rng = np.random.default_rng(4)
    blk = np.ascontiguousarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))
    cog = g.CogFile.__new__(g.CogFile)
    assert (cog._decode(_lvl(50000), ic.zstd_encode(blk.tobytes())) == blk).all()
    monkeypatch.setattr(g, "_ic", None)
    with pytest.raises(g.CogError):
        cog._decode(_lvl(50000), ic.zstd_encode(blk.tobytes()))


@pytest.mark.skipif(g._ic is None, reason="imagecodecs is not installed")
def test_zstd_multisegment_blocks():
    """libtiff writes big ZSTD strips as segments each prefixed with its
    big-endian compressed size; they decode like a single frame."""
    import imagecodecs as ic
    rng = np.random.default_rng(5)
    blk = np.ascontiguousarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))
    frame = ic.zstd_encode(blk.tobytes())
    two = blk.tobytes() + blk.tobytes()  # a block made of two equal halves
    blob = len(frame).to_bytes(4, "big") + frame + len(frame).to_bytes(4, "big") + frame
    assert g._zstd_decode(blob) == two
    lvl = _lvl(50000)
    lvl.tileh = 128  # two stacked 64-row segments = one 128x64 block
    lvl.tilew = 64
    cog = g.CogFile.__new__(g.CogFile)
    a = cog._decode(lvl, blob)
    assert a.shape == (128, 64, 3) and (a[:64] == blk).all() and (a[64:] == blk).all()
    with pytest.raises(g.CogError):  # a mangled segment table must not read past the end
        g._zstd_decode(len(frame).to_bytes(4, "big") + frame[:8])


def test_pure_packbits_decode_golden():
    """count<128: literal copy of count+1 bytes; >128: repeat 257-count; 128: no-op."""
    raw = bytes([2, 10, 20, 30]) + bytes([0xFB, 0x42]) + bytes([0x80]) + bytes([0, 99])
    assert g._packbits_decode(raw) == bytes([10, 20, 30] + [0x42] * 6 + [99])


def test_pure_lzw_decoder_roundtrip():
    """The pure-Python decoder and the conftest mirror-encoder must agree: this
    pins the one dialect that fallback can read (libtiff's needs imagecodecs).
    The sizes cross the 512/1024 code-width changes and the 4096 clear restart."""
    from .conftest import _lzw_encode, pattern
    blobs = (b"Hi!", b"", b"ABABABAB" * 700, pattern().tobytes()[: 128 * 128 * 3],
             np.zeros(100_000, np.uint8).tobytes(),
             np.random.default_rng(1).integers(0, 256, 60_000, dtype=np.uint8).tobytes())
    for blob in blobs:
        assert g._lzw_decode(_lzw_encode(blob)) == blob, len(blob)


def test_pure_lzw_decoder_rejects_unknown_code():
    """9 bits with value 300 while the table holds only up to 258."""
    with pytest.raises(g.CogError, match="bad LZW code"):
        g._lzw_decode(b"\x96\x00")


def test_predictor2_fixture_renders(fixtures_dir):
    """Deflate + predictor 2 (horizontal differencing): after the modular cumsum
    the blocks are bit-identical to the plain deflate fixture."""
    ref = _rgb_valid(*_render(fixtures_dir, COG4326))
    assert (_rgb_valid(*_render(fixtures_dir, COGPRED)) == ref).all()


@pytest.mark.skipif(g._ic is None, reason="imagecodecs is not installed")
def test_packbits_fixture_renders(fixtures_dir, monkeypatch):
    """A PackBits COG renders like the deflate fixture, through imagecodecs
    (standard variant) and through our pure-Python decoder when it is absent."""
    ref = _rgb_valid(*_render(fixtures_dir, COG4326))
    assert (_rgb_valid(*_render(fixtures_dir, COGPACK)) == ref).all()
    monkeypatch.setattr(g, "_ic", None)
    assert (_rgb_valid(*_render(fixtures_dir, COGPACK)) == ref).all()


def test_lzw_variants_end_to_end(fixtures_dir, monkeypatch):
    """imagecodecs reads the libtiff-written blocks; with it absent only the
    blocks written in _lzw_decode's own dialect render - the other variant is
    refused (known limit, see AGENTS)."""
    if g._ic is not None:
        ref = _rgb_valid(*_render(fixtures_dir, COG4326))
        assert (_rgb_valid(*_render(fixtures_dir, COGLZWLIB)) == ref).all()
    monkeypatch.setattr(g, "_ic", None)
    ref = _rgb_valid(*_render(fixtures_dir, COG4326))
    assert (_rgb_valid(*_render(fixtures_dir, COGLZW)) == ref).all()
    with pytest.raises(g.CogError):
        _render(fixtures_dir, COGLZWLIB)


def _lvl(comp):
    lvl = g._Level.__new__(g._Level)
    lvl.tilew = lvl.tileh = 64
    lvl.samples, lvl.bps, lvl.dtype, lvl.predictor = 3, 8, np.dtype("u1"), 0
    lvl.compression = comp
    return lvl


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
