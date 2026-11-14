"""Minimal Cloud Optimized GeoTIFF reader: classic tiled TIFF, internal overviews.

Only the tags needed to read pixel windows by byte-range are parsed. Supported:
little/big endian classic TIFF and BigTIFF, tiled layout, planar contig, no
rotation; compressions None/Deflate/LZW/PackBits/ZSTD/JPEG (JPEG-in-TIFF blocks
are spliced with JPEGTables then decoded by simplejpeg, Pillow as fallback; LZW,
PackBits and ZSTD use imagecodecs, pure-Python decoders for LZW/PackBits).
TileOffsets/TileByteCounts are fetched lazily, per overview level, so opening a
COG stays cheap.
"""
import struct
import zlib
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import numpy as np
from PIL import Image

from . import config, crs

try:  # ~3x faster than Pillow on JPEG-in-TIFF, same pixels, and releases the GIL
    from simplejpeg import decode_jpeg as _sjpeg
except ImportError:  # exotic platforms: Pillow only
    _sjpeg = None

try:  # C decoders: PackBits ~15x faster than ours; imagecodecs also decodes the
    import imagecodecs as _ic  # libtiff LZW variant that our reader would reject
except ImportError:  # slim installs: the pure-Python decoders below take over
    _ic = None


class CogError(Exception):
    pass


_TYPES = {1: ("B", 1), 2: ("c", 1), 3: ("H", 2), 4: ("I", 4), 5: ("II", 8),
          6: ("b", 1), 7: ("c", 1), 8: ("h", 2), 9: ("i", 4), 10: ("ii", 8),
          11: ("f", 4), 12: ("d", 8), 16: ("Q", 8), 18: ("q", 8)}

_NEW_SUBFILE, _W, _H, _BPS, _COMP, _PHOTOM = 254, 256, 257, 258, 259, 262
_SAMPLES, _EXTRA, _PLANAR = 277, 338, 284
_TILEW, _TILEH, _TILOFFS, _TILCNTS, _PREDICTOR = 322, 323, 324, 325, 317
_TIEPOINT, _SCALE, _MODELTRANSF = 33922, 33550, 34264
_GEOKEYS, _GEODOUBLE, _JPEGTABLES, _GDAL_NODATA = 34735, 34736, 347, 42113

_DT = {8: "u1", 16: "u2", 32: "u4"}

# TileOffsets/TileByteCounts bigger than this are fetched per window (a few
# hundred bytes) instead of whole: L0 of a big COG indexes millions of blocks
# (~6 MB of pure metadata) but a tile touches a handful of them.
_SLICE_AT = 1 << 20

# threads used to decode the blocks of one window when that is worth it (the pool
# is created on first use); see _threaded_decode
_POOL = (ThreadPoolExecutor(max_workers=config.DECODE_THREADS, thread_name_prefix="cog-decode")
         if config.DECODE_THREADS > 1 else None)

# the handoff to those threads costs ~0.3 ms: it pays from about a megapixel of
# source blocks (8 x 512^2 measured -3.1 ms, 4 x 128^2 +0.3 ms)
_PAR_MIN_PX = 1 << 20


def _threaded_decode(lvl: "_Level", n: int) -> bool:
    """Whether the n blocks of a window are worth decoding in threads."""
    if n < 2 or n * lvl.tilew * lvl.tileh < _PAR_MIN_PX:
        return False
    if lvl.compression in (6, 7):
        return _sjpeg is not None  # Pillow keeps the GIL while decoding JPEG
    if lvl.compression == 5:
        return _ic is not None  # imagecodecs LZW releases the GIL, ours is Python
    # zlib releases the GIL; imagecodecs PackBits and ZSTD do not (but are fast
    # enough inline: a whole window of them decodes in around a millisecond)
    return lvl.compression in (8, 32946)


def _scalar(end, tags, code):
    if code not in tags:
        return None
    t, _, v = tags[code]
    return struct.unpack(end + _TYPES[t][0], v[: _TYPES[t][1]])[0]


def _doubles(end, tags, code):
    if code not in tags:
        return ()
    _, c, v = tags[code]
    return struct.unpack(end + "d" * c, v[: 8 * c]) if len(v) >= 8 * c else ()


def _lzw_decode(raw: bytes) -> bytes:
    """TIFF LZW (8-bit codes, MSB-first, early code-size change)."""
    clear, eoi = 256, 257
    base = [bytes([i]) for i in range(256)]
    table, width, prev = list(base), 9, b""
    out = bytearray()
    acc = nbits = 0
    for byte in raw:
        acc = (acc << 8) | byte
        nbits += 8
        while nbits >= width:
            nbits -= width
            code = (acc >> nbits) & ((1 << width) - 1)
            if code == clear:
                table, width, prev = list(base), 9, b""
            elif code == eoi:
                return bytes(out)
            else:
                if code < len(table):
                    entry = table[code]
                elif code == len(table):
                    entry = prev + prev[:1]
                else:
                    raise CogError("bad LZW code")
                out += entry
                if prev:
                    table.append(prev + entry[:1])
                    if len(table) + 1 >= (1 << width) and width < 12:
                        width += 1
                prev = entry
    return bytes(out)


def _packbits_decode(raw: bytes) -> bytes:
    out = bytearray()
    i = 0
    while i < len(raw):
        n = raw[i]
        if n < 128:
            out += raw[i + 1: i + 2 + n]
            i += 2 + n
        elif n > 128:
            out += bytes([raw[i + 1]]) * (257 - n)
            i += 2
        else:
            i += 1
    return bytes(out)


def _zstd_decode(raw: bytes) -> bytes:
    """TIFF ZSTD (compression 50000): imagecodecs required. A block is one plain
    zstd frame, or libtiff's strip arrays of segments each prefixed with its
    big-endian compressed size; both decode to the same pixels."""
    if _ic is None:
        raise CogError("ZSTD compression requires imagecodecs")
    if raw.startswith(b"\x28\xb5\x2f\xfd"):
        return _ic.zstd_decode(raw)
    out = bytearray()
    i = 0
    while i < len(raw):
        n = int.from_bytes(raw[i:i + 4], "big")
        if n <= 0 or i + 4 + n > len(raw):
            raise CogError("truncated ZSTD block")
        out += _ic.zstd_decode(raw[i + 4:i + 4 + n])
        i += 4 + n
    return bytes(out)


class _Level:
    def __init__(self, end: str, tags: dict, reader):
        self.end = end
        self._reader = reader
        self.width = _scalar(end, tags, _W)
        self.height = _scalar(end, tags, _H)
        self.compression = _scalar(end, tags, _COMP) or 1
        self.photometric = _scalar(end, tags, _PHOTOM) or 0
        self.samples = _scalar(end, tags, _SAMPLES) or 1
        self.planar = _scalar(end, tags, _PLANAR) or 1
        self.predictor = _scalar(end, tags, _PREDICTOR) or 1
        self.bps = _scalar(end, tags, _BPS) or 8
        self.tilew = _scalar(end, tags, _TILEW)
        self.tileh = _scalar(end, tags, _TILEH)
        self.jpegtables = tags.get(_JPEGTABLES, (0, 0, b""))[2] or None
        if not (self.width and self.height and self.tilew and self.tileh):
            raise CogError("not a tiled GeoTIFF")
        if self.planar != 1:
            raise CogError("planarconfig separate not supported")
        if _TILOFFS not in tags or _TILCNTS not in tags:
            raise CogError("missing TileOffsets/TileByteCounts")
        self._lazy = (_TILOFFS, _TILCNTS)
        self._tag = {c: tags[c] for c in self._lazy}
        self.offsets = self.counts = None  # loaded only if this level is read
        self.cols = (self.width + self.tilew - 1) // self.tilew
        self.rows = (self.height + self.tileh - 1) // self.tileh
        self.dtype = end + _DT.get(self.bps, "u1")

    def index(self, idx0: int, idx1: int):
        """Block index entries [idx0..idx1] (row-major block range): returns
        (offsets, counts). Small arrays are loaded once and kept; huge ones are
        fetched as a tight slice around the requested range (both arrays are
        adjacent in COGs, so this usually costs one small request)."""
        if self.offsets is not None:
            return self.offsets[idx0:idx1 + 1], self.counts[idx0:idx1 + 1]
        szs = [_TYPES[typ][1] for typ, _c, _r in self._tag.values()]
        if (max(sz * cnt for sz, (_t, cnt, _r) in zip(szs, self._tag.values())) <= _SLICE_AT
                or not all(isinstance(r, tuple) for _t, _c, r in self._tag.values())):
            self.ensure()
            return self.offsets[idx0:idx1 + 1], self.counts[idx0:idx1 + 1]
        spans = [(ref[0] + sz * idx0, sz * (idx1 - idx0 + 1))
                 for sz, (_typ, _cnt, ref) in zip(szs, self._tag.values())]
        arrs = [np.frombuffer(b, dtype=self.end + ("Q" if sz == 8 else "I"))
                for sz, b in zip(szs, self._reader.read_many(spans))]
        return arrs[0], arrs[1]

    def ensure(self):
        """Fetch the (potentially multi-MB) TileOffsets/TileByteCounts arrays.
        Both are batched in one read_many: in COGs they are adjacent."""
        if self.offsets is not None:
            return
        arrs: list = [None, None]
        holes, spans = [], []
        for i, code in enumerate(self._lazy):
            typ, cnt, ref = self._tag[code]
            sz = _TYPES[typ][1]
            dt = self.end + ("Q" if sz == 8 else "I")
            if isinstance(ref, tuple):
                holes.append((i, dt))
                spans.append((ref[0], sz * cnt))
            else:
                arrs[i] = np.frombuffer(ref[: sz * cnt], dtype=dt)
        for (i, dt), blob in zip(holes, self._reader.read_many(spans)):
            arrs[i] = np.frombuffer(blob, dtype=dt)
        self.offsets, self.counts = arrs


class CogFile:
    def __init__(self, reader):
        self.reader = reader
        if config.HEAD_PREFETCH:  # COGs pack all IFDs near the head: 1 RTT
            reader.prefetch(0, config.HEAD_PREFETCH)
        h = reader.read(0, 8)
        if h[:2] == b"II":
            self.end = "<"
        elif h[:2] == b"MM":
            self.end = ">"
        else:
            raise CogError("not a TIFF")
        magic, aux = struct.unpack(self.end + "HI", h[2:8])
        if magic == 42:
            self.abis, ifd_off = 4, aux
        elif magic == 43 and aux == 8:  # BigTIFF: offsets are 64-bit
            self.abis, ifd_off = 8, struct.unpack(self.end + "Q", reader.read(8, 8))[0]
        else:
            raise CogError("not a TIFF/BigTIFF")
        if config.HEAD_PREFETCH and ifd_off >= config.HEAD_PREFETCH:
            # unoptimized TIFF: IFD (and often its value arrays) near EOF
            reader.prefetch(ifd_off, config.HEAD_PREFETCH)

        self.levels: list[_Level] = []
        self.masks: list[_Level] = []  # alpha masks (NewSubfileType & 4)
        tags0 = None
        while ifd_off:
            tags, ifd_off = self._parse_ifd(ifd_off)
            sub = _scalar(self.end, tags, _NEW_SUBFILE) or 0
            if tags0 is None:
                tags0 = tags
                self.levels.append(_Level(self.end, tags, self.reader))
            elif sub & 4:  # coverage mask, main level or overview level
                self.masks.append(_Level(self.end, tags, self.reader))
            elif sub & 1:  # reduced-resolution image: an internal overview
                self.levels.append(_Level(self.end, tags, self.reader))
            else:
                break
        if not self.levels:
            raise CogError("no image IFD")
        self._init_geo(tags0)

    # --- IFD parsing ------------------------------------------------------
    def _parse_ifd(self, off: int):
        """IFD layout adapts to classic TIFF (12-byte entries) vs BigTIFF (20)."""
        end, abis = self.end, self.abis
        cntsz = 2 if abis == 4 else 8  # IFD entry count: SHORT (classic) / QUAD (BigTIFF)
        n = struct.unpack(end + ("H" if abis == 4 else "Q"), self.reader.read(off, cntsz))[0]
        esz = 2 * abis + 4  # entry = code(2)+type(2)+count(abis)+value(abis)
        data = self.reader.read(off + cntsz, esz * n)
        tags = {}
        pending = []  # (code, typ, cnt) + spans, out-of-line values fetched at once
        spans = []
        for i in range(n):
            code, typ = struct.unpack(end + "HH", data[esz * i: esz * i + 4])
            cnt = struct.unpack(end + ("I" if abis == 4 else "Q"),
                                data[esz * i + 4: esz * i + 4 + abis])[0]
            raw = data[esz * i + 4 + abis: esz * (i + 1)]
            f, sz = _TYPES.get(typ, ("I", 4))
            if sz * cnt > abis:
                ptr = struct.unpack(end + ("I" if abis == 4 else "Q"), raw)[0]
                if code in (_TILOFFS, _TILCNTS):
                    tags[code] = (typ, cnt, (ptr, b""))  # lazy: big block index
                else:
                    pending.append((code, typ, cnt))
                    spans.append((ptr, sz * cnt))
            else:
                tags[code] = (typ, cnt, raw[: sz * cnt])
        noff = off + cntsz + esz * n
        spans.append((noff, abis))  # next IFD pointer in the same batch
        vals = self.reader.read_many(spans)
        for (code, typ, cnt), v in zip(pending, vals):
            tags[code] = (typ, cnt, v)
        nxt = struct.unpack(end + ("I" if abis == 4 else "Q"), vals[-1])[0]
        return tags, nxt

    def _init_geo(self, tags):
        if _MODELTRANSF in tags:
            raise CogError("rotated rasters not supported")
        tie = _doubles(self.end, tags, _TIEPOINT)
        sc = _doubles(self.end, tags, _SCALE)
        if len(tie) < 6 or len(sc) < 2:
            raise CogError("missing ModelTiepoint/ModelPixelScale")
        self.psx, self.psy = abs(sc[0]), abs(sc[1])
        self.ox = tie[3] - tie[0] * self.psx
        self.oy = tie[4] + tie[1] * self.psy
        v = tags.get(_GDAL_NODATA, (0, 0, b""))[2].decode(errors="ignore").strip()
        self.nodata = [float(p) for p in v.split(",")] if v else None
        self.extra_alpha = _scalar(self.end, tags, _EXTRA) is not None
        keys = self._geokeys(tags)
        self.crs = crs.crs_from_geokeys(keys)
        self.crs_name = self.crs.name
        self.unit_m = crs.unit_scale(keys.get(3076, 9001))
        self._cache: OrderedDict = OrderedDict()
        self._web_bbox = None

    def _geokeys(self, tags) -> dict:
        if _GEOKEYS not in tags:
            return {}
        _, cnt, v = tags[_GEOKEYS]
        keys = struct.unpack(self.end + "H" * (cnt // 4 * 4), v[: 2 * cnt])
        dbl = _doubles(self.end, tags, _GEODOUBLE)
        out = {}
        for i in range(0, len(keys) - 3, 4):
            kid, loc, _c, val = keys[i: i + 4]
            if loc == 0:
                out[kid] = val
            elif loc == _GEODOUBLE and val < len(dbl):
                out[kid] = dbl[val]
        return out

    # --- geometry ----------------------------------------------------------
    def web_bbox(self):
        """Footprint densified in the source CRS, reprojected to EPSG:3857."""
        if self._web_bbox is None:
            lvl = self.levels[0]
            x0, y1 = self.ox, self.oy
            x1, y0 = self.ox + lvl.width * self.psx, self.oy - lvl.height * self.psy
            pts, n = [], 25
            for i in range(n + 1):
                t = i / n
                pts += [(x0 + (x1 - x0) * t, y0), (x0 + (x1 - x0) * t, y1),
                        (x0, y0 + (y1 - y0) * t), (x1, y0 + (y1 - y0) * t)]
            lonslats = [self.crs.to_lonlat(x, y) for x, y in pts]
            web = crs.WEBMERCATOR
            a = web.to_xy(min(p[0] for p in lonslats), min(p[1] for p in lonslats))
            b = web.to_xy(max(p[0] for p in lonslats), max(p[1] for p in lonslats))
            self._web_bbox = (a[0], a[1], b[0], b[1])
        return self._web_bbox

    def footprint_lonlat(self) -> list[list[float]]:
        """Densified source bbox, projected to lon/lat, closed ring (for GeoJSON)."""
        lvl = self.levels[0]
        x0, y1 = self.ox, self.oy
        x1, y0 = self.ox + lvl.width * self.psx, self.oy - lvl.height * self.psy
        n, pts = 20, []
        for i in range(n + 1):  # top edge, left->right
            pts.append((x0 + (x1 - x0) * i / n, y1))
        for i in range(n + 1):  # right edge
            pts.append((x1, y1 - (y1 - y0) * i / n))
        for i in range(n + 1):  # bottom edge, right->left
            pts.append((x1 - (x1 - x0) * i / n, y0))
        for i in range(n + 1):  # left edge
            pts.append((x0, y0 + (y1 - y0) * i / n))
        ring = [[round(lon, 7), round(lat, 7)] for lon, lat in
                (self.crs.to_lonlat(x, y) for x, y in pts)]
        ring.append(ring[0])
        return ring

    # --- pixel windows ------------------------------------------------------
    def window(self, level: int, c0: int, r0: int, c1: int, r1: int):
        """Read [c0:c1, r0:r1] of overview `level`; outside pixels valid=False.
        Applies the coverage mask, nodata and band selection when present."""
        lvl = self.levels[level]
        out, valid = self._window(lvl, level, c0, r0, c1, r1)
        for m in self.masks:
            if m.width == lvl.width and m.height == lvl.height:
                mout, _ = self._window(m, ~level, c0, r0, c1, r1)
                valid &= mout[..., 0] > 0
                break
        s = lvl.samples
        if lvl.photometric == 6 and lvl.compression not in (6, 7):
            out = CogFile._ycbcr_to_rgb(out)
            s = 3
        if self.extra_alpha and s >= 4:
            valid &= out[..., 3] > 0
            out = out[..., :3]
            s = 3
        if self.nodata is not None:
            nd = np.array(self.nodata[: s], dtype=out.dtype)
            valid &= ~np.any(out == nd, axis=-1)
        return out, valid

    def _window(self, lvl: _Level, ns: int, c0: int, r0: int, c1: int, r1: int):
        """Assemble a raw pixel window from compressed blocks (ns: cache space)."""
        c0, c1, r0, r1 = int(c0), int(c1), int(r0), int(r1)
        out = np.zeros((r1 - r0, c1 - c0, lvl.samples), dtype=lvl.dtype)
        valid = np.zeros((r1 - r0, c1 - c0), dtype=bool)

        lo_c, hi_c = max(c0, 0), min(c1, lvl.width)
        lo_r, hi_r = max(r0, 0), min(r1, lvl.height)
        if lo_c >= hi_c or lo_r >= hi_r:
            return out, valid

        ti0, ti1 = lo_c // lvl.tilew, (hi_c - 1) // lvl.tilew
        tj0, tj1 = lo_r // lvl.tileh, (hi_r - 1) // lvl.tileh
        blocks, missing = {}, []
        for tj in range(tj0, tj1 + 1):
            for ti in range(ti0, ti1 + 1):
                k = (ns, ti, tj)
                blk = self._cache.get(k)
                if blk is not None:
                    self._cache.move_to_end(k)
                    blocks[k] = blk
                else:
                    missing.append((k, tj * lvl.cols + ti))
        if missing:
            base = tj0 * lvl.cols + ti0
            offs, cnts = lvl.index(base, tj1 * lvl.cols + ti1)
            spans = [(int(offs[idx - base]), int(cnts[idx - base])) for _k, idx in missing]
            raws = self.reader.read_many(spans)
            if _POOL is not None and _threaded_decode(lvl, len(raws)):
                decoded = list(_POOL.map(lambda raw: self._decode(lvl, raw), raws))
            else:
                decoded = [self._decode(lvl, raw) for raw in raws]
            for (k, _idx), blk in zip(missing, decoded):
                blocks[k] = blk
                self._cache[k] = blk
            while len(self._cache) > config.BLOCK_CACHE:
                self._cache.popitem(last=False)

        for tj in range(tj0, tj1 + 1):
            for ti in range(ti0, ti1 + 1):
                blk = blocks[(ns, ti, tj)]
                by, bx = tj * lvl.tileh, ti * lvl.tilew
                oy, ox = max(by, lo_r), max(bx, lo_c)
                eh = min(by + lvl.tileh, lvl.height, hi_r) - oy
                ew = min(bx + lvl.tilew, lvl.width, hi_c) - ox
                if eh <= 0 or ew <= 0:
                    continue
                out[oy - r0: oy - r0 + eh, ox - c0: ox - c0 + ew] = blk[oy - by: oy - by + eh, ox - bx: ox - bx + ew]
                valid[oy - r0: oy - r0 + eh, ox - c0: ox - c0 + ew] = True
        return out, valid

    def _decode(self, lvl: _Level, raw: bytes) -> np.ndarray:
        th, tw, s, comp = lvl.tileh, lvl.tilew, lvl.samples, lvl.compression
        if comp in (6, 7):
            a = self._decode_jpeg(raw, lvl.jpegtables, s)
            if a.shape[2] not in (s, 1):
                raise CogError(f"unexpected JPEG band count {a.shape[2]}")
            return a
        if comp == 1:
            buf = raw
        elif comp in (8, 32946):
            buf = zlib.decompress(raw)
        elif comp == 5:
            buf = _ic.lzw_decode(raw) if _ic else _lzw_decode(raw)
        elif comp == 32773:
            buf = _ic.packbits_decode(raw) if _ic else _packbits_decode(raw)
        elif comp == 50000:
            buf = _zstd_decode(raw)
        else:
            raise CogError(f"unsupported compression {comp}")
        if lvl.bps == 1:  # masks: 1 bit per pixel, MSB-first
            a = np.unpackbits(np.frombuffer(buf, dtype=np.uint8))
        else:
            a = np.frombuffer(buf, dtype=lvl.dtype)
        if a.size != th * tw * s:
            raise CogError("truncated block")
        a = a.reshape((th, tw, s))
        if lvl.predictor == 2 and comp != 1 and lvl.bps in (8, 16):
            if lvl.bps == 8:
                a = np.cumsum(a, axis=1, dtype=np.uint8)
            elif lvl.bps == 16:
                a = np.cumsum(a, axis=1, dtype=np.uint16)
            else:
                raise CogError("unsupported predictor/bit depth")
        return a

    @staticmethod
    def _decode_jpeg(raw: bytes, jpegtables: bytes | None, samples: int) -> np.ndarray:
        # JPEG-in-TIFF: blocks may miss the DQT/DHT segments (stored once in the
        # IFD JPEGTables tag) -> splice IFD tables + block body, GDAL-style.
        if jpegtables:
            tab = jpegtables
            if tab[-2:] in (b"\xff\x7f", b"\xff\xd9"):
                tab = tab[:-2]  # drop trailing EOI
            body = raw[2:] if raw[:2] == b"\xff\xd8" else raw
            raw = tab + body
        # GRAY for 1-sample levels only: a 3-component JPEG decoded as GRAY is not Pillow's L
        mode = "GRAY" if samples == 1 else "RGB"
        if _sjpeg is not None:
            try:
                return _sjpeg(raw, colorspace=mode)
            except Exception:
                pass  # Pillow is more permissive on odd streams
        im = Image.open(BytesIO(raw))
        im.load()
        a = np.asarray(im.convert("L" if samples == 1 else "RGB"))
        return a[..., None] if a.ndim == 2 else a

    @staticmethod
    def _ycbcr_to_rgb(a: np.ndarray) -> np.ndarray:
        a = a.astype(np.float32)
        y, cb, cr = a[..., 0], a[..., 1], a[..., 2]
        r = y + 1.402 * (cr - 128)
        g = y - 0.3445 * (cb - 128) - 0.7145 * (cr - 128)
        b = y + 1.772 * (cb - 128)
        return np.dstack([r, g, b]).clip(0, 255).astype(np.uint8)

    def close(self):
        self._cache.clear()
        self.reader.close()
