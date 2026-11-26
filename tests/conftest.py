"""Offline fixtures: small COGs written by a pure-python TIFF writer, served
over a local Range-capable HTTP server that counts transferred bytes."""
import http.server
import io
import os
import socketserver
import struct
import threading
import zlib
from functools import lru_cache

import numpy as np
import pytest
from PIL import Image

try:  # only used to write the ZSTD fixture
    import imagecodecs as _ic
except ImportError:
    _ic = None

try:  # only used to place the EPSG:2056 fixture (PROJ fallback tests)
    import pyproj as _pyproj
except ImportError:
    _pyproj = None

os.environ["ALLOW_PRIVATE_NET"] = "1"  # fixture server is on loopback

W = H = 512
TILE = 128
# what a tag 347 (JPEGTables) blob ends with: libtiff writes the JPEGTM EOI
# (FF 7F), other writers the JPEG one (FF D9), a few nothing at all
TBL_JPEGTM, TBL_EOI, TBL_NONE = b"\xff\x7f", b"\xff\xd9", b""

# fixture footprint (near Paris) in lon/lat and its EPSG:3857 box
BBOX4326 = (2.30, 48.83, 2.36, 48.88)  # w,s,e,n
RWEB = 6378137.0

# CH1903+ test box (near Bern): an EPSG:2056 COG, outside the CRS table and
# rendered only through the PROJ fallback (pyproj). Chosen so the
# whole footprint sits inside the z14 centre tile of swiss_tile() (the x+1 tile
# boundary is at lon 7.4487).
SWISS = (7.4300, 46.9500, 7.4450, 46.9600)  # w,s,e,n


def _swiss_grid():
    """(xmin, ymin, xmax, ymax) in EPSG:2056 m of the SWISS box, projected
    through PROJ - the grid of the cog2056.tif fixture."""
    tf = _pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
    pts = [tf.transform(lo, la) for lo, la in ((SWISS[0], SWISS[1]),
                                               (SWISS[0], SWISS[3]),
                                               (SWISS[2], SWISS[1]),
                                               (SWISS[2], SWISS[3]))]
    xs, ys = zip(*pts)
    return min(xs), min(ys), max(xs), max(ys)


def _web(x, y):  # lon/lat deg -> EPSG:3857 m
    return (RWEB * np.radians(x),
            RWEB * np.log(np.tan(np.pi / 4 + np.radians(y) / 2)))


@lru_cache(maxsize=None)
def pattern(half=False):
    """Deterministic RGB content; red square in the northern half of the image."""
    w, h = (W // 2, H // 2) if half else (W, H)
    a = np.zeros((h, w, 3), np.uint8)
    a[..., 0] = (np.arange(w) * 255 // (w - 1))[None, :]
    a[..., 1] = (np.arange(h) * 255 // (h - 1))[:, None]
    a[h // 4: h // 2, w // 4: w // 2] = [255, 0, 0]
    return a


def _block(arr, comp, lzw_ours, predictor, bps, webp_lossless=True,
           jxl_lossless=True, j2k_lossless=True):
    """One encoded TIFF block: bit-packed when bps=1 (masks), horizontally
    differenced when predictor=2, then compressed."""
    if bps == 1:
        raw = np.packbits(np.ascontiguousarray(arr).ravel()).tobytes()
    else:
        if predictor == 2:  # libtiff's horizontal differencing (modular per dtype)
            d = np.ascontiguousarray(arr).copy()
            d[:, 1:] -= d[:, :-1]
            arr = d
        raw = np.ascontiguousarray(arr).tobytes()
    if comp == 7:  # every block keeps its own JPEG tables (no JPEGTables tag)
        img = Image.fromarray(arr) if arr.ndim == 3 else Image.fromarray(arr, "L")
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=90)
        return buf.getvalue()
    if comp == 50001:  # TIFF WebP: a block is one whole RIFF/WEBP image
        if arr.ndim == 2:  # libwebp cannot write grayscale: the single-band
            arr = np.repeat(arr[..., None], 3, axis=2)  # fixture carries RGB
            # blocks, irrelevant - the level is refused at open, never decoded
        return (_ic.webp_encode(arr, lossless=True) if webp_lossless
                else _ic.webp_encode(arr, lossless=False, level=75))
    if comp in (50002, 52546):  # TIFF JPEG XL (GDAL / DNG 1.7 ids): one bare
        # codestream per block, the layout GDAL's libjxl codec writes
        return (_ic.jpegxl_encode(arr, lossless=True) if jxl_lossless
                else _ic.jpegxl_encode(arr, lossless=False, distance=1.0,
                                       effort=5))
    if comp in (34712, 34713):  # TIFF JPEG 2000 (GDAL's two codec ids): one bare
        # codestream per block (FF4F), 5/3 reversible or 9/7 lossy at 40 dB
        return (_ic.jpeg2k_encode(arr, codecformat="J2K") if j2k_lossless
                else _ic.jpeg2k_encode(arr, codecformat="J2K", reversible=False,
                                       level=40))
    if comp == 50000:
        return _ic.zstd_encode(raw)
    if comp == 5:  # TIFF LZW: "ours" = the variant the pure-Python decoder reads
        return _lzw_encode(raw) if lzw_ours else _ic.lzw_encode(raw)
    if comp == 32773:
        return _ic.packbits_encode(raw)
    return zlib.compress(raw)


def _tiles(arr, comp=8, lzw_ours=True, predictor=1, bps=8, webp_lossless=True,
           jxl_lossless=True, j2k_lossless=True):
    th = tw = TILE
    h, w = arr.shape[:2]
    return [_block(arr[tj * th:(tj + 1) * th, ti * tw:(ti + 1) * tw],
                   comp, lzw_ours, predictor, bps, webp_lossless, jxl_lossless,
                   j2k_lossless)
            for tj in range(h // th) for ti in range(w // tw)]


def _jpeg_tables(blocks, eoi, keep=False, bad=False):
    """Rewrite self-contained JPEG blocks the way GDAL/libtiff store them in a COG:
    the DQT segments move to the IFD as tag 347 (`SOI` + DQT + terminator) while
    each block keeps SOI + SOF + DHT + SOS + scan data - the DHT stay per block,
    as they do in the reference COGs. The walk stops at SOS: scan data may hold
    0xFF bytes. Returns (blocks, tables); `keep` leaves the blocks alone (tag 347
    declared *and* blocks carry their own DQT - a later DQT wins, so the splice
    is still right), `bad` omits the SOI, making the spliced stream no JPEG."""
    out, tab = [], None
    for b in blocks:
        head, dqt, i = bytearray(), bytearray(), 2
        while i < len(b) - 1 and b[i] == 0xFF and b[i + 1] not in (0xD9, 0xDA):
            ln = int.from_bytes(b[i + 2:i + 4], "big")
            (dqt if b[i + 1] == 0xDB else head).extend(b[i:i + 2 + ln])  # 0xDB = DQT
            i += 2 + ln
        if dqt and tab is None:  # one blob per IFD: every block shares these tables
            tab = (b"" if bad else b[:2]) + bytes(dqt) + eoi
        out.append(b if keep or not dqt else b[:2] + bytes(head) + b[i:])
    return out, tab


def _lzw_encode(data: bytes) -> bytes:
    """Encoder for the exact LZW dialect of app.geotiff._lzw_decode: that decoder
    starts its table appends at 256, so codes 256/257 are dead for data (the
    match is shrunk there). This mirrors the decoder's state transitions code
    for code; it is self-consistency only — libtiff's dialect needs imagecodecs."""
    clear, eoi = 256, 257
    out, acc, nbits = bytearray(), 0, 0
    base = [bytes([i]) for i in range(256)]
    table, pos, width = list(base), {bytes([i]): i for i in range(256)}, 9
    prev_entry = b""

    def emit(code):
        nonlocal acc, nbits
        acc = (acc << width) | code
        nbits += width
        while nbits >= 8:
            nbits -= 8
            out.append((acc >> nbits) & 0xFF)

    def process(code):  # apply what the decoder does when it reads code
        nonlocal prev_entry, width
        if code == clear:
            table[:] = base
            pos.clear()
            pos.update({s: i for i, s in enumerate(table)})
            width = 9
            prev_entry = b""
            return
        if code < len(table):
            entry = table[code]
        else:  # code == len(table): KwKwK, the not-yet-appended entry
            entry = prev_entry + prev_entry[:1]
        if prev_entry:
            table.append(prev_entry + entry[:1])
            pos[table[-1]] = len(table) - 1
            if len(table) + 1 >= (1 << width) and width < 12:
                width += 1
        prev_entry = entry

    emit(clear)
    p = 0
    while p < len(data):
        best, best_code, w = 1, data[p], 1
        while w < 64:  # longest match (capped: fast, all decoder paths covered)
            s = data[p:p + w + 1]
            if len(s) < w + 1:
                break
            code = pos.get(s, len(table) if prev_entry and s == prev_entry + prev_entry[:1] else None)
            if code is None or code in (clear, eoi) or code >= 4096:
                break
            best, best_code, w = w + 1, code, w + 1
        emit(best_code)
        process(best_code)
        p += best
        if len(table) >= 4096:  # beyond 12-bit codes: restart like a COG writer would
            emit(clear)
            process(clear)
    emit(eoi)
    if nbits:
        out.append((acc << (8 - nbits)) & 0xFF)
    return bytes(out)


def pattern4(half=False):
    """RGB pattern + alpha channel: a hole in the middle is fully transparent."""
    a = pattern(half)
    h, w = a.shape[:2]
    al = np.full((h, w), 255, np.uint8)
    al[3 * h // 8: 5 * h // 8, 3 * w // 8: 5 * w // 8] = 0
    return np.dstack([a, al])


# declared (BitsPerSample, SampleFormat) -> numpy dtype of the pixel payload
_DTS = {(8, 1): np.uint8, (16, 1): np.uint16, (32, 1): np.uint32, (64, 1): np.uint64,
        (8, 2): np.int8, (16, 2): np.int16, (32, 2): np.int32, (64, 2): np.int64,
        (16, 3): np.float16, (32, 3): np.float32, (64, 3): np.float64}


def write_cog(path, ox, oy, psx, psy, geographic, comp=8, samples=3, bps=8,
              nodata=None, geokeys=None, lzw_ours=True, predictor=1, mask=False,
              rotated=False, subifd=False, holes=False, webp_lossless=True,
              fmt=None, sentinel=None, jpegtables=None, keep_tables=False,
              bad_tables=False, jxl_lossless=True, big=False, j2k_lossless=True,
              no_overview=False):
    """Tiled little-endian TIFF (classic, or BigTIFF when big=True: version 43,
    64-bit offsets/counts, LONG8 block index - what gdal_translate switches to
    past 4 GB): level 0 + a 2x internal overview (no_overview=True writes level 0
    alone, the COG without a pyramid that CogFile.synth_to exists for), plus
    optional extras - a 1-bit internal coverage mask (NewSubfileType & 4), a
    horizontal-differencing predictor, GDAL_NODATA, ExtraSamples=ALPHA, a
    ModelTransformation (rotated raster, which the reader must refuse),
    overviews as SubIFDs (tag 330) instead of the linked IFD chain,
    count-0 blocks (cog_merger holes: absent data in a merged footprint),
    WebP blocks (compression 50001, lossless unless webp_lossless=False), JPEG XL
    blocks (50002, the GDAL/libtiff codec id, or 52546 as DNG 1.7 spells it: one
    bare codestream per block, lossless unless jxl_lossless=False), JPEG 2000
    blocks (34712, or 34713 as GDAL spells its lossless "read only" code: one bare
    FF4F codestream per block, 5/3 reversible unless j2k_lossless=False drops to
    9/7 at 40 dB), a
    declared SampleFormat (tag 339: 1 unsigned, 2 signed, 3 float, 4 complex for
    a file that must be refused) with the pattern widened to that depth, or a
    nodata sentinel written over the first column (-9999, NaN: values a reader
    must never turn into a pixel), or JPEG blocks stored the GDAL/libtiff way -
    `jpegtables=<terminator>` moves their DQT segments to the IFD JPEGTables tag
    (347) and strips them from the blocks, `keep_tables` declares the tag while
    the blocks keep their own DQT, `bad_tables` writes a tag 347 without SOI.
    Layout: header | tiles of each IFD | value areas | IFDs."""
    src = pattern() if samples == 3 else pattern4() if samples == 4 else pattern()[..., 0]
    src_h = (pattern(half=True) if samples == 3 else pattern4(half=True)
             if samples == 4 else pattern(half=True)[..., 0])
    # an undeclared depth (12-bit...) is still written as a byte payload: the
    # point of that fixture is that the reader refuses it, not what it holds
    dt = _DTS.get((bps, fmt or 1), np.uint16)
    mul = 257 if (bps, fmt or 1) == (16, 1) else 1  # the 16-bit fixture keeps the pattern
    if dt is not np.uint8:
        src, src_h = src.astype(dt) * mul, src_h.astype(dt) * mul
    if sentinel is not None:  # pattern() is lru_cached: copy before marking it out
        src, src_h = src.copy(), src_h.copy()
        src[:, 0] = src_h[:, 0] = sentinel
    phot = 6 if (comp == 7 and samples == 3) else (1 if samples == 1 else 2)
    lvls = [dict(arr=a, sub=sub, samples=samples, bps=bps, comp=comp, phot=phot, geo=True)
            for a, sub in ((src, 0), (src_h, 1))]
    if no_overview:  # a COG written without its internal pyramid: level 0 alone,
        # which is what CogFile.synth_to exists for
        lvls = [l for l in lvls if not l["sub"]]
    if mask:  # coverage: the west third of the raster is no-data
        bits = np.zeros((H, W), np.uint8)
        bits[:, W // 3:] = 1
        lvls.append(dict(arr=bits, sub=4, samples=1, bps=1, comp=8, phot=1, geo=False))
    abis = 8 if big else 4  # address size: BigTIFF widens offsets, counts, inline values
    for l in lvls:  # a value array that fits the IFD entry's value field must stay
        # inline (TIFF rule: 4 bytes classic, 8 in BigTIFF); 1-bit masks never difference
        l["bps_len"] = 2 * l["samples"] if 2 * l["samples"] > abis else 0
        l["fmt"] = 0 if l["bps"] == 1 else (fmt or 1)  # 0 = tag 339 absent (unsigned)
        l["fmt_len"] = 2 * l["samples"] if l["fmt"] and 2 * l["samples"] > abis else 0
        l["pred"] = predictor if l["bps"] in (8, 16) or l["fmt"] else 1
    blobs = [_tiles(l["arr"], l["comp"], lzw_ours, l["pred"], l["bps"],
                    webp_lossless, jxl_lossless, j2k_lossless)
             for l in lvls]
    tabs = [None] * len(lvls)
    if comp == 7 and jpegtables is not None:  # one tag 347 per level (per IFD)
        for i, blocks in enumerate(blobs):
            blobs[i], tabs[i] = _jpeg_tables(blocks, jpegtables, keep_tables, bad_tables)

    # rough layout: header | tiles | areas | ifds ; areas are small so offsets
    # are computed exactly in a second pass
    hdr = 16 if big else 8
    tile_off = hdr
    sizes0 = sum(len(b) for lv in blobs for b in lv)
    ityp = 16 if big else 4  # block index arrays: LONG8 in BigTIFF, LONG classic

    geokeys = geokeys or ([1, 1, 0, 2, 1024, 0, 1, 2, 2048, 0, 1, 4326] if geographic else
                          [1, 1, 0, 1, 1024, 0, 1, 1, 2048, 0, 1, 4979, 2052, 0, 1, 9001,
                           3072, 0, 1, 3857, 3076, 0, 1, 9001])

    nts = [(l["arr"].shape[1] // TILE) * (l["arr"].shape[0] // TILE) for l in lvls]

    def ifd_entries(i):
        """IFD entries in tag order, and the out-of-line areas they point to."""
        l, nt = lvls[i], nts[i]
        ent = [
            (254, 4, 1, struct.pack("<I", l["sub"])[:4]),
            (256, 4, 1, struct.pack("<I", l["arr"].shape[1])[:4]),
            (257, 4, 1, struct.pack("<I", l["arr"].shape[0])[:4]),
            (258, 3, l["samples"], None if l["bps_len"]
             else struct.pack("<" + "H" * l["samples"], *([l["bps"]] * l["samples"]))),
            (259, 3, 1, struct.pack("<H", l["comp"])[:4]),
            (262, 3, 1, struct.pack("<H", l["phot"])[:4]),
            (277, 3, 1, struct.pack("<H", l["samples"])[:4]),
        ]
        if l["samples"] == 4:  # ExtraSamples=ALPHA: the reader turns band 4 into a mask
            ent.append((338, 3, 1, struct.pack("<H", 2)[:4]))
        if l["fmt"]:  # SampleFormat: 1 unsigned / 2 signed / 3 float / 4 complex
            ent.append((339, 3, l["samples"],
                        None if l["fmt_len"]
                        else struct.pack("<" + "H" * l["samples"],
                                         *([l["fmt"]] * l["samples"]))))
        if tabs[i]:  # JPEGTables (tag 347): the DQT segments the blocks lack
            ent.append((347, 7, len(tabs[i]), None))
        ent += [(284, 3, 1, struct.pack("<H", 1)[:4]),  # planar contig
                (317, 3, 1, struct.pack("<H", l["pred"])[:4]),
                (322, 3, 1, struct.pack("<H", TILE)[:4]),
                (323, 3, 1, struct.pack("<H", TILE)[:4]),
                (324, ityp, nt, None), (325, ityp, nt, None)]
        if l["geo"]:
            ent += [(33550, 12, 3, None), (33922, 12, 6, None), (34735, 3, len(geokeys), None)]
            if rotated:  # ModelTransformation: the reader refuses rotated rasters
                ent.append((34264, 12, 16, None))
            if nodata:  # GDAL_NODATA (ASCII, inline when it fits the value field)
                ent.append((42113, 2, len(nodata),
                            nodata.encode() if len(nodata) <= abis else None))
        if i == 0 and subifd:  # SubIFDs (cog_merger layout): overview IFD offsets
            no = len(lvls) - 1
            # a single value that fits the entry must be inline (spec: no pointer)
            ent.append((330, ityp, no, b"PTRO" if no == 1 else None))
            if no > 1:
                areas.append((330, abis * no))
        # area layout: bps (0 if inline) | fmt (0 if inline/absent) | offs | cnts |
        # jpegtables (0 if absent) | then for geo IFDs: scale | tiepoint | geokeys |
        # transformation | nodata | then subifds
        areas = [(258, l["bps_len"]), (339, l["fmt_len"]),
                 (324, abis * nt), (325, abis * nt)]
        if tabs[i]:
            areas.append((347, len(tabs[i])))
        if l["geo"]:
            areas += [(33550, 24), (33922, 48), (34735, 2 * len(geokeys))]
            if rotated:
                areas.append((34264, 128))
            if nodata and len(nodata) > abis:
                areas.append((42113, len(nodata)))
        if i == 0 and subifd:
            areas.append((330, abis * (len(lvls) - 1)))
        return ent, areas

    def area_bytes(i):
        l, nt = lvls[i], nts[i]
        cols = l["arr"].shape[1] // TILE
        # cog_merger holes: last block column of each image level has count 0
        cnts = [0 if holes and i < 2 and j % cols == cols - 1 else len(b)
                for j, b in enumerate(blobs[i])]
        a = bytearray()
        a.extend(struct.pack("<" + "H" * l["samples"], *([l["bps"]] * l["samples"]))
                 if l["bps_len"] else b"")
        a.extend(struct.pack("<" + "H" * l["samples"], *([l["fmt"]] * l["samples"]))
                 if l["fmt_len"] else b"")
        of = "Q" if big else "I"  # block index entries: 64-bit in BigTIFF
        a.extend(struct.pack("<" + of * nt, *([0] * nt)))  # TileOffsets (patched)
        a.extend(struct.pack("<" + of * nt, *cnts))
        if tabs[i]:
            a.extend(tabs[i])  # JPEGTables: SOI + DQT + terminator
        if l["geo"]:
            a.extend(struct.pack("<3d", psx, psy, 0.0))
            a.extend(struct.pack("<6d", 0, 0, 0, ox, oy, 0.0))
            a.extend(struct.pack("<" + "H" * len(geokeys), *geokeys))
            if rotated:
                m = [psx, 0, 0, 0, 0, -psy, 0, 0, 0, 0, 0, 0, ox, oy, 0, 1]
                a.extend(struct.pack("<16d", *m))
            if nodata and len(nodata) > abis:
                a.extend(nodata.encode())  # GDAL_NODATA beyond the value field: out of line
        if i == 0 and subifd and len(lvls) > 2:
            a.extend(bytes(abis * (len(lvls) - 1)))  # overview IFD offsets (patched)
        return bytes(a)

    area_blobs = [area_bytes(i) for i in range(len(lvls))]

    def ifd_bytes(i, next_off):
        ent, areas = ifd_entries(i)
        p = area_start + sum(len(x) for x in area_blobs[:i])
        pos = {}
        for code, size in areas:
            pos[code] = p
            p += size
        cnt_f = "Q" if big else "H"  # IFD entry count: QUAD in BigTIFF, SHORT classic
        val_f = "Q" if big else "I"  # counts, pointers and inline values: 64-bit
        out = struct.pack("<" + cnt_f, len(ent))
        for code, typ, cnt, raw in ent:
            if raw is None:
                raw = struct.pack("<" + val_f, pos[code])
            elif raw == b"PTRO":  # single SubIFD offset: inline, resolved now
                raw = struct.pack("<" + val_f, ifd_offs[1])
            out += struct.pack("<HH" + val_f, code, typ, cnt) + raw[:abis].ljust(abis, b"\0")
        out += struct.pack("<" + val_f, next_off)
        return out

    area_start = tile_off + sizes0
    ifd_offs, p = [], area_start + sum(len(a) for a in area_blobs)
    for i in range(len(lvls)):  # IFD sizes follow from their entry lists
        ifd_offs.append(p)
        p += (8 if big else 2) + (2 * abis + 4) * len(ifd_entries(i)[0]) + abis

    # patch tile offsets inside area_blobs (absolute file offsets); merger
    # holes get count 0 and an offset past EOF: a reader that (wrongly) tried
    # to fetch them would hit an HTTP error, making the bug loud in tests
    cur = tile_off
    for i, lv in enumerate(blobs):
        ab = bytearray(area_blobs[i])
        base = lvls[i]["bps_len"] + lvls[i]["fmt_len"]
        cols = lvls[i]["arr"].shape[1] // TILE
        for j, b in enumerate(lv):
            bad = holes and i < 2 and j % cols == cols - 1
            ab[base + abis * j: base + abis * (j + 1)] = struct.pack(
                "<" + ("Q" if big else "I"), 0x7FFF_FFFF if bad else cur)
            cur += len(b)
        area_blobs[i] = bytes(ab)

    if subifd and len(lvls) > 2:  # SubIFDs array of IFD0 (last area): overview offsets
        ab = bytearray(area_blobs[0])
        p330 = len(ab) - abis * (len(lvls) - 1)
        for k, o in enumerate(ifd_offs[1:]):
            ab[p330 + abis * k: p330 + abis * (k + 1)] = struct.pack(
                "<" + ("Q" if big else "I"), o)
        area_blobs[0] = bytes(ab)

    if big:  # BigTIFF: version 43, offset size 8 + reserved, then the 64-bit IFD offset
        out = bytearray(struct.pack("<2sHH2sQ", b"II", 43, 8, b"\0\0", ifd_offs[0]))
    else:
        out = bytearray(struct.pack("<2sHI", b"II", 42, ifd_offs[0]))
    for lv in blobs:
        for b in lv:
            out.extend(b)
    for ab in area_blobs:
        out.extend(ab)
    for i in range(len(lvls)):
        nxt = 0 if subifd else (ifd_offs[i + 1] if i + 1 < len(ifd_offs) else 0)
        out.extend(ifd_bytes(i, nxt))
    assert len(out) == p, (len(out), p)
    with open(path, "wb") as f:
        f.write(out)


# --- Range-capable file server with byte counting --------------------------

class _Handler(http.server.BaseHTTPRequestHandler):
    served = 0
    requests = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        type(self).requests.append((self.path, self.headers.get("Range")))
        path = os.path.join(self.server.root, self.path.lstrip("/"))
        if not os.path.isfile(path):
            self.send_error(404)
            return
        size = os.path.getsize(path)
        rng = self.headers.get("Range")
        with open(path, "rb") as f:
            if rng and rng.startswith("bytes="):
                a, b = rng[6:].split(",")[0].split("-")  # multi-range: serve first
                a = int(a)
                b = size - 1 if b == "" else int(b)
                f.seek(a)
                data = f.read(b - a + 1)
                type(self).served += len(data)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
                self.send_header("Content-Length", str(len(data)))
            else:
                data = f.read()
                type(self).served += len(data)
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(data)


class RangeServer:
    def __init__(self, root):
        self.root = root

    def __enter__(self):
        # fresh handler class per server so counters are isolated
        handler = type("H", (_Handler,), {"served": 0, "requests": []})
        self.httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
        self.httpd.root = self.root
        self.httpd.daemon_threads = True
        self.handler = handler
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.httpd.server_address[1]
        return self

    def __exit__(self, *a):
        self.httpd.shutdown()
        self.httpd.server_close()

    @property
    def base(self):
        return f"http://127.0.0.1:{self.port}"

    def reset(self):
        self.handler.served = 0
        self.handler.requests = []


@pytest.fixture(scope="session")
def fixtures_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("cogs")
    w0, s0, e0, n0 = BBOX4326
    write_cog(d / "cog4326.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H, geographic=True)
    x0, y1 = _web(w0, n0)
    x1, y0 = _web(e0, s0)
    write_cog(d / "cog3857.tif", x0, y1, (x1 - x0) / W, (y1 - y0) / H, geographic=False)
    # the same box, projected, without its pyramid: the pair pins what CogFile.synth_to
    # adds (projected so the TileJSON zoom math sees metres, not degrees)
    write_cog(d / "cog1l3857.tif", x0, y1, (x1 - x0) / W, (y1 - y0) / H,
              geographic=False, no_overview=True)
    if _pyproj:  # CH1903+/LV95 grid, plus the 3857 twin of the same box:
        # rendering both must match - a missing CH1903+ datum shift (270 m)
        # would put the 2056 tile 40 px away at this zoom
        gx0, gy0, gx1, gy1 = _swiss_grid()
        write_cog(d / "cog2056.tif", gx0, gy1, (gx1 - gx0) / W, (gy1 - gy0) / H,
                  geographic=False,
                  geokeys=[1, 1, 0, 5, 1024, 0, 1, 1, 2048, 0, 1, 4150, 2052, 0,
                           1, 9001, 3072, 0, 1, 2056, 3076, 0, 1, 9001])
        x0, y1 = _web(SWISS[0], SWISS[3])
        x1, y0 = _web(SWISS[2], SWISS[1])
        write_cog(d / "cogswiss3857.tif", x0, y1, (x1 - x0) / W, (y1 - y0) / H,
                  geographic=False)
    write_cog(d / "cogjpeg.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
              geographic=True, comp=7)
    write_cog(d / "cogjpeg1.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
              geographic=True, comp=7, samples=1)
    write_cog(d / "cogjpt.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
              geographic=True, comp=7, jpegtables=TBL_JPEGTM)
    write_cog(d / "cogjpt9.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
              geographic=True, comp=7, jpegtables=TBL_EOI)
    write_cog(d / "cogjptne.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
              geographic=True, comp=7, jpegtables=TBL_NONE)
    write_cog(d / "cogjptself.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
              geographic=True, comp=7, jpegtables=TBL_JPEGTM, keep_tables=True)
    write_cog(d / "cogjptbad.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
              geographic=True, comp=7, jpegtables=TBL_JPEGTM, bad_tables=True)
    write_cog(d / "cogjpt1.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
              geographic=True, comp=7, samples=1, jpegtables=TBL_JPEGTM)
    if _ic is not None:
        write_cog(d / "cogzstd.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
                  geographic=True, comp=50000)
        write_cog(d / "cogpack.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
                  geographic=True, comp=32773)
        write_cog(d / "coglzwlib.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
                  geographic=True, comp=5, lzw_ours=False)  # imagecodecs/libtiff LZW
        write_cog(d / "cogwebp.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
                  geographic=True, comp=50001)  # lossless WebP blocks
        write_cog(d / "cogwebpl.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
                  geographic=True, comp=50001, webp_lossless=False)  # lossy VP8
        write_cog(d / "cogwebpa.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
                  geographic=True, comp=50001, samples=4)  # RGBA WebP + ExtraSamples
        write_cog(d / "cogwebp1.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
                  geographic=True, comp=50001, samples=1)  # refused: not RGB/RGBA
        write_cog(d / "cogjxl.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
                  geographic=True, comp=50002)  # lossless JPEG XL codestreams
        write_cog(d / "cogjxll.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
                  geographic=True, comp=50002, jxl_lossless=False)  # distance 1.0
        write_cog(d / "cogjxla.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
                  geographic=True, comp=50002, samples=4)  # RGBA JXL + ExtraSamples
        write_cog(d / "cogjxlg.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
                  geographic=True, comp=50002, samples=1)  # gray: valid JXL, unlike WebP
        write_cog(d / "cogjxld.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
                  geographic=True, comp=52546)  # same blocks, DNG 1.7 codec id
        write_cog(d / "cogjxl2.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
                  geographic=True, comp=50002, samples=2)  # gray+alpha: refused
        write_cog(d / "cogj2k.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
                  geographic=True, comp=34712)  # lossless JPEG 2000 codestreams
        write_cog(d / "cogj2kl.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
                  geographic=True, comp=34712, j2k_lossless=False)  # 9/7, 40 dB
        write_cog(d / "cogj2ka.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
                  geographic=True, comp=34712, samples=4)  # RGBA J2K + ExtraSamples
        write_cog(d / "cogj2kg.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
                  geographic=True, comp=34713, samples=1)  # gray, "read only" code
        write_cog(d / "cogj2k2.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
                  geographic=True, comp=34712, samples=2)  # gray+alpha: refused
    write_cog(d / "cog1lvl.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
              geographic=True, no_overview=True)  # no pyramid: derived overviews
    write_cog(d / "cog1lvlm.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
              geographic=True, no_overview=True, mask=True)  # ... and with a mask
    write_cog(d / "coglzw.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
              geographic=True, comp=5)  # LZW in the variant _lzw_decode reads
    write_cog(d / "cog16.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
              geographic=True, samples=1, bps=16)
    write_cog(d / "cognd.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
              geographic=True, nodata="0,0,0")
    write_cog(d / "cogalpha.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
              geographic=True, samples=4)  # RGB + ExtraSamples=ALPHA
    write_cog(d / "cogbadcrs.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
              geographic=True, geokeys=[1, 1, 0, 2, 1024, 0, 1, 2, 2048, 0, 1, 42304])
    write_cog(d / "cogpred.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
              geographic=True, predictor=2)  # deflate + horizontal differencing
    write_cog(d / "cogmask.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
              geographic=True, mask=True)  # 1-bit internal coverage mask
    write_cog(d / "cogbig.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
              geographic=True, mask=True, big=True)  # same file as BigTIFF (version 43)
    write_cog(d / "cogrot.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
              geographic=True, rotated=True)  # ModelTransformation: refused
    write_cog(d / "cogsub.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
              geographic=True, subifd=True)  # overviews as SubIFDs (tag 330)
    write_cog(d / "coghole.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
              geographic=True, holes=True)  # cog_merger count-0 hole blocks
    write_cog(d / "cogndneg.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
              geographic=True, fmt=1, nodata="-9999")  # sentinel off the sample range
    write_cog(d / "cogi8.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
              geographic=True, samples=1, bps=8, fmt=2)  # signed bytes
    write_cog(d / "cogi16.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
              geographic=True, samples=1, bps=16, fmt=2,
              sentinel=-9999, nodata="-9999")  # DEM: -9999 over the first column
    write_cog(d / "cogf32.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
              geographic=True, samples=1, bps=32, fmt=3,
              sentinel=np.nan, nodata="nan")  # float32 with NaN no-data
    write_cog(d / "cog12.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
              geographic=True, samples=1, bps=12)  # odd depth: refused, not guessed
    write_cog(d / "cogcx.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
              geographic=True, samples=1, bps=32, fmt=4)  # complex samples: refused
    write_cog(d / "cogpred3.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
              geographic=True, predictor=3)  # predictor 3 (IEEE float shuffle)
    write_cog(d / "cogfp2.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
              geographic=True, samples=1, bps=32, fmt=3,
              predictor=2)  # differencing declared on float samples
    write_cog(d / "cogndjunk.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
              geographic=True, nodata="N/A")  # GDAL_NODATA that is not a number
    (d / "junk.txt").write_text("this is not a GeoTIFF\n")
    return d


@pytest.fixture(scope="session")
def server(fixtures_dir):
    with RangeServer(str(fixtures_dir)) as s:
        yield s


@pytest.fixture()
def counter(server):
    import app.httprange as hr
    from app.main import clear_caches, pool
    pool.clear()  # force fresh upstream reads (the pool would reuse cached blocks)
    clear_caches()  # drop encoded-tile cache so bytes are really served
    hr._negative.clear()  # a previous 404 must not short-circuit this test
    hr._versions.clear()  # ... nor a stale version anchor
    hr._lastmod.clear()
    hr._checked.clear()
    hr._resolved.clear()  # ... nor a stale redirect target
    server.reset()
    return server.handler


@pytest.fixture(scope="session")
def client(fixtures_dir):
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        yield c


COG4326 = "cog4326.tif"
COG3857 = "cog3857.tif"
COG1L3857 = "cog1l3857.tif"  # the 3857 box again, level 0 only (no pyramid)
COGJPEG = "cogjpeg.tif"  # same content, JPEG-in-TIFF blocks (compression 7)
COGJPEG1 = "cogjpeg1.tif"  # same, single-band JPEG blocks
COGJPT = "cogjpt.tif"  # JPEG blocks without DQT + IFD tag 347 (the GDAL layout)
COGJPTD9 = "cogjpt9.tif"  # tag 347 terminated by the JPEG EOI (FF D9)
COGJPTNE = "cogjptne.tif"  # tag 347 with no terminator at all
COGJPTSELF = "cogjptself.tif"  # tag 347 declared, blocks keep their own DQT
COGJPTBAD = "cogjptbad.tif"  # tag 347 without SOI: the splice is not a JPEG
COGJPT1 = "cogjpt1.tif"  # single-band JPEG with tag 347 (the GRAY path)
COGZSTD = "cogzstd.tif"  # same content, ZSTD blocks (compression 50000)
COGPACK = "cogpack.tif"  # PackBits blocks (imagecodecs encoder)
COGLZW = "coglzw.tif"  # LZW blocks in the pure-Python decoder's variant
COGLZWLIB = "coglzwlib.tif"  # LZW blocks as libtiff/imagecodecs write them
COG16 = "cog16.tif"  # single band, 16-bit (values = uint8 pattern * 257)
COGND = "cognd.tif"  # GDAL_NODATA = 0
COGALPHA = "cogalpha.tif"  # RGB + alpha band (ExtraSamples), transparent hole
COGPRED = "cogpred.tif"  # deflate blocks with a horizontal-differencing predictor
COGMASK = "cogmask.tif"  # 1-bit internal coverage mask (west third no-data)
COGBIG = "cogbig.tif"  # the mask fixture again, written as BigTIFF (64-bit layout)
COGROT = "cogrot.tif"  # ModelTransformation instead of tiepoint: refused
COGWEBP = "cogwebp.tif"  # same content, lossless WebP blocks (compression 50001)
COGWEBPL = "cogwebpl.tif"  # lossy WebP blocks (VP8, 4:2:0)
COGWEBPA = "cogwebpa.tif"  # RGBA WebP blocks + ExtraSamples (transparent hole)
COGWEBP1 = "cogwebp1.tif"  # single-band WebP: refused at open (not RGB/RGBA)
COGJXL = "cogjxl.tif"  # lossless JPEG XL blocks (GDAL's codec id 50002)
COGJXLL = "cogjxll.tif"  # lossy JPEG XL blocks (distance 1.0, GDAL default)
COGJXLA = "cogjxla.tif"  # RGBA JPEG XL + ExtraSamples (transparent hole)
COGJXLG = "cogjxlg.tif"  # single-band gray JPEG XL (valid JXL, refused as WebP)
COGJXLD = "cogjxld.tif"  # JPEG XL blocks under the DNG 1.7 codec id 52546
COGJXL2 = "cogjxl2.tif"  # 2-sample (gray+alpha) JPEG XL: refused at open
COGJ2K = "cogj2k.tif"  # lossless JPEG 2000 blocks (GDAL's codec 34712)
COGJ2KL = "cogj2kl.tif"  # lossy JPEG 2000 blocks (9/7 wavelet, 40 dB)
COGJ2KA = "cogj2ka.tif"  # RGBA JPEG 2000 + ExtraSamples
COGJ2KG = "cogj2kg.tif"  # single-band gray J2K, under the 34713 "read only" code
COGJ2K2 = "cogj2k2.tif"  # 2-sample (gray+alpha) JPEG 2000: refused at open
COG1L = "cog1lvl.tif"  # level 0 only, no internal overview
COG1LM = "cog1lvlm.tif"  # level 0 only, with the 1-bit coverage mask
COGSUB = "cogsub.tif"  # overviews via SubIFDs (tag 330), empty IFD chain
COGHOLE = "coghole.tif"  # east block column: count 0 = no-data merger holes
COGBADCRS = "cogbadcrs.tif"  # geographic EPSG:42304 -> refused
COG2056 = "cog2056.tif"  # CH1903+/LV95: PROJ fallback
COGSWISS3857 = "cogswiss3857.tif"  # same box in EPSG:3857: parity twin of COG2056
COGNDNEG = "cogndneg.tif"  # GDAL_NODATA=-9999 on unsigned 8-bit: matches nothing
COGI8 = "cogi8.tif"  # SampleFormat 2 with bps 8: signed bytes
COGI16 = "cogi16.tif"  # int16 DEM: -9999 sentinel column + GDAL_NODATA=-9999
COGF32 = "cogf32.tif"  # float32 samples, NaN no-data (GDAL_NODATA="nan")
COG12 = "cog12.tif"  # 12-bit samples: refused, never silently read as 8-bit
COGCX = "cogcx.tif"  # SampleFormat 4 (complex): refused
COGPRED3 = "cogpred3.tif"  # predictor 3 (IEEE float shuffle): refused
COGFP2 = "cogfp2.tif"  # predictor 2 declared on float samples: refused
COGNDJUNK = "cogndjunk.tif"  # GDAL_NODATA="N/A": refused, not turned into 0.0

TILE_ZOOM = 14  # one tile covers the whole 4 km fixture footprint


def fixture_tile(z=TILE_ZOOM):
    """(z, x, y) of the tile containing the fixture centre, OSM/XYZ numbering."""
    lon = (BBOX4326[0] + BBOX4326[2]) / 2
    lat = (BBOX4326[1] + BBOX4326[3]) / 2
    x = int((lon + 180) / 360 * 2 ** z)
    y = int((1 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2 * 2 ** z)
    return z, x, y


def swiss_tile(z=TILE_ZOOM):
    """(z, x, y) of the tile holding the centre of the EPSG:2056 fixture."""
    gx0, gy0, gx1, gy1 = _swiss_grid()
    lon, lat = _pyproj.Transformer.from_crs(
        "EPSG:2056", "EPSG:4326", always_xy=True).transform((gx0 + gx1) / 2,
                                                            (gy0 + gy1) / 2)
    x = int((lon + 180) / 360 * 2 ** z)
    y = int((1 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2 * 2 ** z)
    return z, x, y


def tile_url(server, name, z, x, y, **q):
    qs = "&".join(f"{k}={v}" for k, v in q.items())
    u = f"{server.base}/{name}"
    return f"/c2t/{z}/{x}/{y}?u={u}" + (f"&{qs}" if qs else "")
