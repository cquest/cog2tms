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

os.environ["ALLOW_PRIVATE_NET"] = "1"  # fixture server is on loopback

W = H = 512
TILE = 128

# fixture footprint (near Paris) in lon/lat and its EPSG:3857 box
BBOX4326 = (2.30, 48.83, 2.36, 48.88)  # w,s,e,n
RWEB = 6378137.0


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


def _block(arr, comp, lzw_ours, predictor, bps):
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
    if comp == 50000:
        return _ic.zstd_encode(raw)
    if comp == 5:  # TIFF LZW: "ours" = the variant the pure-Python decoder reads
        return _lzw_encode(raw) if lzw_ours else _ic.lzw_encode(raw)
    if comp == 32773:
        return _ic.packbits_encode(raw)
    return zlib.compress(raw)


def _tiles(arr, comp=8, lzw_ours=True, predictor=1, bps=8):
    th = tw = TILE
    h, w = arr.shape[:2]
    return [_block(arr[tj * th:(tj + 1) * th, ti * tw:(ti + 1) * tw],
                   comp, lzw_ours, predictor, bps)
            for tj in range(h // th) for ti in range(w // tw)]


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


def write_cog(path, ox, oy, psx, psy, geographic, comp=8, samples=3, bps=8,
              nodata=None, geokeys=None, lzw_ours=True, predictor=1, mask=False,
              rotated=False):
    """Classic tiled little-endian TIFF: level 0 + a 2x internal overview, plus
    optional extras - a 1-bit internal coverage mask (NewSubfileType & 4), a
    horizontal-differencing predictor, GDAL_NODATA, ExtraSamples=ALPHA, or a
    ModelTransformation (rotated raster, which the reader must refuse).
    Layout: header | tiles of each IFD | value areas | IFDs."""
    src = pattern() if samples == 3 else pattern4() if samples == 4 else pattern()[..., 0]
    src_h = (pattern(half=True) if samples == 3 else pattern4(half=True)
             if samples == 4 else pattern(half=True)[..., 0])
    if bps == 16:
        src, src_h = src.astype(np.uint16) * 257, src_h.astype(np.uint16) * 257
    phot = 6 if (comp == 7 and samples == 3) else (1 if samples == 1 else 2)
    lvls = [dict(arr=a, sub=sub, samples=samples, bps=bps, comp=comp, phot=phot, geo=True)
            for a, sub in ((src, 0), (src_h, 1))]
    if mask:  # coverage: the west third of the raster is no-data
        bits = np.zeros((H, W), np.uint8)
        bits[:, W // 3:] = 1
        lvls.append(dict(arr=bits, sub=4, samples=1, bps=1, comp=8, phot=1, geo=False))
    for l in lvls:  # a lone SHORT fits inline in the IFD; 1-bit masks never difference
        l["bps_len"] = 2 * l["samples"] if l["samples"] > 1 else 0
        l["pred"] = predictor if l["bps"] in (8, 16) else 1
    blobs = [_tiles(l["arr"], l["comp"], lzw_ours, l["pred"], l["bps"]) for l in lvls]

    # rough layout: header | tiles | areas | ifds ; areas are small so offsets
    # are computed exactly in a second pass
    hdr = 8
    tile_off = hdr
    sizes0 = sum(len(b) for lv in blobs for b in lv)

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
            (258, 3, l["samples"], None if l["bps_len"] else struct.pack("<H", l["bps"])[:4]),
            (259, 3, 1, struct.pack("<H", l["comp"])[:4]),
            (262, 3, 1, struct.pack("<H", l["phot"])[:4]),
            (277, 3, 1, struct.pack("<H", l["samples"])[:4]),
        ]
        if l["samples"] == 4:  # ExtraSamples=ALPHA: the reader turns band 4 into a mask
            ent.append((338, 3, 1, struct.pack("<H", 2)[:4]))
        ent += [(284, 3, 1, struct.pack("<H", 1)[:4]),  # planar contig
                (317, 3, 1, struct.pack("<H", l["pred"])[:4]),
                (322, 3, 1, struct.pack("<H", TILE)[:4]),
                (323, 3, 1, struct.pack("<H", TILE)[:4]),
                (324, 4, nt, None), (325, 4, nt, None)]
        if l["geo"]:
            ent += [(33550, 12, 3, None), (33922, 12, 6, None), (34735, 3, len(geokeys), None)]
            if rotated:  # ModelTransformation: the reader refuses rotated rasters
                ent.append((34264, 12, 16, None))
            if nodata:  # GDAL_NODATA (ASCII, inline when 4 bytes or fewer)
                ent.append((42113, 2, len(nodata),
                            struct.pack("<4s", nodata.encode()) if len(nodata) <= 4 else None))
        # area layout: bps (0 if inline) | offs | cnts | then for geo IFDs:
        # scale | tiepoint | geokeys | transformation | nodata
        areas = [(258, l["bps_len"]), (324, 4 * nt), (325, 4 * nt)]
        if l["geo"]:
            areas += [(33550, 24), (33922, 48), (34735, 2 * len(geokeys))]
            if rotated:
                areas.append((34264, 128))
            if nodata and len(nodata) > 4:
                areas.append((42113, len(nodata)))
        return ent, areas

    def area_bytes(i):
        l, nt = lvls[i], nts[i]
        a = bytearray()
        a.extend(struct.pack("<" + "H" * l["samples"], *([l["bps"]] * l["samples"]))
                 if l["bps_len"] else b"")
        a.extend(struct.pack("<" + "I" * nt, *([0] * nt)))   # TileOffsets (patched)
        a.extend(struct.pack("<" + "I" * nt, *[len(b) for b in blobs[i]]))
        if l["geo"]:
            a.extend(struct.pack("<3d", psx, psy, 0.0))
            a.extend(struct.pack("<6d", 0, 0, 0, ox, oy, 0.0))
            a.extend(struct.pack("<" + "H" * len(geokeys), *geokeys))
            if rotated:
                m = [psx, 0, 0, 0, 0, -psy, 0, 0, 0, 0, 0, 0, ox, oy, 0, 1]
                a.extend(struct.pack("<16d", *m))
            if nodata and len(nodata) > 4:
                a.extend(nodata.encode())  # GDAL_NODATA beyond 4 bytes: out of line
        return bytes(a)

    area_blobs = [area_bytes(i) for i in range(len(lvls))]

    def ifd_bytes(i, next_off):
        ent, areas = ifd_entries(i)
        p = area_start + sum(len(x) for x in area_blobs[:i])
        pos = {}
        for code, size in areas:
            pos[code] = p
            p += size
        out = struct.pack("<H", len(ent))
        for code, typ, cnt, raw in ent:
            if raw is None:
                raw = struct.pack("<I", pos[code])
            out += struct.pack("<HHI", code, typ, cnt) + raw[:4].ljust(4, b"\0")
        out += struct.pack("<I", next_off)
        return out

    area_start = tile_off + sizes0
    ifd_offs, p = [], area_start + sum(len(a) for a in area_blobs)
    for i in range(len(lvls)):  # IFD sizes follow from their entry lists
        ifd_offs.append(p)
        p += 2 + 12 * len(ifd_entries(i)[0]) + 4

    # patch tile offsets inside area_blobs (absolute file offsets)
    cur = tile_off
    for i, lv in enumerate(blobs):
        ab = bytearray(area_blobs[i])
        base = lvls[i]["bps_len"]
        for j, b in enumerate(lv):
            ab[base + 4 * j: base + 4 * j + 4] = struct.pack("<I", cur)
            cur += len(b)
        area_blobs[i] = bytes(ab)

    out = bytearray(struct.pack("<2sHI", b"II", 42, ifd_offs[0]))
    for lv in blobs:
        for b in lv:
            out.extend(b)
    for ab in area_blobs:
        out.extend(ab)
    for i in range(len(lvls)):
        out.extend(ifd_bytes(i, ifd_offs[i + 1] if i + 1 < len(ifd_offs) else 0))
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
    write_cog(d / "cogjpeg.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
              geographic=True, comp=7)
    write_cog(d / "cogjpeg1.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
              geographic=True, comp=7, samples=1)
    if _ic is not None:
        write_cog(d / "cogzstd.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
                  geographic=True, comp=50000)
        write_cog(d / "cogpack.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
                  geographic=True, comp=32773)
        write_cog(d / "coglzwlib.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
                  geographic=True, comp=5, lzw_ours=False)  # imagecodecs/libtiff LZW
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
    write_cog(d / "cogrot.tif", w0, n0, (e0 - w0) / W, (n0 - s0) / H,
              geographic=True, rotated=True)  # ModelTransformation: must be refused
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
COGJPEG = "cogjpeg.tif"  # same content, JPEG-in-TIFF blocks (compression 7)
COGJPEG1 = "cogjpeg1.tif"  # same, single-band JPEG blocks
COGZSTD = "cogzstd.tif"  # same content, ZSTD blocks (compression 50000)
COGPACK = "cogpack.tif"  # PackBits blocks (imagecodecs encoder)
COGLZW = "coglzw.tif"  # LZW blocks in the pure-Python decoder's variant
COGLZWLIB = "coglzwlib.tif"  # LZW blocks as libtiff/imagecodecs write them
COG16 = "cog16.tif"  # single band, 16-bit (values = uint8 pattern * 257)
COGND = "cognd.tif"  # GDAL_NODATA = 0
COGALPHA = "cogalpha.tif"  # RGB + alpha band (ExtraSamples), transparent hole
COGPRED = "cogpred.tif"  # deflate blocks with a horizontal-differencing predictor
COGMASK = "cogmask.tif"  # 1-bit internal coverage mask (west third no-data)
COGROT = "cogrot.tif"  # ModelTransformation instead of tiepoint: refused
COGBADCRS = "cogbadcrs.tif"  # geographic EPSG:42304 -> refused

TILE_ZOOM = 14  # one tile covers the whole 4 km fixture footprint


def fixture_tile(z=TILE_ZOOM):
    """(z, x, y) of the tile containing the fixture centre, OSM/XYZ numbering."""
    lon = (BBOX4326[0] + BBOX4326[2]) / 2
    lat = (BBOX4326[1] + BBOX4326[3]) / 2
    x = int((lon + 180) / 360 * 2 ** z)
    y = int((1 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2 * 2 ** z)
    return z, x, y


def tile_url(server, name, z, x, y, **q):
    qs = "&".join(f"{k}={v}" for k, v in q.items())
    u = f"{server.base}/{name}"
    return f"/c2t/{z}/{x}/{y}?u={u}" + (f"&{qs}" if qs else "")
