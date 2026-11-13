"""Offline fixtures: small COGs written by a pure-python TIFF writer, served
over a local Range-capable HTTP server that counts transferred bytes."""
import http.server
import os
import socketserver
import struct
import threading
import zlib
from functools import lru_cache

import numpy as np
import pytest

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


def _tiles(arr):
    th = tw = TILE
    h, w = arr.shape[:2]
    blobs = []
    for tj in range(h // th):
        for ti in range(w // tw):
            blob = np.ascontiguousarray(arr[tj * th:(tj + 1) * th,
                                           ti * tw:(ti + 1) * tw]).tobytes()
            blobs.append(zlib.compress(blob))
    return blobs


def write_cog(path, ox, oy, psx, psy, geographic):
    """Classic tiled little-endian TIFF with a 2x internal overview.
    Layout: header | level-0 tiles | level-1 tiles | value areas | IFD0 | IFD1."""
    lvls = [(pattern(), 0), (pattern(half=True), 1)]
    blobs = [_tiles(arr) for arr, _ in lvls]

    # rough layout: header | tiles | areas | ifds ; areas are small so offsets
    # are computed exactly in a second pass
    hdr = 8
    tile_off = hdr
    tile_sizes = [[len(b) for b in lv] for lv in blobs]
    sizes0 = sum(sum(s) for s in tile_sizes)

    geokeys = ([1, 1, 0, 2, 1024, 0, 1, 2, 2048, 0, 1, 4326] if geographic else
               [1, 1, 0, 1, 1024, 0, 1, 1, 2048, 0, 1, 4979, 2052, 0, 1, 9001,
                3072, 0, 1, 3857, 3076, 0, 1, 9001])

    def areas_for(w, h, sub):
        nt = (w // TILE) * (h // TILE)
        a = bytearray()
        a.extend(b"\x08\x00" * 3)                       # BitsPerSample
        a.extend(struct.pack("<" + "I" * nt, *([0] * nt)))   # TileOffsets (patched)
        a.extend(struct.pack("<" + "I" * nt, *[len(b) for b in blobs[0 if sub == 0 else 1]]))
        a.extend(struct.pack("<3d", psx, psy, 0.0))
        a.extend(struct.pack("<6d", 0, 0, 0, ox, oy, 0.0))
        a.extend(struct.pack("<" + "H" * len(geokeys), *geokeys))
        return bytes(a), nt

    area_blobs, nts = [], []
    for (arr, sub) in lvls:
        ab, nt = areas_for(arr.shape[1], arr.shape[0], sub)
        area_blobs.append(ab)
        nts.append(nt)

    def ifd_bytes(i, next_off):
        w = lvls[i][0].shape[1]
        h = lvls[i][0].shape[0]
        sub = lvls[i][1]
        nt = nts[i]
        base_area = area_start + sum(len(x) for x in area_blobs[:i])
        ent = [
            (254, 4, 1, struct.pack("<I", sub)[:4]),
            (256, 4, 1, struct.pack("<I", w)[:4]),
            (257, 4, 1, struct.pack("<I", h)[:4]),
            (258, 3, 3, None), (259, 3, 1, struct.pack("<H", 8)[:4]),
            (262, 3, 1, struct.pack("<H", 2)[:4]),
            (277, 3, 1, struct.pack("<H", 3)[:4]),
            (284, 3, 1, struct.pack("<H", 1)[:4]),
            (322, 3, 1, struct.pack("<H", TILE)[:4]),
            (323, 3, 1, struct.pack("<H", TILE)[:4]),
            (324, 4, nt, None), (325, 4, nt, None),
            (33550, 12, 3, None), (33922, 12, 6, None),
            (34735, 3, len(geokeys), None),
        ]
        # value-area offsets within area_blobs[i]: bps(6) | offs(4nt) | cnts(4nt) | scale(24) | tie(48) | keys(2*len)
        p = base_area
        pos = {}
        for code, size in ((258, 6), (324, 4 * nt), (325, 4 * nt),
                           (33550, 24), (33922, 48), (34735, 2 * len(geokeys))):
            pos[code] = p
            p += size
        out = struct.pack("<H", len(ent))
        for code, typ, cnt, raw in ent:
            if raw is None:
                raw = struct.pack("<I", pos[code])
            out += struct.pack("<HHI", code, typ, cnt) + raw[:4].ljust(4, b"\0")
        out += struct.pack("<I", next_off)
        return out

    n_ent = 15
    ifd_size = 2 + 12 * n_ent + 4
    area_start = tile_off + sizes0
    ifd0 = area_start + sum(len(a) for a in area_blobs)
    ifd1 = ifd0 + ifd_size

    # patch tile offsets inside area_blobs (absolute file offsets)
    cur = tile_off
    for i, lv in enumerate(blobs):
        ab = bytearray(area_blobs[i])
        w = lvls[i][0].shape[1]
        nt = nts[i]
        offs_pos = 6  # after BitsPerSample
        for j, b in enumerate(lv):
            ab[offs_pos + 4 * j: offs_pos + 4 * j + 4] = struct.pack("<I", cur)
            cur += len(b)
        area_blobs[i] = bytes(ab)

    out = bytearray(struct.pack("<2sHI", b"II", 42, ifd0))
    for lv in blobs:
        for b in lv:
            out.extend(b)
    for ab in area_blobs:
        out.extend(ab)
    out.extend(ifd_bytes(0, ifd1))
    out.extend(ifd_bytes(1, 0))
    assert len(out) == ifd1 + ifd_size, (len(out), ifd1 + ifd_size)
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
