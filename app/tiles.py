"""Render EPSG:3857 XYZ tiles from a CogFile (any source CRS in the supported
subset). Non-affine warps use a 17x17 control grid (inverse projection at grid
nodes) interpolated bilinearly per pixel, then bilinear resampling of pixels."""
import cv2
import numpy as np

from . import crs

cv2.setNumThreads(1)  # 256 px warps: OpenCV's internal threading gains nothing and
# would oversubscribe the cores between uvicorn workers

TILE_EPS = 1e-6
_K = 17


class EmptyTile(Exception):
    pass


class MissingRescale(Exception):
    pass


def _tile_lat(y: int, z: int) -> float:
    n = np.pi * (1 - 2 * y / (1 << z))
    return float(np.degrees(np.arctan(np.sinh(n))))


def tile_web_bounds(z: int, x: int, y: int):
    """(X0, Y0, X1, Y1) in EPSG:3857 meters for an OSM/XYZ tile."""
    n = 1 << z
    x0, y0 = crs.WEBMERCATOR.to_xy(x / n * 360 - 180, _tile_lat(y + 1, z))
    x1, y1 = crs.WEBMERCATOR.to_xy((x + 1) / n * 360 - 180, _tile_lat(y, z))
    return x0, y0, x1, y1


def web_bbox_lonlat(bbox):
    a = crs.WEBMERCATOR.to_lonlat(bbox[0], bbox[1])
    b = crs.WEBMERCATOR.to_lonlat(bbox[2], bbox[3])
    return min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])


def tile_exists(cog, z, x, y) -> bool:
    bx0, by0, bx1, by1 = cog.web_bbox()
    tx0, ty0, tx1, ty1 = tile_web_bounds(z, x, y)
    return (tx0 < bx1 - TILE_EPS and tx1 > bx0 + TILE_EPS
            and ty0 < by1 - TILE_EPS and ty1 > by0 + TILE_EPS)


def _pick_level(cog, span_main_px: float, size: int) -> int:
    """Largest overview whose window stays >= size/2 pixels."""
    w0 = cog.levels[0].width
    level = 0
    while level + 1 < len(cog.levels) and span_main_px * cog.levels[level + 1].width / w0 >= size / 2:
        level += 1
    return level


def render_tile(cog, z: int, x: int, y: int, size: int,
                bidx: list[int] | None = None, rescale: tuple[float, float] | None = None):
    """Return (rgb uint8 (size,size,3), valid bool (size,size))."""
    if not tile_exists(cog, z, x, y):
        raise EmptyTile()

    lvl0 = cog.levels[0]
    tx0, ty0, tx1, ty1 = tile_web_bounds(z, x, y)

    # source pixel window covering the tile (sample the 9 key points)
    cols, rows = [], []
    for t in (0.0, 0.5, 1.0):
        for s in (0.0, 0.5, 1.0):
            lon, lat = crs.WEBMERCATOR.to_lonlat(tx0 + (tx1 - tx0) * t, ty0 + (ty1 - ty0) * s)
            sx, sy = cog.crs.to_xy(lon, lat)
            cols.append((sx - cog.ox) / cog.psx)
            rows.append((cog.oy - sy) / cog.psy)
    c0, c1, r0, r1 = min(cols), max(cols), min(rows), max(rows)
    if c1 <= c0 or r1 <= r0:
        raise EmptyTile()

    level = _pick_level(cog, c1 - c0, size)
    lvl = cog.levels[level]
    scale = lvl0.width / lvl.width  # exact decimation factor of this overview
    # window past the raster on purpose: _window pads with valid=False, which keeps
    # tiles straddling the COG edge transparent instead of smearing edge pixels
    wc0 = int(np.floor(c0 / scale)) - 1
    wr0 = int(np.floor(r0 / scale)) - 1
    wc1 = int(np.ceil(c1 / scale)) + 1
    wr1 = int(np.ceil(r1 / scale)) + 1

    data, valid = cog.window(level, wc0, wr0, wc1, wr1)
    if bidx:
        data = data[..., [i - 1 for i in bidx]]
    if rescale or data.dtype.itemsize > 1:
        if not rescale:
            raise MissingRescale("16/32-bit data requires rescale=min,max")
        lo, hi = rescale
        data = np.clip((data.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1e-9), 0, 255)
        data = data.astype(np.uint8)
    if data.shape[2] == 1:
        data = np.repeat(data, 3, axis=2)
    elif data.shape[2] > 3:
        data = data[..., :3]

    if isinstance(cog.crs, crs.WebMercator):
        u = np.arange(size) + 0.5
        X = tx0 + (tx1 - tx0) * u / size
        Y = ty1 + (ty0 - ty1) * u / size
        fx = np.broadcast_to((X - cog.ox) / (cog.psx * scale) - wc0, (size, size))
        fy = np.broadcast_to(((cog.oy - Y) / (cog.psy * scale) - wr0)[:, None], (size, size))
    else:
        fx, fy = _warp_grid(cog, tx0, ty0, tx1, ty1, size, scale, wc0, wr0)

    return _sample(data, valid, fx, fy)


def _warp_grid(cog, tx0, ty0, tx1, ty1, size, scale, wc0, wr0):
    """Per-pixel source (col, row) via bilinear on a _K x _K control grid."""
    C = np.empty((_K, _K))
    R = np.empty((_K, _K))
    for iv in range(_K):
        Y = ty1 + (ty0 - ty1) * (iv * (size - 1) / (_K - 1) + 0.5) / size
        for iu in range(_K):
            X = tx0 + (tx1 - tx0) * (iu * (size - 1) / (_K - 1) + 0.5) / size
            lon, lat = crs.WEBMERCATOR.to_lonlat(X, Y)
            sx, sy = cog.crs.to_xy(lon, lat)
            C[iv, iu] = (sx - cog.ox) / (cog.psx * scale) - wc0
            R[iv, iu] = (cog.oy - sy) / (cog.psy * scale) - wr0

    p = np.linspace(0, _K - 1, size)
    i0 = np.clip(np.floor(p).astype(np.intp), 0, _K - 2)
    i1 = i0 + 1
    t = p - i0
    tv, tu = t[:, None], t[None, :]

    def interp(M):
        return ((1 - tv) * ((1 - tu) * M[i0][:, i0] + tu * M[i0][:, i1])
                + tv * ((1 - tu) * M[i1][:, i0] + tu * M[i1][:, i1]))

    return interp(C), interp(R)


def _sample(data, valid, fx, fy):
    """cv2.remap bilinear warp with mask-aware renormalization: data*valid and
    valid are remapped as float32, so out-of-window pixels contribute 0 to both
    (constant border) and the division restores valid pixels exactly, leaving 0
    under invalid ones. Returns (uint8 rgb, valid)."""
    mapx = np.ascontiguousarray(fx, np.float32)
    mapy = np.ascontiguousarray(fy, np.float32)
    a = cv2.remap((data * valid[..., None]).astype(np.float32), mapx, mapy,
                  cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    wv = cv2.remap(valid.astype(np.float32), mapx, mapy,
                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    a /= np.maximum(wv, TILE_EPS)[..., None]
    np.clip(a, 0, 255, out=a)
    return a.astype(np.uint8), wv >= 0.5
