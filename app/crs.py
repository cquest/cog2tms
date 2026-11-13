import math

D2R = math.pi / 180.0


class UnsupportedCRS(Exception):
    pass


class Ellipsoid:
    __slots__ = ("a", "f", "e2", "ep2")

    def __init__(self, a, f):
        self.a, self.f = a, f
        self.e2 = f * (2 - f)
        self.ep2 = self.e2 / (1 - self.e2)


GRS80 = Ellipsoid(6378137.0, 1 / 298.257222101)
WGS84 = Ellipsoid(6378137.0, 1 / 298.257223563)
R_WEB = 6378137.0  # EPSG:3857 uses a sphere


class CRS:
    name = "?"

    def to_xy(self, lon, lat):
        raise NotImplementedError

    def to_lonlat(self, x, y):
        raise NotImplementedError


class Geographic(CRS):
    name = "EPSG:4326"

    def to_xy(self, lon, lat):
        return lon, lat

    def to_lonlat(self, x, y):
        return x, y


class WebMercator(CRS):
    name = "EPSG:3857"

    def to_xy(self, lon, lat):
        lat = max(-85.05, min(85.05, lat))
        return (R_WEB * lon * D2R,
                R_WEB * math.log(math.tan(math.pi / 4 + lat * D2R / 2)))

    def to_lonlat(self, x, y):
        return (x / R_WEB / D2R,
                (2 * math.atan(math.exp(y / R_WEB)) - math.pi / 2) / D2R)


class LambertConformalConic(CRS):
    """Lambert conformal conic, 1 or 2 standard parallels (Snyder, ellipsoidal)."""

    def __init__(self, name, ell, lat_0, lon_0, lat_1, lat_2, x_0=0.0, y_0=0.0):
        self.name = name
        self.ell, self.x_0, self.y_0 = ell, x_0, y_0
        self.lon_0 = lon_0 * D2R
        e = math.sqrt(ell.e2)

        def m(p):
            return math.cos(p) / math.sqrt(1 - ell.e2 * math.sin(p) ** 2)

        def t(p):
            s = math.sin(p)
            return math.tan(math.pi / 4 - p / 2) / ((1 - e * s) / (1 + e * s)) ** (e / 2)

        p1, p2, p0 = lat_1 * D2R, (lat_2 if lat_2 else lat_1) * D2R, lat_0 * D2R
        if abs(p1 - p2) > 1e-12:
            self.n = (math.log(m(p1)) - math.log(m(p2))) / (math.log(t(p1)) - math.log(t(p2)))
        else:
            self.n = math.sin(p1)
        self.F = m(p1) / (self.n * t(p1) ** self.n)
        self.rho0 = ell.a * self.F * t(p0) ** self.n
        self.e = e

    def _rho(self, lat):
        p = lat * D2R
        s = math.sin(p)
        t = math.tan(math.pi / 4 - p / 2) / ((1 - self.e * s) / (1 + self.e * s)) ** (self.e / 2)
        return self.ell.a * self.F * t ** self.n

    def to_xy(self, lon, lat):
        n = self.n
        rho = self._rho(lat)
        th = n * (lon * D2R - self.lon_0)
        if n < 0:
            rho, th = -rho, -th
        return (self.x_0 + rho * math.sin(th),
                self.y_0 + self.rho0 - rho * math.cos(th))

    def to_lonlat(self, x, y):
        n = self.n
        dx, dy = x - self.x_0, self.rho0 - (y - self.y_0)
        if n < 0:
            dx, dy = -dx, -dy
        rho = math.copysign(math.hypot(dx, dy), n)
        th = math.atan2(dx, dy)
        t = (rho / (self.ell.a * self.F)) ** (1 / n)
        p = math.pi / 2 - 2 * math.atan(t)
        for _ in range(6):  # iterate for the ellipsoid
            s = math.sin(p)
            p = math.pi / 2 - 2 * math.atan(t * ((1 - self.e * s) / (1 + self.e * s)) ** (self.e / 2))
        lon = self.lon_0 + th / n
        return lon / D2R, p / D2R


class TransverseMercator(CRS):
    """Ellipsoidal transverse mercator (Snyder series); covers UTM."""

    def __init__(self, name, ell, lon_0, k_0, x_0=500000.0, y_0=0.0):
        self.name = name
        self.ell, self.k_0, self.x_0, self.y_0 = ell, k_0, x_0, y_0
        self.lon_0 = lon_0 * D2R
        e2 = ell.e2
        self.m0 = 1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256
        self.m1 = 3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024
        self.m2 = 15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024
        self.m3 = 35 * e2 ** 3 / 3072

    def _M(self, p):
        e2 = self.ell.e2
        a = self.ell.a
        return a * (self.m0 * p - self.m1 * math.sin(2 * p)
                    + self.m2 * math.sin(4 * p) - self.m3 * math.sin(6 * p))

    def to_xy(self, lon, lat):
        ell, e2 = self.ell, self.ell.e2
        p, d = lat * D2R, (lon * D2R - self.lon_0)
        sp, cp = math.sin(p), math.cos(p)
        N = ell.a / math.sqrt(1 - e2 * sp * sp)
        T, C = math.tan(p) ** 2, self.ell.ep2 * cp ** 2
        A = d * cp
        x = self.x_0 + self.k_0 * N * (A + (1 - T + C) * A ** 3 / 6
                                       + (5 - 18 * T + T * T + 72 * C - 58 * self.ell.ep2) * A ** 5 / 120)
        y = self.y_0 + self.k_0 * (self._M(p) + N * math.tan(p) * (A * A / 2
                                    + (5 - T + 9 * C + 4 * C * C) * A ** 4 / 24
                                    + (61 - 58 * T + T * T + 600 * C - 330 * self.ell.ep2) * A ** 6 / 720))
        return x, y

    def to_lonlat(self, x, y):
        ell, e2 = self.ell, self.ell.e2
        e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
        M1 = (y - self.y_0) / self.k_0
        mu = M1 / (ell.a * self.m0)
        p1 = (mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * math.sin(2 * mu)
              + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * math.sin(4 * mu)
              + 151 * e1 ** 3 / 96 * math.sin(6 * mu) + 1097 * e1 ** 4 / 512 * math.sin(8 * mu))
        sp, cp = math.sin(p1), math.cos(p1)
        C1 = self.ell.ep2 * cp ** 2
        T1, NN = math.tan(p1) ** 2, ell.a / math.sqrt(1 - e2 * sp * sp)
        R1 = ell.a * (1 - e2) / (1 - e2 * sp * sp) ** 1.5
        D = (x - self.x_0) / (NN * self.k_0)
        p = (p1 - NN * math.tan(p1) / R1 * (D * D / 2
             - (5 + 3 * T1 + 10 * C1 - 4 * C1 ** 2 - 9 * self.ell.ep2) * D ** 4 / 24
             + (61 + 90 * T1 + 298 * C1 + 45 * T1 * T1 - 252 * self.ell.ep2 - 3 * C1 ** 2) * D ** 6 / 720))
        d = (D - (1 + T1 - C1) * D ** 3 / 6 + (5 - 2 * C1 + C1 * C1 - 9 * T1) * D ** 5 / 120) / cp
        return (self.lon_0 + d) / D2R, p / D2R


# geographic datums handled as WGS84 (all WGS84 realizations or <1m offsets)
GEO_WGS_LIKE = {4326, 4258, 4979, 4171, 6318, 6319, 4269}

GEOGRAPHIC = Geographic()
WEBMERCATOR = WebMercator()

# minimal EPSG subset: {code: LambertConformalConic params}
_LCC_EPSG = {
    2154: (46.5, 3.0, 49.0, 44.0, 700000.0, 6600000.0),   # Lambert-93
    3942: (42.0, 3.0, 41.25, 42.75, 1700000.0, 1200000.0),  # CC42
    3943: (43.0, 3.0, 42.25, 43.75, 1700000.0, 2200000.0),  # CC43
    3944: (44.0, 3.0, 43.25, 44.75, 1700000.0, 3200000.0),
    3945: (45.0, 3.0, 44.25, 45.75, 1700000.0, 4200000.0),
    3946: (46.0, 3.0, 45.25, 46.75, 1700000.0, 5200000.0),
    3947: (47.0, 3.0, 46.25, 47.75, 1700000.0, 6200000.0),
    3948: (48.0, 3.0, 47.25, 48.75, 1700000.0, 7200000.0),
    3078: (46.5, 3.0, 49.0, 44.0, 700000.0, 6600000.0),   # same as L93
}

_LINEAR_UNITS = {9001: 1.0, 9002: 1200 / 3937, 9003: 0.3048, 9036: 0.304800609601}


def crs_from_geokeys(key: dict) -> CRS:
    """Build a CRS from a parsed GeoKeyDirectory (1024 ModelType, 2048 geographic,
    3072 projected CS code). Supports a small EPSG subset."""
    if key.get(1024, 1) == 2:  # geographic
        if key.get(2048, 4326) in GEO_WGS_LIKE:
            return GEOGRAPHIC
        raise UnsupportedCRS(f"unsupported geographic CRS {key.get(2048)}")

    code = key.get(3072)
    if code == 3857:
        return WEBMERCATOR
    if code in (4326,) or code in GEO_WGS_LIKE and key.get(1024) == 2:
        return GEOGRAPHIC
    if code in _LCC_EPSG:
        lat_0, lon_0, s1, s2, x_0, y_0 = _LCC_EPSG[code]
        return LambertConformalConic(f"EPSG:{code}", GRS80, lat_0, lon_0, s1, s2, x_0, y_0)
    if 32601 <= code <= 32660 or 32701 <= code <= 32760:
        zone = code % 32700 if code > 32700 else code - 32600
        return TransverseMercator(f"EPSG:{code}", WGS84, 6 * zone - 183,
                                  0.9996, 500000.0, 10000000.0 if code > 32700 else 0.0)
    if code == 32767:  # user-defined: build from GeoTIFF projection keys
        pc = key.get(3075)
        if pc in (2, 3):
            return LambertConformalConic(
                "user-defined LCC", GRS80, key.get(3085, 0.0), key.get(3081, key.get(3084, 0.0)),
                key.get(3078, 0.0), key.get(3079, 0.0), key.get(3082, 0.0), key.get(3083, 0.0))
        if pc in (13, 26, 27):
            return TransverseMercator("user-defined TM", WGS84, key.get(3084, 0.0),
                                      key.get(3086, 1.0), key.get(3082, 500000.0), key.get(3083, 0.0))
    if code == 4326:
        return GEOGRAPHIC
    raise UnsupportedCRS(f"unsupported projected CRS {code}")


def unit_scale(linear_unit_key: int | None) -> float:
    """Meters per source-CRS unit."""
    return _LINEAR_UNITS.get(linear_unit_key or 9001, 1.0)
