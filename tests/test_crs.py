"""CRS math unit tests: forward and inverse transforms pinned against PROJ
(cs2cs) golden values, crs_from_geokeys dispatch/refusal, unit scales."""
import math

import pytest

from app import crs

# x/y golden values, produced once with:
#   cs2cs -f %.8f +proj=longlat +ellps=E +no_defs +to +proj=<same as crs.py uses>
GOLDEN = [
    ({3072: 3943}, "EPSG:3943", [  # Lambert conformal 2SP, GRS80
        (1.5, 43.5, 1578697.19758124, 2256627.73119666),
        (-1.421406, 43.465922, 1342394.26387077, 2261171.38484356),
        (3.0, 43.0, 1700000.0, 2200000.0),
        (4.5, 42.5, 1823286.13502427, 2145560.77830782)]),
    ({3072: 2154}, "EPSG:2154", [  # Lambert-93
        (2.35, 48.855, 652306.14141471, 6861858.68007955),
        (-4.8, 48.4, 123608.94834434, 6839569.96880563),
        (7.7, 48.6, 1046350.12199384, 6843621.60701494)]),
    ({3072: 3946}, "EPSG:3946", [  # CC46
        (6.0, 46.0, 1932315.11214005, 5204375.66526719),
        (2.5, 45.5, 1660923.07324321, 5144553.47533847)]),
    ({3072: 32631}, "EPSG:32631", [  # UTM 31N
        (5.9, 58.0, 671384.63449062, 6432389.60876449),
        (0.2, 60.0, 343853.56603026, 6654716.39247272),
        (3.0, 0.0, 500000.0, 0.0)]),
    ({3072: 32733}, "EPSG:32733", [  # UTM 33S: false northing 1e7
        (15.0, -33.0, 500000.0, 6348713.05604032),
        (12.0, -28.0, 204956.51142260, 6899169.18185011)]),  # 3 deg from CM
    ({3072: 3857}, "EPSG:3857", [  # spherical web mercator
        (2.35, 48.855, 261600.80336419, 6250293.64590400),
        (-71.0667, -45.0, -7911108.85635833, -5621521.48619207)]),
    ({3072: 2056}, "EPSG:2056", [  # CH1903+/LV95: not in the table - this is
        # the PROJ fallback (extra "crs"); cs2cs here is the very PROJ that
        # pyproj embeds (both 9.9), values full precision: PROJ's oblique
        # Mercator inverse is iterative (round trip ~0.04 mas), so each pair
        # is the exact inverse of its x/y, embedded as cs2cs prints them
        (7.43958334069, 46.95240556591, 2600072.38970008818, 1200147.05558741465),
        (8.54170000839, 47.37690001016, 2683303.89373908751, 1247925.62789820321),
        (6.14320000609, 46.20440001024, 2500016.01611669641, 1117821.06973344414)]),
]


@pytest.mark.parametrize("keys,name,pts", GOLDEN)
def test_against_proj_golden(keys, name, pts):
    """Reprojection parity with PROJ: forward < 1 cm, inverse < 1 mm."""
    c = crs.crs_from_geokeys(keys)
    assert c.name == name
    for lon, lat, x, y in pts:
        fx, fy = c.to_xy(lon, lat)
        assert abs(fx - x) < 0.01 and abs(fy - y) < 0.01, (name, lon, lat)
        ilon, ilat = c.to_lonlat(x, y)
        assert abs(ilon - lon) < 1e-8 and abs(ilat - lat) < 1e-8


def test_lcc_single_standard_parallel():
    """One standard parallel: n = sin(lat_1) instead of the 2SP ratio."""
    c = crs.LambertConformalConic("1SP", crs.GRS80, 45.0, 0.0, 45.0, 0.0)
    for lon, lat, x, y in [(10.0, 50.0, 717947.76080407, 600982.36928682),
                           (-1.0, 50.0, -71975.52282653, 557068.00386245)]:
        fx, fy = c.to_xy(lon, lat)
        assert abs(fx - x) < 0.01 and abs(fy - y) < 0.01
        ilon, ilat = c.to_lonlat(x, y)
        assert abs(ilon - lon) < 1e-8 and abs(ilat - lat) < 1e-8


def test_geokeys_dispatch():
    assert crs.crs_from_geokeys({1024: 2}) is crs.GEOGRAPHIC  # 2048 defaults to 4326
    assert crs.crs_from_geokeys({1024: 2, 2048: 4979}) is crs.GEOGRAPHIC
    assert crs.crs_from_geokeys({3072: 4326}) is crs.GEOGRAPHIC
    assert crs.crs_from_geokeys({3072: 3078}).name == "EPSG:3078"  # same L93 again


def test_geokeys_refused():
    with pytest.raises(crs.UnsupportedCRS, match="unsupported geographic CRS 42304"):
        crs.crs_from_geokeys({1024: 2, 2048: 42304})
    with pytest.raises(crs.UnsupportedCRS, match="unsupported projected CRS 999999"):
        crs.crs_from_geokeys({3072: 999999})  # 2000 would pass: PROJ knows it
    with pytest.raises(crs.UnsupportedCRS, match="unsupported projected CRS None"):
        crs.crs_from_geokeys({})


def test_utm_zone_edges():
    c = crs.crs_from_geokeys({3072: 32601})
    assert c.lon_0 == pytest.approx(math.radians(-177)) and c.y_0 == 0.0
    c = crs.crs_from_geokeys({3072: 32660})
    assert c.lon_0 == pytest.approx(math.radians(177))
    c = crs.crs_from_geokeys({3072: 32701})
    assert c.y_0 == 10_000_000.0 and c.lon_0 == pytest.approx(math.radians(-177))


def test_user_defined_projections():
    """EPSG:32767 COGs carry the projection in GeoTIFF keys (3075 selects it);
    the three key sets below are what gdalinfo dumps show GDAL/ESRI write."""
    c = crs.crs_from_geokeys({3072: 32767, 3075: 8, 3078: 42.25, 3079: 43.75,  # GDAL LCC 2SP
                              3084: 3.0, 3085: 43.0, 3086: 1700000.0, 3087: 2200000.0})
    assert isinstance(c, crs.LambertConformalConic) and c.name == "user-defined LCC"
    x, y = c.to_xy(1.5, 43.5)  # same parameters as EPSG:3943: same coordinates
    assert abs(x - 1578697.19758124) < 0.01 and abs(y - 2256627.73119666) < 0.01
    c1 = crs.crs_from_geokeys({3072: 32767, 3075: 9, 3080: 0.0, 3081: 45.0,  # GDAL LCC 1SP
                               3082: 0.0, 3083: 0.0, 3092: 1.0})
    x, y = c1.to_xy(10.0, 50.0)  # same parameters as the 1SP golden above
    assert abs(x - 717947.76080407) < 0.01 and abs(y - 600982.36928682) < 0.01
    t = crs.crs_from_geokeys({3072: 32767, 3075: 1, 3080: 3.0, 3082: 500000.0,  # ESRI-style TM
                              3083: 0.0, 3092: 0.9996})
    assert isinstance(t, crs.TransverseMercator) and t.name == "user-defined TM"
    x, y = t.to_xy(5.9, 58.0)  # same parameters as EPSG:32631
    assert abs(x - 671384.63449062) < 0.01 and abs(y - 6432389.60876449) < 0.01
    assert t.to_lonlat(x, y)[0] == pytest.approx(5.9, abs=1e-8)  # far from CM
    with pytest.raises(crs.UnsupportedCRS):
        crs.crs_from_geokeys({3072: 32767, 3075: 99})  # unknown projection key


def test_geographic_and_clamp():
    assert crs.GEOGRAPHIC.to_xy(1.0, 2.0) == (1.0, 2.0)
    assert crs.GEOGRAPHIC.to_lonlat(1.0, 2.0) == (1.0, 2.0)
    _, y = crs.WEBMERCATOR.to_xy(0.0, 89.0)  # lat clamped at +-85.05
    assert y == crs.R_WEB * math.log(math.tan(math.pi / 4 + 85.05 * crs.D2R / 2))


def test_unit_scale():
    assert crs.unit_scale(9001) == 1.0  # metres
    assert crs.unit_scale(9003) == 0.3048  # US survey feet... international foot
    assert abs(crs.unit_scale(9002) - 1200 / 3937) < 1e-12  # US survey foot
    assert crs.unit_scale(9036) == 0.304800609601  # Indian foot
    assert crs.unit_scale(None) == 1.0  # key absent: assume metres
    assert crs.unit_scale(9093) == 1.0  # unknown unit: assume metres


def test_pyproj_fallback_geographic_shift():
    """A geographic datum that is not WGS-like (NAD27): the fallback's
    to_xy must apply the datum shift, not pass degrees through unchanged."""
    c = crs.crs_from_geokeys({1024: 2, 2048: 4267})
    assert c.name == "EPSG:4267"
    lon, lat = -95.4, 29.9  # inside NAD27's area of use
    x, y = c.to_xy(lon, lat)  # WGS84 lon/lat -> NAD27 lon/lat
    shift = math.hypot((x - lon) * math.cos(math.radians(lat)), y - lat) * 111_320
    assert 0.1 < shift < 100.0  # metres: shifted, but still a NAD27 neighbour
    rlon, rlat = c.to_lonlat(x, y)
    assert abs(rlon - lon) < 1e-8 and abs(rlat - lat) < 1e-8


def test_pyproj_fallback_batch_is_pointwise():
    """to_xy_many (warp control grid) must be bit-identical to the per-point
    contract - same PROJ operation, just looped in C."""
    c = crs.crs_from_geokeys({3072: 2056})
    lons, lats = [7.4, 8.5, 6.1], [46.9, 47.3, 46.2]
    bx, by = c.to_xy_many(lons, lats)
    for i, (lo, la) in enumerate(zip(lons, lats)):
        px, py = c.to_xy(lo, la)
        assert bx[i] == px and by[i] == py


def test_pyproj_absent_keeps_the_400(monkeypatch):
    """Without the extra, nothing changes: codes outside the table stay 400
    (the fallback must never be a silent second path that half works)."""
    monkeypatch.setattr(crs, "_pp", None)  # pretend the wheel is absent
    for keys, msg in (({3072: 2056}, "unsupported projected CRS 2056"),
                      ({1024: 2, 2048: 4267}, "unsupported geographic CRS 4267")):
        with pytest.raises(crs.UnsupportedCRS, match=msg):
            crs.crs_from_geokeys(keys)
    assert crs.crs_from_geokeys({3072: 3857}).name == "EPSG:3857"  # table untouched
