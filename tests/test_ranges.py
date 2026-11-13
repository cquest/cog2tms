"""Byte-range proof: a tile render must fetch only a fraction of the file and
every upstream request must carry a Range header."""
from .conftest import COG4326, fixture_tile, tile_url


def test_render_reads_only_needed_bytes(client, server, counter, fixtures_dir):
    file_size = (fixtures_dir / COG4326).stat().st_size
    z, x, y = fixture_tile()
    r = client.get(tile_url(server, COG4326, z, x, y))
    assert r.status_code == 200
    served = counter.served
    assert 0 < served < file_size, f"served {served} of {file_size} bytes"
    # sanity: the headers alone already move some bytes; the tile blocks are a
    # small minority of the file (overview path, not the full raster)
    assert served < file_size * 0.5


def test_all_upstream_requests_use_range(client, server, counter):
    z, x, y = fixture_tile()
    client.get(tile_url(server, COG4326, z, x, y, format="jpeg"))
    assert counter.requests, "server saw no request"
    bad = [(p, r) for p, r in counter.requests if not (r and r.startswith("bytes="))]
    assert not bad, f"requests without Range: {bad}"
