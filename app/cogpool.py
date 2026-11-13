import asyncio
from collections import OrderedDict

from .geotiff import CogFile
from .httprange import open_reader


class ReaderPool:
    """LRU+TTL cache of open COGs: opening a remote COG costs a few byte-range
    reads (TIFF headers + overview layouts), so cache Readers, not tiles."""

    def __init__(self, maxsize: int, ttl: float, max_concurrent_reads: int):
        self._maxsize = maxsize
        self._ttl = ttl
        self._items: OrderedDict[str, tuple[float, CogFile]] = OrderedDict()
        self._locks: dict[str, asyncio.Lock] = {}
        self._sem = asyncio.Semaphore(max_concurrent_reads)

    async def get(self, url: str) -> CogFile:
        loop = asyncio.get_running_loop()
        lock = self._locks.setdefault(url, asyncio.Lock())
        async with lock:
            item = self._items.get(url)
            if item and loop.time() - item[0] < self._ttl:
                self._items.move_to_end(url)
                return item[1]
            if item:
                self._close(item[1])
                del self._items[url]
            cog = await asyncio.to_thread(CogFile, open_reader(url))
            self._items[url] = (loop.time(), cog)
            while len(self._items) > self._maxsize:
                _, oldest = self._items.popitem(last=False)
                self._close(oldest)
            return cog

    async def render(self, cog: CogFile, z: int, x: int, y: int, size: int,
                     bidx=None, rescale=None, blackmax=None):
        from . import tiles
        async with self._sem:
            return await asyncio.to_thread(tiles.render_tile, cog, z, x, y, size, bidx, rescale, blackmax)

    def invalidate(self, url: str):
        """Drop the reader of a COG whose remote file was replaced: its header,
        index slices and decoded blocks all belong to the previous version."""
        item = self._items.pop(url, None)
        if item:
            self._close(item[1])

    def clear(self):
        for _, cog in self._items.values():
            self._close(cog)
        self._items.clear()

    @staticmethod
    def _close(cog: CogFile):
        try:
            cog.close()
        except Exception:
            pass
