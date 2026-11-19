"""SQLite persistence of COG head spans across service restarts.

The spans `RangeReader.prefetch` bulk-fetches (the head zone, or the IFD zone
near EOF on unoptimized TIFFs) are saved with the version the response carried
— and **only** when it carried a change detector (ETag or Last-Modified): a
span we cannot re-identify later is worthless. Reopening adopts the span with
its validator, so `_versions` is anchored and the very next range read carries
If-Range: a file replaced while we were down is caught exactly as if the span
had stayed in memory (200 where a 206 is expected -> CogChanged -> purge).
The parsed IFD list needs no row of its own: parsing the cached span is free;
it is the fetch that costs. Security notes: the row key carries the full url
(so never pass signed urls through a persistent cache), the file must be
owned by the service user (it holds image bytes at rest), a corrupt or
unwritable database is a miss rather than an outage, and rows are bounded in
count and age whatever the url flood."""
import sqlite3
import threading
import time

from . import config

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None
_path = ""
# bound what flood of distinct urls can accumulate: 4096 spans <= ~0.5 GB
_MAX_ROWS = 4096


def _db() -> sqlite3.Connection | None:
    """One connection per process, reopened when the setting moves (tests).
    WAL: uvicorn workers are separate processes sharing the file. A cache that
    cannot be opened (corrupt file, unwritable path) is a miss, never a
    failure: the service keeps working without persistence."""
    global _conn, _path
    if not config.HEAD_CACHE:
        return None
    if _conn is None or _path != config.HEAD_CACHE:
        if _conn is not None:
            _conn.close()
            _conn = None
        try:
            _conn = sqlite3.connect(config.HEAD_CACHE, timeout=5.0,
                                    check_same_thread=False)
            _conn.execute("PRAGMA journal_mode = WAL")
            _conn.execute("CREATE TABLE IF NOT EXISTS heads ("
                          "url TEXT NOT NULL, off INTEGER NOT NULL,"
                          "validator TEXT NOT NULL, total INTEGER NOT NULL,"
                          "blob BLOB NOT NULL, saved_at REAL NOT NULL,"
                          "PRIMARY KEY (url, off))")
            _conn.commit()
            _path = config.HEAD_CACHE
        except sqlite3.Error:
            return None
    return _conn


def load(url: str, off: int) -> tuple[str, int, bytes] | None:
    """(validator, total size, bytes) of a span saved by a direct sighting."""
    db = _db()
    if db is None:
        return None
    with _lock:
        try:
            return db.execute("SELECT validator, total, blob FROM heads"
                              " WHERE url = ? AND off = ?", (url, off)).fetchone()
        except sqlite3.Error:
            return None


def save(url: str, off: int, validator: str | None, total: int, blob: bytes):
    """Keep one span of a COG we can re-identify; no validator, no row."""
    if not config.HEAD_CACHE or validator is None or total <= 0:
        return
    db = _db()
    if db is None:
        return
    now = time.time()
    with _lock:
        try:
            db.execute("INSERT OR REPLACE INTO heads VALUES (?,?,?,?,?,?)",
                       (url, off, validator, total, blob, now))
            # stale rows expire; and whatever the url count, disk stays bounded
            db.execute("DELETE FROM heads WHERE saved_at < ?", (now - 30 * 86400,))
            db.execute("DELETE FROM heads WHERE rowid NOT IN"
                       " (SELECT rowid FROM heads ORDER BY rowid DESC"
                       " LIMIT ?)", (_MAX_ROWS,))
            db.commit()
        except sqlite3.Error:
            pass


def forget(url: str):
    """Drop every saved span of a URL (version change detected, or entry moved)."""
    db = _db()
    if db is None:
        return
    with _lock:
        try:
            db.execute("DELETE FROM heads WHERE url = ?", (url,))
            db.commit()
        except sqlite3.Error:
            pass
