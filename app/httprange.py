import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin, urlsplit

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import __version__, config
from .security import check_url

USER_AGENT = f"cog2tms/{__version__} (https://github.com/cquest/cog2tms)"

# learned per host (netloc): whether it honors multipart/byteranges, so a
# server that rejected it once is never probed again by any new reader
_host_multi: dict[str, bool] = {}

# url -> (status, expiry): a recent remote 404/403 short-circuits further
# requests for that COG (tile clients retry aggressively)
_negative: dict[str, tuple[int, float]] = {}

# url -> final url last seen behind redirects: only the revalidation probe
# compares it, so a COG whose entry moved to another server is caught (reads
# themselves go straight to the settled target, they cannot see the move)
_resolved: dict[str, str] = {}


def _remember_resolved(url: str, final: str):
    if final != url:
        _resolved[url] = final


def _neg_check(url: str):
    item = _negative.get(url)
    if item:
        status, until = item
        if time.monotonic() < until:
            if status == 404:
                raise CogNotFound(f"{status} (recent)")
            raise CogFetch(f"{status} (recent)")
        del _negative[url]


def _neg_mark(url: str, status: int):
    if config.NEGATIVE_TTL > 0:
        _negative[url] = (status, time.monotonic() + config.NEGATIVE_TTL)


# url -> (validator, total size): last version of a remote file seen, taken
# from ETag/Last-Modified and Content-Range of the responses we already had
_versions: dict[str, tuple[str | None, int | None]] = {}

# url -> monotonic time of last version look (bounds revalidation probes)
_checked: dict[str, float] = {}

_vsession: requests.Session | None = None


def _probe_session() -> requests.Session:
    """Keeps its own keep-alive connection; no retry: a probe that fails only
    means the cached tile is served as-is."""
    global _vsession
    if _vsession is None:
        _vsession = requests.Session()
        _vsession.headers["User-Agent"] = USER_AGENT
    return _vsession


def _validator(r) -> str | None:
    v = r.headers.get("ETag") or r.headers.get("Last-Modified")
    return v[2:] if v and v.startswith("W/") else v  # weak/strong both identify


def _note_version(url: str, r):
    """Remember the version carried by a response; drift since the last one we
    saw means the file was replaced under the same URL."""
    v = _validator(r)
    if v is None:
        return
    known, total = _versions.get(url, (None, None))
    if known is not None and known != v:
        raise CogChanged(f"ETag changed: {known!r} -> {v!r}")
    _versions[url] = (v, total)
    _checked[url] = time.monotonic()


def _note_total(url: str, total: int):
    validator, known = _versions.get(url, (None, None))
    if known is not None and known != total:
        raise CogChanged(f"size changed: {known} -> {total}")
    _versions[url] = (validator, total)
    _checked[url] = time.monotonic()


def forget(url: str):
    """Drop version knowledge of a URL (used after a change was detected and
    evicted: the reopen must adopt the new version, not compare again)."""
    _versions.pop(url, None)
    _checked.pop(url, None)
    _resolved.pop(url, None)


def revalidate(url: str):
    """Raise CogChanged if the remote file was replaced since our last look.
    Costs one 1-byte GET carrying If-Range: a server whose validator no longer
    matches answers 200 + whole file, which we never read. Gated by
    VALIDATE_TTL and given a short timeout (VALIDATE_TIMEOUT), since it sits in
    front of the tile response."""
    if config.VALIDATE_TTL < 0 or not url.startswith(("http://", "https://")):
        return
    if time.monotonic() - _checked.get(url, 0.0) < config.VALIDATE_TTL:
        return
    validator, total = _versions.get(url, (None, None))
    if validator is None and total is None:
        return  # never read: the reads themselves will anchor the version
    _neg_check(url)
    hdr = {"Range": "bytes=0-0"}
    if validator:
        hdr["If-Range"] = validator
    # always from the requested url: behind a redirect, the entry is what tells
    # us the file moved to another server (reads go to the settled target)
    r, final = _follow(_probe_session(), url, "GET", hdr, stream=True,
                       timeout=(config.VALIDATE_TIMEOUT,) * 2)
    try:
        if r.status_code in (404, 403):
            _neg_mark(url, r.status_code)
            raise CogNotFound("404") if r.status_code == 404 else CogFetch("403")
        if final != _resolved.get(url, url):
            raise CogChanged(f"COG now served from {urlsplit(final).netloc}")
        _remember_resolved(url, final)
        if final == url:  # only a direct host speaks our validators
            _note_version(url, r)
        if r.status_code == 200:  # If-Range mismatch: new file, body unread
            raise CogChanged("If-Range mismatch")
        if r.status_code != 206:
            raise CogFetch(f"expected 206, got {r.status_code}")
        cr = r.headers.get("Content-Range")
        if cr:
            _note_total(url, int(cr.rsplit("/", 1)[1]))
    finally:
        r.close()
        _checked[url] = time.monotonic()


def _retrying_session() -> requests.Session:
    """Session with bounded transparent retries on transient upstream errors
    (urllib3: exponential backoff; Retry-After honored on 413/429/503)."""
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    retry = Retry(total=config.HTTP_RETRIES, connect=config.HTTP_RETRIES,
                  read=config.HTTP_RETRIES, status=config.HTTP_RETRIES,
                  backoff_factor=config.HTTP_BACKOFF,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=frozenset({"GET", "HEAD"}))
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


def _executor(n: int) -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=min(8, n), thread_name_prefix="range")


class CogNotFound(Exception):
    pass


class CogFetch(Exception):
    pass


class CogChanged(CogFetch):
    """The remote file was replaced under the same URL: readers, decoded blocks
    and encoded tiles built on the previous version are invalid."""


class CogRedirect(CogFetch):
    """A redirect target was refused (private/reserved address, scheme or host
    not allowed) or the chain is too long: the COG url itself is at fault."""


def request_timeout():
    return (config.HTTP_CONNECT_TIMEOUT, config.HTTP_READ_TIMEOUT)


def _follow(s: requests.Session, url: str, method: str, headers: dict,
            stream: bool, timeout: tuple | None = None) -> tuple[requests.Response, str]:
    """Request, following redirects ourselves: every Location is re-validated
    (scheme, private/reserved IPs, ALLOWED_HOSTS) before we connect to it, so a
    remote server cannot point us at an internal host. Returns the final
    response and the url it came from; 3xx bodies are never read."""
    seen = {url}
    for _ in range(config.MAX_REDIRECTS + 1):
        r = s.request(method, url, headers=headers,
                      timeout=timeout or request_timeout(),
                      stream=stream, allow_redirects=False)
        code = r.status_code
        if code not in (301, 302, 303, 307, 308):
            return r, url
        r.close()  # 3xx carries no image data: drop it
        loc = r.headers.get("Location")
        if not loc:
            raise CogFetch(f"{code} without Location")
        url = urljoin(r.url, loc)
        if url in seen:
            raise CogRedirect("redirect loop")
        seen.add(url)
        try:
            check_url(url)
        except ValueError as e:
            raise CogRedirect(f"redirect to {urlsplit(url).netloc}: {e}")
    raise CogRedirect(f"too many redirects (limit {config.MAX_REDIRECTS})")


class RangeReader:
    """HTTP byte-range access to a remote file (no full download). Large spans
    fetched by `prefetch` are kept so subsequent small reads (COG headers/IFDs)
    cost no extra round-trips."""

    def __init__(self, url: str, session: requests.Session | None = None):
        self.url = url
        self._target = url  # where reads go once the redirect chain is walked
        self._settled = False
        self._host = urlsplit(url).netloc
        self._s = session or _retrying_session()
        self._s.headers["User-Agent"] = USER_AGENT
        self._size = None
        self._multi_ok = bool(config.MULTIRANGE) and _host_multi.get(self._host, True)
        self._bufs: list[tuple[int, bytes]] = []  # (start, blob), newest first

    @property
    def _direct(self) -> bool:
        """data comes straight from the requested url: only there do our own
        ETag/If-Range mean something to the server holding the file"""
        return self._target == self.url

    def _fetch(self, method: str, headers: dict, stream: bool) -> requests.Response:
        """The redirect chain is walked once per reader: reads after that go
        straight to the settled host, and the revalidation probe re-walks the
        requested url to spot a move (`revalidate`)."""
        r, final = _follow(self._s, self._target if self._settled else self.url,
                           method, headers, stream)
        if final != self._target:
            self._target = final
            self._settled = final != self.url
            self._host = urlsplit(final).netloc
            self._multi_ok = bool(config.MULTIRANGE) and _host_multi.get(self._host, True)
            _remember_resolved(self.url, final)
        return r

    def _multi_failed(self):
        self._multi_ok = False
        _host_multi[self._host] = False

    def size(self) -> int:
        _neg_check(self.url)
        if self._size is None:
            r = self._fetch("HEAD", {}, stream=False)
            if r.status_code in (404, 403):
                _neg_mark(self.url, r.status_code)
                raise CogNotFound("404") if r.status_code == 404 else CogFetch("403")
            if r.status_code >= 400:
                raise CogFetch(f"HEAD {r.status_code}")
            self._size = int(r.headers["Content-Length"])
            if self._direct:
                _note_version(self.url, r)
            _note_total(self.url, self._size)
        return self._size

    def _raw(self, offset: int, length: int) -> bytes:
        _neg_check(self.url)
        validator = _versions.get(self.url, (None, None))[0] if self._direct else None
        hdr = {"Range": f"bytes={offset}-{offset + length - 1}"}
        if validator:
            hdr["If-Range"] = validator
        # stream once a validator is known: a replaced file answers 200 with
        # the whole file, and that body must never reach memory
        r = self._fetch("GET", hdr, stream=bool(validator))
        try:
            if r.status_code in (404, 403):
                _neg_mark(self.url, r.status_code)
                raise CogNotFound("404") if r.status_code == 404 else CogFetch("403")
            if r.status_code == 200 and validator:
                raise CogChanged("If-Range mismatch")  # file replaced
            if r.status_code != 206:
                raise CogFetch(f"expected 206, got {r.status_code}")
            if self._direct:
                _note_version(self.url, r)
            cr = r.headers.get("Content-Range")
            if cr:
                total = int(cr.rsplit("/", 1)[1])
                if self._size is None:  # free file size from the 206 answer
                    self._size = total
                _note_total(self.url, total)
            return r.content
        finally:
            r.close()

    def prefetch(self, offset: int, length: int):
        """Bulk-fetch one span (COG header zone, or the IFD zone near EOF on
        files whose IFDs sit at the end) and keep it for later reads."""
        if self._size is not None and self._size <= length:
            return  # whole file in one chunk anyway: pointless
        if not any(s <= offset and offset + length <= s + len(b) for s, b in self._bufs):
            self._bufs.insert(0, (offset, self._raw(offset, length)))
            del self._bufs[2:]

    def _from_buf(self, offset: int, length: int) -> bytes | None:
        for start, blob in self._bufs:
            if start <= offset and offset + length <= start + len(blob):
                i = offset - start
                return blob[i:i + length]
        return None

    def read(self, offset: int, length: int) -> bytes:
        seg = self._from_buf(offset, length)
        if seg is not None:
            return seg
        return self._raw(offset, length)

    def _multi_raw(self, fetch: list[tuple[int, int]]) -> list[bytes] | None:
        """One GET carrying several ranges (RFC 7233 multipart/byteranges).
        Returns blobs aligned with `fetch`, or None if the server does not
        honor multi-ranges (S3 answers 200 + whole file, nginx answers the
        first range only): the body is NOT consumed unless the answer is a
        real multipart 206, so huge files are never downloaded."""
        _neg_check(self.url)
        # no If-Range here: servers that refuse multi-ranges answer 200 + whole
        # file whatever the validator, which would look like a version change.
        # Drift is still caught below by the response ETag and part totals.
        hdr = ", ".join(f"{o}-{o + n - 1}" for o, n in fetch)
        r = self._fetch("GET", {"Range": "bytes=" + hdr}, stream=True)
        try:
            if r.status_code in (404, 403):
                _neg_mark(self.url, r.status_code)
                raise CogNotFound("404") if r.status_code == 404 else CogFetch("403")
            if self._direct:
                _note_version(self.url, r)  # headers only: body untouched below
            ctype = r.headers.get("Content-Type") or ""
            if r.status_code != 206 or "multipart/byteranges" not in ctype:
                self._multi_failed()
                return None
            body = r.content  # multipart 206 confirmed: body is range-sized
        finally:
            r.close()
        m = re.search(r'boundary="?([^";]+)"?', ctype)
        parts = {}
        if m:
            for p in body.split(b"--" + m.group(1).encode())[1:]:
                if p.startswith(b"--"):
                    break  # closing boundary reached
                head, sep, data = p.partition(b"\r\n\r\n")
                cr = re.search(rb"bytes (\d+)-(\d+)/(\d+|\*)", head)
                if not sep or not cr:
                    self._multi_failed()
                    return None
                start, end = int(cr.group(1)), int(cr.group(2))
                if cr.group(3) != b"*":
                    total = int(cr.group(3))
                    if self._size is None:
                        self._size = total
                    _note_total(self.url, total)
                parts[start] = data[:-2] if data.endswith(b"\r\n") else data
        if all(o in parts and len(parts[o]) == n for o, n in fetch):
            _host_multi[self._host] = True
            return [parts[o] for o, n in fetch]
        self._multi_failed()  # parts did not match the request exactly
        return None

    def read_many(self, spans: list[tuple[int, int]]) -> list[bytes]:
        """Fetch several (offset, length) spans: served from prefetched chunks
        when possible, otherwise coalesced nearby, then fetched in parallel
        (or in one multi-range GET when the server supports it)."""
        if not spans:
            return []
        out: list[bytes | None] = [None] * len(spans)
        todo = []  # (span index, offset, length)
        for i, (off, ln) in enumerate(spans):
            seg = self._from_buf(off, ln)
            if seg is None:
                todo.append((i, off, ln))
            else:
                out[i] = seg
        if not todo:
            return out
        order = sorted(range(len(todo)), key=lambda k: todo[k][1])
        groups = []  # [start, end, [todo positions]]
        for k in order:
            _, off, ln = todo[k]
            if groups and off - groups[-1][1] <= config.MERGE_GAP:
                groups[-1][1] = max(groups[-1][1], off + ln)
                groups[-1][2].append(k)
            else:
                groups.append([off, off + ln, [k]])
        fetch = [(g[0], g[1] - g[0]) for g in groups]
        blobs = self._multi_raw(fetch) if self._multi_ok and len(fetch) > 1 else None
        if blobs is None:
            if len(fetch) > 1:
                with _executor(len(fetch)) as ex:
                    blobs = list(ex.map(lambda a: self._raw(*a), fetch))
            else:
                blobs = [self._raw(*a) for a in fetch]
        for (start, _, keys), blob in zip(groups, blobs):
            for k in keys:
                _, off, ln = todo[k]
                out[todo[k][0]] = blob[off - start: off - start + ln]
        return out

    def close(self):
        self._bufs.clear()
        self._s.close()


class LocalReader:
    """Same byte-range interface for local paths (tests/dev)."""

    def __init__(self, path: str):
        if not os.path.isfile(path):
            raise CogNotFound(path)
        self._f = open(path, "rb")
        self._f.seek(0, os.SEEK_END)
        self._size = self._f.tell()

    def size(self) -> int:
        return self._size

    def read(self, offset: int, length: int) -> bytes:
        self._f.seek(offset)
        return self._f.read(length)

    def prefetch(self, offset: int, length: int):  # seeks are cheap
        pass

    def read_many(self, spans):
        return [self.read(o, n) for o, n in spans]

    def close(self):
        self._f.close()


def open_reader(u: str):
    if u.startswith(("http://", "https://")):
        return RangeReader(u)
    return LocalReader(u)  # local path accepted for dev/tests only
