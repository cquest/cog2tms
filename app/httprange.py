import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.client import parse_headers
from io import BytesIO
from urllib.parse import urljoin, urlsplit

from curl_cffi import CurlError, CurlHttpVersion, CurlOpt
from curl_cffi.requests import Session

from . import __version__, config
from .security import check_url

USER_AGENT = f"cog2tms/{__version__} (https://github.com/cquest/cog2tms)"

# status line of an HTTP/1.1 answer or an HTTP/2 one (no minor version)
_STATUS = re.compile(rb"HTTP/[\d.]+ (\d{3})")

# statuses worth another attempt (the rest is the origin's answer as-is)
_RETRY_STATUS = (429, 500, 502, 503, 504)

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

_local = threading.local()


def _header_line(line: bytes) -> int:
    """curl reports every header line as it arrives, even when the body then
    breaks: that is how a 200 answer to a range request stays recognizable after
    we cut the transfer short (it means the file was replaced, not fetched).
    This replaces curl_cffi's own header capture, so the lines are kept here and
    parsed in `_perform`; it runs in the thread performing the transfer, hence
    the thread-local box."""
    box = getattr(_local, "answer", None)
    if box is None:
        return len(line)
    box[1].write(line)
    m = _STATUS.match(line)
    if m:
        box[0] = int(m.group(1))
    return len(line)


def _session() -> Session:
    """One curl handle per thread: libcurl easy handles are not thread-safe,
    and the range pool threads keep their connection warm across windows.
    `HTTP2=0` pins HTTP/1.1, which also rebuilds an existing thread's handle."""
    s = getattr(_local, "session", None)
    if s is not None and _local.http2 == config.HTTP2:
        return s
    if s is not None:
        s.close()
    kw = {"curl_options": {CurlOpt.HEADERFUNCTION: _header_line}}
    if not config.HTTP2:
        kw["http_version"] = int(CurlHttpVersion.V1_1)
    s = _local.session = Session(**kw)
    _local.http2 = config.HTTP2
    s.headers["User-Agent"] = USER_AGENT
    return s


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
    matches answers 200 + whole file, which is cut short by the byte cap.
    Gated by VALIDATE_TTL and given a short timeout (VALIDATE_TIMEOUT), since it
    sits in front of the tile response."""
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
    # us the file moved to another server (reads go straight to the settled target)
    try:
        r, final = _follow(url, "GET", hdr, cap=64, retries=0,
                           timeout=(config.VALIDATE_TIMEOUT,) * 2)
    except _Aborted as e:
        if e.status == 200:  # 200 + whole file, cut off: body never buffered
            raise CogChanged("If-Range mismatch") from e
        raise CogFetch(f"probe answered {e.status} beyond our cap") from e
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
    _checked[url] = time.monotonic()


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


class _Aborted(Exception):
    """A body we must not buffer: a range request answered 200 (the whole file,
    i.e. our If-Range no longer matches) or more than we asked for. `status` is
    what the peer said — the header callback makes it known even though the
    transfer was cut short, so the reason is reported, not a fetch failure."""

    def __init__(self, status: int | None):
        super().__init__(f"answer {status} to a range request")
        self.status = status


class _Resp:
    """A finished transfer: status, the headers captured by `_header_line`, and
    the body — its size bounded by the cap given to `_perform`."""

    __slots__ = ("status_code", "headers", "url", "body", "proto")

    def __init__(self, status: int, headers, url: str, body: bytes, proto: int):
        self.status_code = status
        self.headers = headers
        self.url = url
        self.body = body
        # 1 = HTTP/1.0, 2 = HTTP/1.1, 3 = HTTP/2, 30 = HTTP/3
        self.proto = proto


def request_timeout():
    return (config.HTTP_CONNECT_TIMEOUT, config.HTTP_READ_TIMEOUT)


def _perform(method: str, url: str, headers: dict, cap: int,
             timeout: tuple) -> _Resp:
    """One transfer whose body is bounded two ways, so a multi-GB COG is never
    pulled into memory: the header callback reveals the status before a single
    body byte arrives (a 200 to a range request is the whole file → stop at
    once) and the write callback stops the transfer past `cap` bytes (returning
    a short value from it does not abort, raising does)."""
    chunks: list[bytes] = []
    got = 0
    answer = _local.answer = [None, BytesIO()]

    def sink(data: bytes) -> int:
        nonlocal got
        chunks.append(data)
        got += len(data)
        if answer[0] == 200 or got > cap:
            raise _Aborted(answer[0])
        return len(data)

    try:
        r = _session().request(method, url, headers=headers, timeout=timeout,
                               allow_redirects=False, accept_encoding=None,
                               content_callback=sink)
    except CurlError as e:
        if answer[0] == 200:  # our abort, or a truncated whole-file answer
            raise _Aborted(200) from e
        raise
    _, _, rest = answer[1].getvalue().partition(b"\r\n")  # first line is status
    return _Resp(answer[0] or r.status_code, parse_headers(BytesIO(rest)), url,
                 b"".join(chunks), r.http_version)


def _retry_pause(headers, attempt: int):
    """Wait before the next attempt: the origin's Retry-After wins over our own
    backoff, and a polite 429 must never park a worker for minutes either."""
    ra = headers.get("Retry-After") if headers else None
    if ra:
        try:
            time.sleep(min(float(ra), 10.0))
            return
        except ValueError:
            pass  # HTTP-date form: fall back to our backoff
    time.sleep(config.HTTP_BACKOFF * 2 ** attempt)


def _request(method: str, url: str, headers: dict, cap: int, timeout: tuple,
             retries: int) -> _Resp:
    """Bounded retries on transient upstream failures (429 + 5xx) and on
    transport errors; a body that broke the cap is never retried."""
    for attempt in range(retries + 1):
        resp_headers = None
        try:
            r = _perform(method, url, headers, cap, timeout)
            if r.status_code not in _RETRY_STATUS:
                return r
            resp_headers, why = r.headers, str(r.status_code)
        except CurlError as e:
            why = str(e)
        if attempt == retries:
            raise CogFetch(f"{method} {url}: {why}")
        _retry_pause(resp_headers, attempt)


def _follow(url: str, method: str, headers: dict, cap: int,
            timeout: tuple | None = None,
            retries: int | None = None) -> tuple[_Resp, str]:
    """Request, following redirects ourselves: every Location is re-validated
    (scheme, private/reserved IPs, ALLOWED_HOSTS) before we connect to it, so a
    remote server cannot point us at an internal host. Returns the final
    response and the url it came from; 3xx bodies are never followed into."""
    retries = config.HTTP_RETRIES if retries is None else retries
    seen = {url}
    for _ in range(config.MAX_REDIRECTS + 1):
        r = _request(method, url, headers, cap, timeout or request_timeout(), retries)
        code = r.status_code
        if code not in (301, 302, 303, 307, 308):
            return r, url
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


_pool: ThreadPoolExecutor | None = None
_pool_lock = threading.Lock()


def _ranges() -> ThreadPoolExecutor:
    """One process-wide pool instead of one per window: its threads keep living
    between renders, and those threads are where our warm connections (and
    thread-local curl handles) are."""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ThreadPoolExecutor(max_workers=8,
                                           thread_name_prefix="range")
    return _pool


class RangeReader:
    """HTTP byte-range access to a remote file (no full download). Large spans
    fetched by `prefetch` are kept so subsequent small reads (COG headers/IFDs)
    cost no extra round-trips."""

    def __init__(self, url: str):
        self.url = url
        self._target = url  # where reads go once the redirect chain is walked
        self._settled = False
        self._host = urlsplit(url).netloc
        self.proto: int | None = None  # wire version of the last response
        self._size = None
        self._multi_ok = bool(config.MULTIRANGE) and _host_multi.get(self._host, True)
        self._bufs: list[tuple[int, bytes]] = []  # (start, blob), newest first

    @property
    def _direct(self) -> bool:
        """data comes straight from the requested url: only there do our own
        ETag/If-Range mean something to the server holding the file"""
        return self._target == self.url

    def _fetch(self, method: str, headers: dict, cap: int) -> _Resp:
        """The redirect chain is walked once per reader: reads after that go
        straight to the settled host, and the revalidation probe re-walks the
        requested url to spot a move (`revalidate`)."""
        r, final = _follow(self._target if self._settled else self.url,
                           method, headers, cap)
        self.proto = r.proto
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
            r = self._fetch("HEAD", {}, cap=0)
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
        # a replaced file answers 200 with the whole file: the cap cuts that
        # off after `length` bytes, whatever the server is willing to send
        try:
            r = self._fetch("GET", hdr, cap=length + 4096)
        except _Aborted as e:
            if e.status != 200:
                raise CogFetch(f"{e.status} answered beyond the range") from e
            if validator:
                raise CogChanged("If-Range mismatch") from e
            raise CogFetch("server ignored our Range") from e
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
        return r.body

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
        honor multi-ranges (S3 answers 200 + whole object, nginx answers the
        first range only): the cap stops a whole-file answer at about the size
        of the ranges we asked for."""
        _neg_check(self.url)
        # no If-Range here: servers that refuse multi-ranges answer 200 + whole
        # file whatever the validator, which would look like a version change.
        # Drift is still caught below by the response ETag and part totals.
        hdr = ", ".join(f"{o}-{o + n - 1}" for o, n in fetch)
        # part headers (boundary + Content-Range) are the only legitimate excess
        cap = sum(n for _, n in fetch) + 2048 * len(fetch) + 4096
        try:
            r = self._fetch("GET", {"Range": "bytes=" + hdr}, cap=cap)
        except _Aborted:  # 200 + whole object, or a body beyond the ranges
            self._multi_failed()
            return None
        if r.status_code in (404, 403):
            _neg_mark(self.url, r.status_code)
            raise CogNotFound("404") if r.status_code == 404 else CogFetch("403")
        if self._direct:
            _note_version(self.url, r)
        ctype = r.headers.get("Content-Type") or ""
        if r.status_code != 206 or "multipart/byteranges" not in ctype:
            self._multi_failed()
            return None
        body = r.body  # multipart 206 confirmed: body is range-sized
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
                blobs = list(_ranges().map(lambda a: self._raw(*a), fetch))
            else:
                blobs = [self._raw(*a) for a in fetch]
        for (start, _, keys), blob in zip(groups, blobs):
            for k in keys:
                _, off, ln = todo[k]
                out[todo[k][0]] = blob[off - start: off - start + ln]
        return out

    def close(self):
        self._bufs.clear()  # sessions are thread-local and shared: not ours to close


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
