import ipaddress
import os
import socket
from urllib.parse import urlparse
from functools import lru_cache

ALLOWED_HOSTS = [h.strip() for h in os.getenv("ALLOWED_HOSTS", "").split(",") if h.strip()]
# set in tests/dev to reach fixture servers on loopback
ALLOW_PRIVATE_NET = os.getenv("ALLOW_PRIVATE_NET", "") == "1"

@lru_cache
def check_url(u: str) -> None:
    """http(s) only; reject private/reserved targets after DNS (anti-SSRF);
    optional deployment host allowlist (comma-separated ALLOWED_HOSTS)."""
    p = urlparse(u)
    if p.scheme not in ("http", "https"):
        raise ValueError("only http(s) URLs are accepted")
    host = p.hostname
    if not host:
        raise ValueError("missing host")
    if ALLOWED_HOSTS and host not in ALLOWED_HOSTS:
        raise ValueError("host not allowed")
    try:
        infos = socket.getaddrinfo(host, p.port or (443 if p.scheme == "https" else 80))
    except socket.gaierror:
        raise ValueError("cannot resolve host")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global and not ALLOW_PRIVATE_NET:
            raise ValueError("private or reserved address")
