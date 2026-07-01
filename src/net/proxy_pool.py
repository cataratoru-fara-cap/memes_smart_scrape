"""
Rotating proxy pool backed by proxifly/free-proxy-list.

https://github.com/proxifly/free-proxy-list publishes a large, frequently
refreshed list of free proxies. Free proxies are unreliable — most are dead or
slow — so this pool fetches many, hands them out round-robin, drops the ones
that fail (``mark_bad``), and can validate candidates against a test URL before
use.

Network calls use ``requests`` (imported lazily); list parsing is pure and
tested offline. Enable via env (``PROXY_POOL_ENABLED=true``) or construct
directly. The pool yields proxy URLs like ``http://1.2.3.4:8080`` and offers
adapters for both Playwright (``to_playwright``) and requests (``to_requests``).
"""

from __future__ import annotations

import json
import os
import random
import re
import threading
from typing import Dict, List, Optional

# proxifly CDN (jsdelivr is steadier than raw.githubusercontent).
_CDN = "https://cdn.jsdelivr.net/gh/proxifly/free-proxy-list@main/proxies"
DEFAULT_LIST_URL = f"{_CDN}/protocols/http/data.txt"

_LINE = re.compile(r"^(?:(?P<scheme>\w+)://)?(?P<host>[\d.]+):(?P<port>\d+)$")


def _get_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes")


class ProxyPool:
    """A rotating, failure-aware pool of proxy URLs."""

    def __init__(
        self,
        url: str = DEFAULT_LIST_URL,
        protocols: Optional[List[str]] = None,
        max_proxies: int = 50,
        test_url: str = "https://knowyourmeme.com/robots.txt",
        timeout: int = 15,
    ):
        self.url = url
        self.protocols = [p.lower() for p in (protocols or ["http"])]
        self.max_proxies = max_proxies
        self.test_url = test_url
        self.timeout = timeout
        self._proxies: List[str] = []
        self._bad: set[str] = set()
        self._idx = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    @staticmethod
    def parse(raw: str, protocols: List[str], is_json: bool = False) -> List[str]:
        """Parse a proxifly ``data.txt`` or ``data.json`` body into proxy URLs."""
        out: List[str] = []
        protocols = [p.lower() for p in protocols]

        if is_json:
            try:
                items = json.loads(raw)
            except json.JSONDecodeError:
                return out
            for it in items if isinstance(items, list) else []:
                proxy = str(it.get("proxy", "")).strip()
                proto = str(it.get("protocol", "")).lower()
                if proxy and (not protocols or proto in protocols):
                    out.append(proxy)
            return out

        for line in raw.splitlines():
            m = _LINE.match(line.strip())
            if not m:
                continue
            scheme = (m.group("scheme") or "http").lower()
            if protocols and scheme not in protocols:
                continue
            out.append(f"{scheme}://{m.group('host')}:{m.group('port')}")
        return out

    def load(self, raw: Optional[str] = None) -> int:
        """
        Populate the pool. If ``raw`` is given it is parsed directly (offline);
        otherwise the list is fetched from ``self.url``. Returns the count.
        """
        if raw is None:
            import requests
            resp = requests.get(self.url, timeout=self.timeout)
            resp.raise_for_status()
            raw = resp.text
        proxies = self.parse(raw, self.protocols, is_json=self.url.endswith(".json"))
        random.shuffle(proxies)
        with self._lock:
            self._proxies = proxies[: self.max_proxies]
            self._bad.clear()
            self._idx = 0
        return len(self._proxies)

    # ------------------------------------------------------------------
    def get(self) -> Optional[str]:
        """Next live proxy (round-robin), or None if the pool is exhausted."""
        with self._lock:
            n = len(self._proxies)
            for _ in range(n):
                proxy = self._proxies[self._idx % n]
                self._idx += 1
                if proxy not in self._bad:
                    return proxy
            return None

    def mark_bad(self, proxy: str) -> None:
        with self._lock:
            self._bad.add(proxy)

    def size(self) -> int:
        with self._lock:
            return len(self._proxies) - len(self._bad)

    # ------------------------------------------------------------------
    @staticmethod
    def to_playwright(proxy: str) -> Dict[str, str]:
        return {"server": proxy}

    @staticmethod
    def to_requests(proxy: str) -> Dict[str, str]:
        return {"http": proxy, "https": proxy}

    def validate(self, proxy: str) -> bool:
        """True if ``proxy`` can reach ``test_url`` within the timeout."""
        try:
            import requests
            r = requests.get(self.test_url, proxies=self.to_requests(proxy),
                             timeout=self.timeout)
            return r.status_code < 500
        except Exception:  # noqa: BLE001 - any failure => unusable
            return False


def proxy_pool_from_env() -> Optional[ProxyPool]:
    """Build a ProxyPool from env, or None when PROXY_POOL_ENABLED is unset."""
    if not _get_bool("PROXY_POOL_ENABLED", False):
        return None
    protocols = [p.strip() for p in os.getenv("PROXY_POOL_PROTOCOLS", "http").split(",") if p.strip()]
    return ProxyPool(
        url=os.getenv("PROXY_LIST_URL", DEFAULT_LIST_URL),
        protocols=protocols,
        max_proxies=int(os.getenv("PROXY_POOL_MAX", "50")),
        test_url=os.getenv("PROXY_TEST_URL", "https://knowyourmeme.com/robots.txt"),
    )


if __name__ == "__main__":  # quick manual check: python -m src.net.proxy_pool
    pool = ProxyPool(url=os.getenv("PROXY_LIST_URL", DEFAULT_LIST_URL),
                     protocols=os.getenv("PROXY_POOL_PROTOCOLS", "http").split(","))
    print(f"Fetching {pool.url} ...")
    print(f"Loaded {pool.load()} proxies. Validating up to 10 ...")
    ok = 0
    for _ in range(min(10, pool.size())):
        p = pool.get()
        if p and pool.validate(p):
            ok += 1
            print(f"  OK   {p}")
        elif p:
            pool.mark_bad(p)
            print(f"  dead {p}")
    print(f"{ok} working out of the sample.")
