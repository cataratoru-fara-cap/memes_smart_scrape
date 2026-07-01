"""
Render a Know Your Meme page to a flat list of DOM nodes for the student model.

Each node is ``{id, text, tag, bbox:[x,y,w,h]}``, which feeds straight into
``encoding`` / ``meme_encoding``. Playwright is imported lazily; this module
imports fine without it (the dataset builder / live inference need it, tests do
not).

Proxy support mirrors the annotation pipeline: set ANNOTATION_PROXY_URL (and
optionally ANNOTATION_PROXY_USERNAME / ANNOTATION_PROXY_PASSWORD) when the
target IP is banned.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

# Collect visible elements that carry their *own* text, with geometry. Kept as a
# JS string so all the DOM work happens in one round-trip.
_COLLECT_JS = r"""
() => {
  const out = [];
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_ELEMENT);
  let el = walker.currentNode;
  while (el) {
    // own text = direct text node children only (not descendants)
    let own = "";
    for (const child of el.childNodes) {
      if (child.nodeType === Node.TEXT_NODE) own += child.nodeValue;
    }
    own = own.replace(/\s+/g, " ").trim();
    if (own.length >= 1) {
      const r = el.getBoundingClientRect();
      if (r.width > 0 && r.height > 0) {
        out.push({
          tag: el.tagName.toLowerCase(),
          text: own,
          bbox: [r.x + window.scrollX, r.y + window.scrollY, r.width, r.height],
        });
      }
    }
    el = walker.nextNode();
  }
  return out;
}
"""


def _proxy_from_env() -> Optional[Dict[str, str]]:
    server = os.getenv("ANNOTATION_PROXY_URL")
    if not server:
        return None
    proxy = {"server": server}
    if os.getenv("ANNOTATION_PROXY_USERNAME"):
        proxy["username"] = os.getenv("ANNOTATION_PROXY_USERNAME")
    if os.getenv("ANNOTATION_PROXY_PASSWORD"):
        proxy["password"] = os.getenv("ANNOTATION_PROXY_PASSWORD")
    return proxy


class KymRenderer:
    """Playwright-backed page renderer producing GNN-ready nodes.

    If a ``proxy_pool`` is supplied (or PROXY_POOL_ENABLED in env), each render
    picks a proxy from the pool and, on failure, rotates to another — dropping
    dead ones — up to ``max_proxy_tries`` times. Otherwise the single static
    ``proxy`` (or ANNOTATION_PROXY_URL) is used.
    """

    def __init__(self, headless: bool = True, timeout_ms: int = 30000,
                 proxy: Optional[Dict[str, str]] = None,
                 max_nodes: int = 1500, proxy_pool: Any = None,
                 max_proxy_tries: int = 5):
        self.headless = headless
        self.timeout_ms = timeout_ms
        self.proxy = proxy if proxy is not None else _proxy_from_env()
        self.max_nodes = max_nodes
        self.pool = proxy_pool
        self.max_proxy_tries = max_proxy_tries

    def render(self, url: str) -> List[Dict[str, Any]]:
        if self.pool is None:
            return self._render_once(url, self.proxy)

        last_err: Optional[Exception] = None
        for _ in range(self.max_proxy_tries):
            proxy_url = self.pool.get()
            if proxy_url is None:
                break
            try:
                return self._render_once(url, {"server": proxy_url})
            except Exception as e:  # noqa: BLE001 - dead proxy, rotate
                last_err = e
                self.pool.mark_bad(proxy_url)
        raise RuntimeError(
            f"All proxy attempts failed for {url} (pool exhausted or every proxy "
            f"dead). Last error: {last_err}"
        )

    def _render_once(self, url: str, proxy: Optional[Dict[str, str]]) -> List[Dict[str, Any]]:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as e:  # pragma: no cover - only when dep missing
            raise RuntimeError(
                "playwright is not installed. `pip install -r requirements-model.txt` "
                "then `playwright install chromium`."
            ) from e

        launch_kwargs: Dict[str, Any] = {"headless": self.headless}
        if proxy:
            launch_kwargs["proxy"] = proxy

        with sync_playwright() as p:
            browser = p.chromium.launch(**launch_kwargs)
            try:
                page = browser.new_page()
                page.goto(url, wait_until="networkidle", timeout=self.timeout_ms)
                raw = page.evaluate(_COLLECT_JS)
            finally:
                browser.close()

        nodes: List[Dict[str, Any]] = []
        for i, item in enumerate(raw[: self.max_nodes]):
            nodes.append({
                "id": f"n{i}",
                "text": item.get("text", ""),
                "tag": item.get("tag", "div"),
                "bbox": [float(v) for v in item.get("bbox", [0, 0, 0, 0])],
            })
        if not nodes:
            raise RuntimeError(
                f"No nodes rendered from {url} — page may be blocked (IP ban?) "
                "or failed to load. Configure a proxy (ANNOTATION_PROXY_URL) or "
                "enable the proxy pool (PROXY_POOL_ENABLED=true)."
            )
        return nodes
