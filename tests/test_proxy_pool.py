"""
Offline tests for the rotating proxy pool and renderer failover.

No network: list parsing is fed raw strings, and renderer rotation is tested by
stubbing the Playwright call.

    python -m unittest tests.test_proxy_pool -v
"""

import unittest

from src.net.proxy_pool import ProxyPool, proxy_pool_from_env
from src.learning.kym_render import KymRenderer

TXT = """
http://1.1.1.1:80
https://2.2.2.2:8443
socks5://3.3.3.3:1080
not-a-proxy
4.4.4.4:3128
""".strip()

JSON = """
[{"proxy":"http://5.5.5.5:80","protocol":"http"},
 {"proxy":"socks4://6.6.6.6:1080","protocol":"socks4"}]
"""


class TestParse(unittest.TestCase):
    def test_txt_filters_by_protocol(self):
        http_only = ProxyPool.parse(TXT, ["http"])
        # bare "4.4.4.4:3128" defaults to http; https/socks5 excluded.
        self.assertEqual(set(http_only), {"http://1.1.1.1:80", "http://4.4.4.4:3128"})

    def test_txt_multi_protocol(self):
        got = ProxyPool.parse(TXT, ["http", "https", "socks5"])
        self.assertIn("https://2.2.2.2:8443", got)
        self.assertIn("socks5://3.3.3.3:1080", got)

    def test_json(self):
        got = ProxyPool.parse(JSON, ["http"], is_json=True)
        self.assertEqual(got, ["http://5.5.5.5:80"])


class TestRotation(unittest.TestCase):
    def _pool(self):
        p = ProxyPool()
        p.load("http://1.1.1.1:80\nhttp://2.2.2.2:80\nhttp://3.3.3.3:80")
        return p

    def test_round_robin_and_mark_bad(self):
        p = self._pool()
        self.assertEqual(p.size(), 3)
        first = p.get()
        p.mark_bad(first)
        self.assertEqual(p.size(), 2)
        # first is never handed out again
        seen = {p.get() for _ in range(6)}
        self.assertNotIn(first, seen)

    def test_exhaustion_returns_none(self):
        p = self._pool()
        for _ in range(3):
            p.mark_bad(p.get())
        self.assertEqual(p.size(), 0)
        self.assertIsNone(p.get())

    def test_adapters(self):
        self.assertEqual(ProxyPool.to_playwright("http://x:1"), {"server": "http://x:1"})
        self.assertEqual(ProxyPool.to_requests("http://x:1"),
                         {"http": "http://x:1", "https": "http://x:1"})


class TestEnvFactory(unittest.TestCase):
    def test_disabled_by_default(self):
        self.assertIsNone(proxy_pool_from_env())  # PROXY_POOL_ENABLED unset


class TestRendererFailover(unittest.TestCase):
    def test_rotates_past_dead_proxy(self):
        pool = ProxyPool()
        pool.load("http://1.1.1.1:80\nhttp://2.2.2.2:80")
        r = KymRenderer(proxy_pool=pool)

        calls = []

        def fake_render_once(url, proxy):
            calls.append(proxy["server"])
            if len(calls) == 1:
                raise RuntimeError("dead proxy")
            return [{"id": "n0", "text": "ok", "tag": "p", "bbox": [0, 0, 1, 1]}]

        r._render_once = fake_render_once
        nodes = r.render("https://knowyourmeme.com/memes/x")
        self.assertEqual(nodes[0]["id"], "n0")
        self.assertEqual(len(calls), 2)
        self.assertIn(calls[0], pool._bad)          # dead one dropped
        self.assertNotIn(calls[1], pool._bad)

    def test_all_dead_raises(self):
        pool = ProxyPool()
        pool.load("http://1.1.1.1:80")
        r = KymRenderer(proxy_pool=pool)
        r._render_once = lambda url, proxy: (_ for _ in ()).throw(RuntimeError("dead"))
        with self.assertRaises(RuntimeError):
            r.render("https://knowyourmeme.com/memes/x")


if __name__ == "__main__":
    unittest.main(verbosity=2)
