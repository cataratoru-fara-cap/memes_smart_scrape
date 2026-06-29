"""
Offline tests for kym_discover.py.

The pure functions (URL classification, sitemap XML parsing, taxonomy
inference, record building) run with stdlib only. The HTML-parsing tests are
skipped automatically when beautifulsoup4 is not installed.

    python -m unittest tests.test_discovery -v
"""

import importlib
import unittest

import kym_discover as kd

try:
    import bs4  # noqa: F401
    HAVE_BS4 = True
except ImportError:
    HAVE_BS4 = False


class TestIsEntryUrl(unittest.TestCase):
    def test_entries(self):
        for url in [
            "https://knowyourmeme.com/memes/distracted-boyfriend",
            "https://knowyourmeme.com/memes/events/some-event",
            "https://knowyourmeme.com/editorials/guides/foo",
            "https://knowyourmeme.com/cultures/rickrolling",
        ]:
            self.assertTrue(kd.is_entry_url(url), url)

    def test_non_entries(self):
        for url in [
            "https://knowyourmeme.com/memes/all",          # listing root
            "https://knowyourmeme.com/memes/deadpool",     # status listing
            "https://knowyourmeme.com/memes",              # too shallow
            "https://knowyourmeme.com/users/someone",      # not an entry root
            "https://knowyourmeme.com/memes/all/page/3",   # pagination
            "https://example.com/memes/foo",               # off-site
        ]:
            self.assertFalse(kd.is_entry_url(url), url)


class TestNamespaceAndRecord(unittest.TestCase):
    def test_namespace_of(self):
        self.assertEqual(kd.namespace_of("/memes/sites/reddit"), "memes/sites")
        self.assertEqual(kd.namespace_of("/memes/doge"), "memes")

    def test_make_record_confirmed_from_lastmod(self):
        r = kd.make_record("https://knowyourmeme.com/memes/a", "2026-01-01", None)
        self.assertTrue(r["Confirmed"])
        r2 = kd.make_record("https://knowyourmeme.com/memes/a", None, None)
        self.assertFalse(r2["Confirmed"])

    def test_non_confirmed_lastmod_forced_null(self):
        # lastmod only belongs to confirmed entries.
        r = kd.make_record("https://knowyourmeme.com/memes/a", "2026-01-01", None, confirmed=False)
        self.assertFalse(r["Confirmed"])
        self.assertIsNone(r["lastmod"])

    def test_make_record_explicit_confirmed_and_resume(self):
        existing = {"last_scraped": "2026-02-02T00:00:00Z"}
        r = kd.make_record("https://knowyourmeme.com/memes/a", None, existing, confirmed=True)
        self.assertTrue(r["Confirmed"])
        self.assertEqual(r["last_scraped"], "2026-02-02T00:00:00Z")


class TestParseSitemap(unittest.TestCase):
    def test_sitemapindex(self):
        xml = b"""<?xml version="1.0"?>
        <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
          <sitemap><loc>https://knowyourmeme.com/sitemaps/memes-1.xml</loc></sitemap>
          <sitemap><loc>https://knowyourmeme.com/sitemaps/memes-2.xml</loc></sitemap>
        </sitemapindex>"""
        children, entries = kd.parse_sitemap(xml)
        self.assertEqual(len(children), 2)
        self.assertEqual(entries, [])

    def test_urlset_with_https_namespace(self):
        xml = b"""<?xml version="1.0"?>
        <urlset xmlns="https://www.sitemaps.org/schemas/sitemap/0.9">
          <url><loc>https://knowyourmeme.com/memes/a</loc><lastmod>2026-06-01</lastmod></url>
          <url><loc>https://knowyourmeme.com/memes/b</loc></url>
        </urlset>"""
        children, entries = kd.parse_sitemap(xml)
        self.assertEqual(children, [])
        self.assertEqual(entries[0], ("https://knowyourmeme.com/memes/a", "2026-06-01"))
        self.assertEqual(entries[1], ("https://knowyourmeme.com/memes/b", None))


class TestInferTaxonomy(unittest.TestCase):
    def test_namespaces_and_listings(self):
        index = {
            "https://knowyourmeme.com/memes/sites/reddit": {},
            "https://knowyourmeme.com/memes/sites/twitter": {},
            "https://knowyourmeme.com/memes/doge": {},
            "https://knowyourmeme.com/memes/all": {},  # listing seen in corpus
        }
        namespaces, listings = kd.infer_taxonomy(index)
        prefixes = [p for p, _ in namespaces]
        self.assertIn("/memes/sites/", prefixes)
        # Longest-first ordering.
        self.assertGreaterEqual(len(prefixes[0]), len(prefixes[-1]))
        # Default status listings are always present.
        listing_paths = {p for p, _ in listings}
        for p in ("/memes/all", "/memes/deadpool", "/memes/submissions", "/memes/researching"):
            self.assertIn(p, listing_paths)


@unittest.skipUnless(HAVE_BS4, "beautifulsoup4 not installed")
class TestHtmlParsing(unittest.TestCase):
    def test_extract_entry_links_is_css_agnostic(self):
        html = """
        <html><body>
          <a class="weird-new-class" href="/memes/distracted-boyfriend">x</a>
          <a href="/memes/events/some-event">sub-namespace entry</a>
          <a href="/memes/all">listing (excluded)</a>
          <a href="/users/bob">user (excluded)</a>
          <a href="https://other.com/memes/x">offsite (excluded)</a>
        </body></html>
        """
        links = kd.extract_entry_links(html)
        self.assertIn("https://knowyourmeme.com/memes/distracted-boyfriend", links)
        self.assertIn("https://knowyourmeme.com/memes/events/some-event", links)
        self.assertNotIn("https://knowyourmeme.com/memes/all", links)
        self.assertTrue(all("users" not in u and "other.com" not in u for u in links))

    def test_find_next_page(self):
        html = '<a class="pagination__next" href="/memes/all/page/2">Next</a>'
        nxt = kd.find_next_page(html, "https://knowyourmeme.com/memes/all")
        self.assertEqual(nxt, "https://knowyourmeme.com/memes/all/page/2")

        html2 = '<div>no pagination here</div>'
        self.assertIsNone(kd.find_next_page(html2, "https://knowyourmeme.com/memes/all"))


if __name__ == "__main__":
    importlib.reload(kd)
    unittest.main(verbosity=2)
