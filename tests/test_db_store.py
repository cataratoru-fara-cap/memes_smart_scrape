"""
Offline tests for the shared storage layer (src/db) and the discovery ->
MongoDB -> annotation flow, using the dependency-free InMemoryStore.

    python -m unittest tests.test_db_store -v
"""

import unittest

from src.db.store import InMemoryStore, merge_discovery, url_doc_id


def _rec(url, confirmed=False, lastmod=None, last_scraped=None):
    return {
        "url": url, "namespace": "memes", "Confirmed": confirmed,
        "lastmod": lastmod, "page_template_type": None, "last_scraped": last_scraped,
    }


class TestMergeDiscovery(unittest.TestCase):
    def test_confirmed_is_monotonic(self):
        old = _rec("u", confirmed=True, lastmod="2026-01-01")
        new = _rec("u", confirmed=False, lastmod=None)
        merged = merge_discovery(old, new)
        self.assertTrue(merged["Confirmed"])           # never downgraded
        self.assertEqual(merged["lastmod"], "2026-01-01")  # null new keeps old

    def test_last_scraped_preserved(self):
        old = _rec("u", last_scraped="2026-02-02T00:00:00Z")
        merged = merge_discovery(old, _rec("u"))
        self.assertEqual(merged["last_scraped"], "2026-02-02T00:00:00Z")

    def test_insert_sets_id(self):
        merged = merge_discovery(None, _rec("https://knowyourmeme.com/memes/a"))
        self.assertEqual(merged["_id"], url_doc_id("https://knowyourmeme.com/memes/a"))


class TestInMemoryStore(unittest.TestCase):
    def test_upsert_added_then_updated(self):
        s = InMemoryStore()
        self.assertEqual(s.upsert_urls([_rec("a"), _rec("b")]), {"added": 2, "updated": 0})
        self.assertEqual(s.upsert_urls([_rec("a")]), {"added": 0, "updated": 1})
        self.assertEqual(s.count_urls(), 2)

    def test_iter_pending_filters(self):
        s = InMemoryStore()
        s.upsert_urls([
            _rec("a", confirmed=True),
            _rec("b", confirmed=False),
            _rec("c", confirmed=True, last_scraped="2026-01-01T00:00:00Z"),
        ])
        pending = [r["url"] for r in s.iter_pending(only_confirmed=True, force=False)]
        self.assertEqual(pending, ["a"])  # b not confirmed, c already scraped

    def test_save_annotation_marks_url(self):
        s = InMemoryStore()
        s.upsert_urls([_rec("https://knowyourmeme.com/memes/a", confirmed=True)])
        doc = {
            "url": "https://knowyourmeme.com/memes/a",
            "page_template_type": "meme",
            "last_scraped": "2026-03-03T00:00:00Z",
            "meme": {},
        }
        s.save_annotation(doc)
        self.assertEqual(len(s.annotations), 1)
        self.assertEqual(s.annotated_urls(), {"https://knowyourmeme.com/memes/a"})
        # Now excluded from pending (resume).
        self.assertEqual(list(s.iter_pending(only_confirmed=True)), [])


class TestDiscoveryToAnnotationFlow(unittest.TestCase):
    def test_end_to_end_with_inmemory_mongo(self):
        import src.db.mongo as mongo_mod
        import annotate_memes
        from src.annotation.config import AnnotationConfig

        store = InMemoryStore()
        store.upsert_urls([
            _rec("https://knowyourmeme.com/memes/doge", confirmed=True),
            _rec("https://knowyourmeme.com/memes/sub", confirmed=False),  # skipped
            _rec("https://knowyourmeme.com/editorials/guides/x", confirmed=True),
        ])

        original = mongo_mod.get_store
        mongo_mod.get_store = lambda *a, **k: store
        try:
            cfg = AnnotationConfig()
            cfg.source = "mongo"
            cfg.concurrency = 2
            rc = annotate_memes.run(cfg, mock=True, limit=0, force=False)
            self.assertEqual(rc, 0)
            # Only the 2 confirmed urls annotated.
            self.assertEqual(len(store.annotations), 2)
            self.assertEqual(len(store.annotated_urls()), 2)

            # Second run resumes -> nothing to do.
            rc2 = annotate_memes.run(cfg, mock=True, limit=0, force=False)
            self.assertEqual(rc2, 0)
            self.assertEqual(len(store.annotations), 2)
        finally:
            mongo_mod.get_store = original


if __name__ == "__main__":
    unittest.main(verbosity=2)
