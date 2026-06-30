"""
Offline tests for the meme annotation pipeline.

These run without scrapegraphai, pydantic, torch, an API key or network access:
    python -m unittest tests.test_meme_annotation -v
"""

import json
import tempfile
import unittest
from pathlib import Path

from src.annotation import meme_schema, url_store
from src.annotation.config import AnnotationConfig
from src.annotation.annotator import MockAnnotator
import annotate_memes


class TestGraphConfig(unittest.TestCase):
    """ScrapeGraph-AI config wiring, incl. the self-hosted gateway path."""

    def test_model_prefixed_with_provider(self):
        cfg = AnnotationConfig()
        cfg.provider, cfg.model = "openai", "llama3.1:8b"
        self.assertEqual(cfg.graph_config()["llm"]["model"], "openai/llama3.1:8b")

    def test_base_url_and_key_passed_through(self):
        cfg = AnnotationConfig()
        cfg.provider, cfg.api_key = "openai", "lab-token"
        cfg.base_url = "https://lab-host/openai"
        llm = cfg.graph_config()["llm"]
        self.assertEqual(llm["base_url"], "https://lab-host/openai")
        self.assertEqual(llm["api_key"], "lab-token")

    def test_structured_output_default_true(self):
        self.assertTrue(AnnotationConfig().structured_output)


class TestTemplateDetection(unittest.TestCase):
    def test_known_sections(self):
        cases = {
            "https://knowyourmeme.com/memes/doge": "meme",
            "https://knowyourmeme.com/editorials/guides/foo": "editorial",
            "https://knowyourmeme.com/cultures/rickrolling": "culture",
            "https://knowyourmeme.com/people/elon-musk": "person",
        }
        for url, expected in cases.items():
            self.assertEqual(meme_schema.detect_template_type(url), expected)

    def test_unknown_section(self):
        self.assertEqual(meme_schema.detect_template_type("https://knowyourmeme.com/"), "unknown")

    def test_spec_family(self):
        self.assertEqual(meme_schema.spec_family("editorial"), "editorial")
        self.assertEqual(meme_schema.spec_family("meme"), "entry")


class TestNormalize(unittest.TestCase):
    def test_empty_record_has_all_fields(self):
        rec = meme_schema.empty_record("meme")
        names = {n for n, _, _ in meme_schema.ENTRY_FIELDS}
        self.assertEqual(set(rec.keys()), names)

    def test_coercion(self):
        raw = {
            "title": 123,                       # -> str
            "year": "2017 (approx)",            # -> int 2017
            "nsfw": "yes",                      # -> True
            "tags": "a, b, c",                  # comma string -> list
            "entry_type": ["Image Macro"],
            "notable_examples": [{"caption": "x"}, "bad"],  # drops non-dict
            "unexpected": "dropped",            # extra key removed
        }
        out = meme_schema.normalize(raw, "meme")
        self.assertEqual(out["title"], "123")
        self.assertEqual(out["year"], 2017)
        self.assertTrue(out["nsfw"])
        self.assertEqual(out["tags"], ["a", "b", "c"])
        self.assertEqual(out["notable_examples"], [{"caption": "x"}])
        self.assertNotIn("unexpected", out)

    def test_normalize_non_dict(self):
        out = meme_schema.normalize("not a dict", "meme")
        self.assertEqual(out, meme_schema.empty_record("meme"))


class TestPromptAndSchema(unittest.TestCase):
    def test_prompt_mentions_fields(self):
        prompt = meme_schema.build_prompt("meme")
        self.assertIn("origin_description", prompt)
        self.assertIn("JSON", prompt)

    def test_editorial_prompt_differs(self):
        self.assertIn("author", meme_schema.build_prompt("editorial"))
        self.assertNotIn("origin_description", meme_schema.build_prompt("editorial"))


class TestUrlStore(unittest.TestCase):
    def _records(self):
        return [
            {"url": "https://knowyourmeme.com/memes/a", "Confirmed": True, "last_scraped": None},
            {"url": "https://knowyourmeme.com/memes/b", "Confirmed": False, "last_scraped": None},
            {"url": "https://knowyourmeme.com/memes/c", "Confirmed": True, "last_scraped": "2026-01-01T00:00:00Z"},
            {"url": "https://knowyourmeme.com/memes/d", "Confirmed": True, "last_scraped": None},
        ]

    def test_iter_pending_only_confirmed_and_resume(self):
        recs = self._records()
        done = {"https://knowyourmeme.com/memes/d"}
        pending = list(url_store.iter_pending(recs, done, only_confirmed=True, force=False))
        urls = [r["url"] for r in pending]
        # b excluded (not confirmed), c excluded (already scraped), d excluded (done)
        self.assertEqual(urls, ["https://knowyourmeme.com/memes/a"])

    def test_iter_pending_force(self):
        recs = self._records()
        done = {"https://knowyourmeme.com/memes/d"}
        pending = list(url_store.iter_pending(recs, done, only_confirmed=False, force=True))
        self.assertEqual(len(pending), 4)

    def test_load_array_and_dir(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "a.json").write_text(json.dumps(self._records()[:2]), encoding="utf-8")
            (d / "b.jsonl").write_text(
                "\n".join(json.dumps(r) for r in self._records()[2:]), encoding="utf-8")
            recs = url_store.load_url_records(str(d))
            self.assertEqual(len(recs), 4)

    def test_writer_resume(self):
        with tempfile.TemporaryDirectory() as d:
            out = str(Path(d) / "annotations.jsonl")
            doc = url_store.build_document(
                {"url": "https://knowyourmeme.com/memes/a", "Confirmed": True},
                "meme", meme_schema.empty_record("meme"),
                model="m", provider="p",
            )
            with url_store.AnnotationWriter(out) as w:
                w.write(doc)
            self.assertEqual(
                url_store.AnnotationWriter(out).existing_urls(),
                {"https://knowyourmeme.com/memes/a"},
            )

    def test_build_document_shape(self):
        doc = url_store.build_document(
            {"url": "https://knowyourmeme.com/memes/a", "Confirmed": True,
             "lastmod": "2026-06-23T12:16:57-04:00"},
            "meme", meme_schema.empty_record("meme"), model="gpt-4o-mini", provider="openai",
        )
        self.assertEqual(doc["_id"], url_store.url_id("https://knowyourmeme.com/memes/a"))
        self.assertEqual(doc["page_template_type"], "meme")
        self.assertTrue(doc["last_scraped"])
        self.assertTrue(doc["_annotation"]["ok"])
        self.assertIn("meme", doc)


class TestMockAnnotator(unittest.TestCase):
    def test_meme(self):
        out = MockAnnotator().annotate("https://knowyourmeme.com/memes/distracted-boyfriend", "meme")
        self.assertEqual(out["title"], "Distracted Boyfriend")
        self.assertEqual(out["entry_type"], ["Image Macro"])

    def test_editorial(self):
        out = MockAnnotator().annotate("https://knowyourmeme.com/editorials/guides/foo-bar", "editorial")
        self.assertEqual(out["title"], "Foo Bar")
        self.assertIn("summary", out)


class TestEndToEndMock(unittest.TestCase):
    def test_pipeline_runs_and_resumes(self):
        with tempfile.TemporaryDirectory() as d:
            inp = Path(d) / "urls.json"
            out = Path(d) / "annotations.jsonl"
            inp.write_text(json.dumps([
                {"url": "https://knowyourmeme.com/memes/a", "Confirmed": True, "last_scraped": None},
                {"url": "https://knowyourmeme.com/memes/b", "Confirmed": True, "last_scraped": None},
            ]), encoding="utf-8")

            cfg = AnnotationConfig()
            cfg.input_path, cfg.output_path, cfg.concurrency = str(inp), str(out), 2

            rc = annotate_memes.run(cfg, mock=True, limit=0, force=False)
            self.assertEqual(rc, 0)
            lines = out.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)

            # Second run should annotate nothing (resume).
            rc2 = annotate_memes.run(cfg, mock=True, limit=0, force=False)
            self.assertEqual(rc2, 0)
            lines2 = out.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines2), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
