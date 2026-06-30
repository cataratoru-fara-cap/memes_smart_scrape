"""
Offline tests for the meme GNN+ILP student model.

Exercises the pure paths — value extraction, node alignment (singletons +
multi-label), the ILP solver's greedy fallback (ortools absent), the section
extractor, and the inference pipeline with uniform GNN scores (torch absent).
torch/ortools/playwright code paths are run by the user once installed.

    python -m unittest tests.test_meme_student -v
"""

import unittest

from src.learning import meme_labels, meme_encoding, meme_sections
from src.learning.meme_config import MEME_CLASSES
from src.learning.meme_pipeline import MemeExtractionPipeline
from src.reasoning.meme_solver import MemeConstraintSolver, year_ok, status_ok

N_CLASSES = len(MEME_CLASSES)

MEME_PAYLOAD = {
    "title": "Doge",
    "entry_type": ["Image Macro"],
    "status": "Confirmed",
    "origin": "Tumblr",
    "year": 2013,
    "parent_meme": "Shiba Inu",
    "aliases": ["Shibe", "Doge Dog"],
    "region": ["Japan"],
    "tags": ["dog", "shiba", "reaction"],
    "related_memes": ["Cheems"],
    "about": "A long description that is not node-localizable.",
}

PAGE_NODES = [
    {"id": "n0", "text": "Doge", "tag": "h1", "bbox": [100, 50, 200, 40]},
    {"id": "n1", "text": "Confirmed", "tag": "a", "bbox": [100, 200, 80, 20]},
    {"id": "n2", "text": "2013", "tag": "span", "bbox": [100, 230, 40, 20]},
    {"id": "n3", "text": "Image Macro", "tag": "a", "bbox": [100, 260, 110, 20]},
    {"id": "n4", "text": "Tumblr", "tag": "a", "bbox": [100, 290, 80, 20]},
    {"id": "n5", "text": "A long paragraph about the meme " * 5, "tag": "p", "bbox": [100, 800, 600, 200]},
    {"id": "n6", "text": "Shiba Inu", "tag": "a", "bbox": [100, 320, 90, 20]},
    {"id": "n7", "text": "Japan", "tag": "a", "bbox": [100, 350, 60, 20]},
    {"id": "n8", "text": "dog", "tag": "a", "bbox": [100, 380, 40, 20]},
    {"id": "n9", "text": "shiba", "tag": "a", "bbox": [150, 380, 50, 20]},
    {"id": "n10", "text": "reaction", "tag": "a", "bbox": [210, 380, 70, 20]},
    {"id": "n11", "text": "Cheems", "tag": "a", "bbox": [100, 420, 70, 20]},
    {"id": "n12", "text": "Shibe", "tag": "a", "bbox": [100, 450, 60, 20]},
]


class TestFieldValues(unittest.TestCase):
    def test_singleton_values(self):
        vals = meme_labels.localizable_field_values(MEME_PAYLOAD)
        self.assertEqual(vals, {"title": "Doge", "type": "Image Macro",
                                "status": "Confirmed", "origin": "Tumblr",
                                "year": "2013", "parent_meme": "Shiba Inu"})

    def test_multi_values(self):
        vals = meme_labels.multi_field_values(MEME_PAYLOAD)
        self.assertEqual(vals["tag"], ["dog", "shiba", "reaction"])
        self.assertEqual(vals["alias"], ["Shibe", "Doge Dog"])
        self.assertEqual(vals["region"], ["Japan"])
        self.assertEqual(vals["related"], ["Cheems"])

    def test_skips_empty_and_zero_year(self):
        vals = meme_labels.localizable_field_values(
            {"title": "X", "entry_type": [], "status": "", "origin": "", "year": 0})
        self.assertEqual(vals, {"title": "X"})


class TestAlignment(unittest.TestCase):
    def setUp(self):
        self.labeled = meme_labels.label_page(PAGE_NODES, MEME_PAYLOAD)
        self.by_id = {n["id"]: n["label"] for n in self.labeled}

    def test_singletons(self):
        self.assertEqual(self.by_id["n0"], "title")
        self.assertEqual(self.by_id["n1"], "status")
        self.assertEqual(self.by_id["n2"], "year")
        self.assertEqual(self.by_id["n3"], "type")
        self.assertEqual(self.by_id["n4"], "origin")
        self.assertEqual(self.by_id["n6"], "parent_meme")
        self.assertEqual(self.by_id["n5"], "other")

    def test_multi_label(self):
        self.assertEqual(self.by_id["n7"], "region")
        self.assertEqual({self.by_id[i] for i in ("n8", "n9", "n10")}, {"tag"})
        self.assertEqual(self.by_id["n11"], "related")
        self.assertEqual(self.by_id["n12"], "alias")

    def test_year_match_requires_digits(self):
        self.assertEqual(meme_labels.match_score("born in 2013", "2013", "year"), 1.0)
        self.assertEqual(meme_labels.match_score("no year here", "2013", "year"), 0.0)


class TestEncoding(unittest.TestCase):
    def test_label_index(self):
        self.assertEqual(meme_encoding.label_index("title"), 0)
        self.assertEqual(meme_encoding.label_index("tag"), MEME_CLASSES.index("tag"))
        self.assertEqual(meme_encoding.label_index("other"), N_CLASSES - 1)
        self.assertEqual(meme_encoding.label_index("bogus"), N_CLASSES - 1)

    def test_input_dim(self):
        self.assertEqual(meme_encoding.INPUT_DIM, 146)


class TestSolver(unittest.TestCase):
    def _uniform(self, n):
        return [[1.0 / N_CLASSES] * N_CLASSES for _ in range(n)]

    def test_format_forces_valid_year_node(self):
        solver = MemeConstraintSolver()
        idx = solver.cls_to_idx
        probs = self._uniform(len(PAGE_NODES))
        probs[0][idx["year"]] = 0.99   # push 'year' high on a NON-year node
        probs[2][idx["year"]] = 0.4
        record = solver.solve(PAGE_NODES, probs)
        self.assertEqual(record["year"]["text"], "2013")

    def test_status_format(self):
        solver = MemeConstraintSolver()
        idx = solver.cls_to_idx
        probs = self._uniform(len(PAGE_NODES))
        probs[5][idx["status"]] = 0.99
        probs[1][idx["status"]] = 0.5
        record = solver.solve(PAGE_NODES, probs)
        self.assertEqual(record["status"]["text"], "Confirmed")

    def test_multi_aggregates_into_list(self):
        solver = MemeConstraintSolver()
        idx = solver.cls_to_idx
        probs = self._uniform(len(PAGE_NODES))
        # Elevate 'tag' on three nodes -> all three should be collected.
        for i in (8, 9, 10):
            probs[i][idx["tag"]] = 0.9
        record = solver.solve(PAGE_NODES, probs)
        self.assertIsInstance(record["tags"], list)
        self.assertEqual({t["text"] for t in record["tags"]}, {"dog", "shiba", "reaction"})

    def test_uniqueness_and_other_excluded(self):
        solver = MemeConstraintSolver()
        record = solver.solve(PAGE_NODES, self._uniform(len(PAGE_NODES)))
        self.assertNotIn("other", record)
        self.assertIn("_solver", record)

    def test_predicates(self):
        self.assertTrue(year_ok("est. 2013"))
        self.assertFalse(year_ok("no digits"))
        self.assertTrue(status_ok("Confirmed"))
        self.assertTrue(status_ok("deadpool"))
        self.assertFalse(status_ok("random text"))


class TestSections(unittest.TestCase):
    def test_extracts_sections_between_headings(self):
        nodes = [
            {"id": "h1", "text": "About", "tag": "h2", "bbox": [0, 100, 100, 30]},
            {"id": "p1", "text": "Doge is a slang term and an internet meme.", "tag": "p", "bbox": [0, 140, 600, 40]},
            {"id": "h2", "text": "Origin", "tag": "h2", "bbox": [0, 200, 100, 30]},
            {"id": "p2", "text": "The first known use appeared on a blog in 2010.", "tag": "p", "bbox": [0, 240, 600, 40]},
            {"id": "h3", "text": "Spread", "tag": "h2", "bbox": [0, 300, 100, 30]},
            {"id": "p3", "text": "It spread widely across Reddit and Tumblr through 2013.", "tag": "p", "bbox": [0, 340, 600, 40]},
            {"id": "h4", "text": "External References", "tag": "h2", "bbox": [0, 400, 200, 30]},
            {"id": "p4", "text": "Wikipedia article and various news outlets.", "tag": "p", "bbox": [0, 440, 600, 40]},
        ]
        sections = meme_sections.extract_sections(nodes)
        self.assertIn("Doge is a slang term", sections["about"])
        self.assertIn("first known use", sections["origin_description"])
        self.assertIn("spread widely", sections["spread_description"])
        # Content after a non-target heading must not leak into spread.
        self.assertNotIn("Wikipedia", sections["spread_description"])

    def test_no_sections_when_no_headings(self):
        self.assertEqual(meme_sections.extract_sections(PAGE_NODES), {})


class TestPipeline(unittest.TestCase):
    def test_run_with_nodes_no_model(self):
        pipe = MemeExtractionPipeline(model_path="does_not_exist.pt")
        record = pipe.run(nodes=PAGE_NODES)
        self.assertEqual(record["title"]["text"], "Doge")
        self.assertEqual(record["year"]["text"], "2013")
        self.assertEqual(record["status"]["text"], "Confirmed")
        self.assertEqual(record["type"]["text"], "Image Macro")
        self.assertFalse(record["_meta"]["model_trained"])
        self.assertEqual(record["_meta"]["num_nodes"], len(PAGE_NODES))
        self.assertIn("stability_score", record["_meta"])

    def test_requires_nodes_or_renderer(self):
        pipe = MemeExtractionPipeline(model_path="does_not_exist.pt")
        with self.assertRaises(ValueError):
            pipe.run(url="https://knowyourmeme.com/memes/doge")


if __name__ == "__main__":
    unittest.main(verbosity=2)
