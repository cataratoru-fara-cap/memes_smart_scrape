"""
Offline tests for the meme GNN+ILP student model.

Exercises the pure paths — value extraction, node alignment, the ILP solver's
greedy fallback (ortools absent), and the inference pipeline with uniform GNN
scores (torch absent). torch/ortools/playwright code paths are run by the user
once installed.

    python -m unittest tests.test_meme_student -v
"""

import unittest

from src.learning import meme_labels, meme_encoding
from src.learning.meme_pipeline import MemeExtractionPipeline
from src.reasoning.meme_solver import MemeConstraintSolver, year_ok, status_ok


MEME_PAYLOAD = {
    "title": "Doge",
    "entry_type": ["Image Macro"],
    "status": "Confirmed",
    "origin": "Tumblr",
    "year": 2013,
    "about": "A long description that is not node-localizable.",
}

PAGE_NODES = [
    {"id": "n0", "text": "Doge", "tag": "h1", "bbox": [100, 50, 200, 40]},
    {"id": "n1", "text": "Confirmed", "tag": "a", "bbox": [100, 200, 80, 20]},
    {"id": "n2", "text": "2013", "tag": "span", "bbox": [100, 230, 40, 20]},
    {"id": "n3", "text": "Image Macro", "tag": "a", "bbox": [100, 260, 110, 20]},
    {"id": "n4", "text": "Tumblr", "tag": "a", "bbox": [100, 290, 80, 20]},
    {"id": "n5", "text": "A long paragraph about the meme " * 5, "tag": "p", "bbox": [100, 400, 600, 200]},
]


class TestFieldValues(unittest.TestCase):
    def test_localizable_values(self):
        vals = meme_labels.localizable_field_values(MEME_PAYLOAD)
        self.assertEqual(vals, {"title": "Doge", "type": "Image Macro",
                                "status": "Confirmed", "origin": "Tumblr", "year": "2013"})

    def test_skips_empty_and_zero_year(self):
        vals = meme_labels.localizable_field_values(
            {"title": "X", "entry_type": [], "status": "", "origin": "", "year": 0})
        self.assertEqual(vals, {"title": "X"})


class TestAlignment(unittest.TestCase):
    def test_alignment_labels_correct_nodes(self):
        labeled = meme_labels.label_page(PAGE_NODES, MEME_PAYLOAD)
        by_id = {n["id"]: n["label"] for n in labeled}
        self.assertEqual(by_id["n0"], "title")
        self.assertEqual(by_id["n1"], "status")
        self.assertEqual(by_id["n2"], "year")
        self.assertEqual(by_id["n3"], "type")
        self.assertEqual(by_id["n4"], "origin")
        self.assertEqual(by_id["n5"], "other")

    def test_one_node_per_field(self):
        labeled = meme_labels.label_page(PAGE_NODES, MEME_PAYLOAD)
        labels = [n["label"] for n in labeled if n["label"] != "other"]
        self.assertEqual(len(labels), len(set(labels)))  # no field used twice

    def test_year_match_requires_digits(self):
        self.assertEqual(meme_labels.match_score("born in 2013", "2013", "year"), 1.0)
        self.assertEqual(meme_labels.match_score("no year here", "2013", "year"), 0.0)


class TestEncoding(unittest.TestCase):
    def test_label_index(self):
        self.assertEqual(meme_encoding.label_index("title"), 0)
        self.assertEqual(meme_encoding.label_index("year"), 4)
        self.assertEqual(meme_encoding.label_index("other"), 5)
        self.assertEqual(meme_encoding.label_index("none"), 5)
        self.assertEqual(meme_encoding.label_index("bogus"), 5)

    def test_input_dim(self):
        self.assertEqual(meme_encoding.INPUT_DIM, 146)


class TestSolver(unittest.TestCase):
    def _uniform(self, n, c=6):
        return [[1.0 / c] * c for _ in range(n)]

    def test_format_forces_valid_year_node(self):
        solver = MemeConstraintSolver()
        idx = solver.cls_to_idx
        probs = self._uniform(len(PAGE_NODES))
        # Push 'year' score highest on a NON-year node (n0 'Doge').
        probs[0][idx["year"]] = 0.99
        # And give the real year node a modest score.
        probs[2][idx["year"]] = 0.4
        record = solver.solve(PAGE_NODES, probs)
        self.assertEqual(record["year"]["text"], "2013")  # format constraint wins

    def test_status_format(self):
        solver = MemeConstraintSolver()
        idx = solver.cls_to_idx
        probs = self._uniform(len(PAGE_NODES))
        probs[5][idx["status"]] = 0.99  # the long paragraph, not a status word
        probs[1][idx["status"]] = 0.5
        record = solver.solve(PAGE_NODES, probs)
        self.assertEqual(record["status"]["text"], "Confirmed")

    def test_uniqueness_and_other_excluded(self):
        solver = MemeConstraintSolver()
        record = solver.solve(PAGE_NODES, self._uniform(len(PAGE_NODES)))
        self.assertNotIn("other", record)
        self.assertIn("_solver", record)
        # Each present field maps to exactly one node.
        for field in ("title", "type", "status", "origin", "year"):
            if field in record:
                self.assertIn("node_id", record[field])

    def test_predicates(self):
        self.assertTrue(year_ok("est. 2013"))
        self.assertFalse(year_ok("no digits"))
        self.assertTrue(status_ok("Confirmed"))
        self.assertTrue(status_ok("deadpool"))
        self.assertFalse(status_ok("random text"))


class TestPipeline(unittest.TestCase):
    def test_run_with_nodes_no_model(self):
        # No model file, no torch -> uniform scores + priors + greedy solver.
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
            pipe.run(url="https://knowyourmeme.com/memes/doge")  # no renderer


if __name__ == "__main__":
    unittest.main(verbosity=2)
