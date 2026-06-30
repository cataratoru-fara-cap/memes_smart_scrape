"""
Constraint solver for the meme student model.

Same neuro-symbolic idea as ``solver_fixed.ConstraintSolver`` (maximise the
GNN's per-node scores subject to hard constraints), generalised to the meme
field set and made dependency-light:

  * Integrity     — each node gets exactly one class.
  * Uniqueness    — at most one node per singleton field (title/type/status/
                    origin/year).
  * Format        — year => text contains a 4-digit year; status => text is a
                    known KYM status word.
  * Title zone    — (optional) title must sit above ``title_zone_max_px``.

OR-Tools (SCIP) is imported lazily; if it is unavailable, or the model is
infeasible after relaxation, a pure-Python greedy fallback is used. The greedy
path needs no third-party packages, so the solver is fully testable offline.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

from src.learning.meme_config import (
    CLASS_TO_FIELD, MEME_CLASSES, MULTI_FIELDS, SINGLETON_FIELDS, STATUS_VALUES,
)

_YEAR_RE = re.compile(r"(?:18|19|20)\d{2}")
_WS = re.compile(r"\s+")


def _norm(s: Any) -> str:
    return _WS.sub(" ", str(s).strip().lower())


def year_ok(text: str) -> bool:
    return bool(_YEAR_RE.search(str(text)))


def status_ok(text: str) -> bool:
    t = _norm(text)
    return any(sv == t or sv in t.split() for sv in STATUS_VALUES)


# Per-field hard format predicate (None = no format constraint).
FORMAT_PREDICATES = {
    "year": year_ok,
    "status": status_ok,
}


class MemeConstraintSolver:
    """Spec-driven ILP over meme fields with a greedy fallback."""

    def __init__(self, classes: Sequence[str] = MEME_CLASSES,
                 title_zone_max_px: float | None = None):
        self.classes = list(classes)
        self.cls_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.fields = [f for f in SINGLETON_FIELDS if f in self.cls_to_idx]  # uniqueness
        self.multi_fields = [f for f in MULTI_FIELDS if f in self.cls_to_idx]
        self.title_zone_max_px = title_zone_max_px

    # ------------------------------------------------------------------
    def solve(self, nodes: List[Dict[str, Any]], probs: Sequence[Sequence[float]],
              page_height: float = 2000.0) -> Dict[str, Any]:
        """
        Select the best field->node assignment. ``probs`` is an N x C matrix
        (list-of-lists or numpy) of per-node class scores.
        """
        try:
            from ortools.linear_solver import pywraplp  # noqa: F401
        except ImportError:
            return self._greedy(nodes, probs, reason="ortools unavailable")

        for relax_zone in (False, True):
            result = self._build_and_solve(nodes, probs, page_height, relax_zone)
            if result is not None:
                return result
        return self._greedy(nodes, probs, reason="ILP infeasible after relaxation")

    # ------------------------------------------------------------------
    def _build_and_solve(self, nodes, probs, page_height, relax_zone):
        from ortools.linear_solver import pywraplp

        solver = pywraplp.Solver.CreateSolver("SCIP")
        if not solver:
            return self._greedy(nodes, probs, reason="SCIP unavailable")

        n, c = len(nodes), len(self.classes)
        x = {(i, j): solver.IntVar(0, 1, f"x_{i}_{j}") for i in range(n) for j in range(c)}
        trace: List[Dict[str, Any]] = []
        relaxed = ["title_zone"] if relax_zone else []

        # Integrity: one class per node.
        for i in range(n):
            solver.Add(solver.Sum([x[i, j] for j in range(c)]) == 1)
        trace.append({"id": "Integrity", "detail": "Σ_j x[i,j] = 1"})

        # Uniqueness: ≤1 node per singleton field.
        for field in self.fields:
            fi = self.cls_to_idx[field]
            solver.Add(solver.Sum([x[i, fi] for i in range(n)]) <= 1)
            trace.append({"id": "Uniqueness", "field": field, "detail": f"Σ_i x[i,{field}] ≤ 1"})

        # Per-node format + zone constraints.
        for i, node in enumerate(nodes):
            text = str(node.get("text", ""))
            for field, pred in FORMAT_PREDICATES.items():
                fi = self.cls_to_idx.get(field)
                if fi is not None and not pred(text):
                    solver.Add(x[i, fi] == 0)

            if not relax_zone and self.title_zone_max_px is not None and "title" in self.cls_to_idx:
                y = self._y(node)
                if y > self.title_zone_max_px:
                    solver.Add(x[i, self.cls_to_idx["title"]] == 0)
        trace.append({"id": "Format", "detail": "year⇒4-digit, status⇒KYM status word"})

        # Objective: maximise total selected score.
        obj = solver.Objective()
        for i in range(n):
            for j in range(c):
                obj.SetCoefficient(x[i, j], float(probs[i][j]))
        obj.SetMaximization()

        if solver.Solve() != pywraplp.Solver.OPTIMAL:
            return None

        return self._assemble(nodes, probs, trace, relaxed, obj.Value(),
                              assignments=self._read_solution(x, n, c))

    def _read_solution(self, x, n, c):
        """Return [(class_name, node_idx)] for selected non-'other' variables."""
        assignments = []
        for i in range(n):
            for j in range(c):
                if x[i, j].solution_value() > 0.5 and self.classes[j] != "other":
                    assignments.append((self.classes[j], i))
        return assignments

    # ------------------------------------------------------------------
    def _greedy(self, nodes, probs, reason="fallback") -> Dict[str, Any]:
        """
        Pure-Python fallback. Singletons: best valid node per field (uniqueness +
        format). Multi: every still-free node whose argmax class is that multi
        field (so a multi class can claim several nodes).
        """
        best: Dict[str, tuple] = {}  # field -> (score, idx)
        for field in self.fields:
            fi = self.cls_to_idx[field]
            pred = FORMAT_PREDICATES.get(field)
            for i, node in enumerate(nodes):
                if pred is not None and not pred(str(node.get("text", ""))):
                    continue
                if self.title_zone_max_px is not None and field == "title" \
                        and self._y(node) > self.title_zone_max_px:
                    continue
                score = float(probs[i][fi])
                if field not in best or score > best[field][0]:
                    best[field] = (score, i)

        assignments = [(field, idx) for field, (_, idx) in best.items()]
        claimed = {idx for _, idx in assignments}

        multi_idx = {self.cls_to_idx[f] for f in self.multi_fields}
        for i in range(len(nodes)):
            if i in claimed:
                continue
            j = max(range(len(self.classes)), key=lambda jj: float(probs[i][jj]))
            if j in multi_idx:
                assignments.append((self.classes[j], i))

        meta = {
            "backend": "greedy fallback", "status": "FALLBACK", "reason": reason,
            "constraints": [], "relaxed": ["all"], "fallback_used": True,
        }
        return self._assemble(nodes, probs, [], ["all"], None, assignments, solver_meta=meta)

    # ------------------------------------------------------------------
    def _assemble(self, nodes, probs, trace, relaxed, objective, assignments,
                  solver_meta=None) -> Dict[str, Any]:
        record: Dict[str, Any] = {}
        for cls, i in assignments:
            if cls == "other":
                continue
            j = self.cls_to_idx[cls]
            item = {
                "text": nodes[i].get("text", ""),
                "bbox": nodes[i].get("bbox") or [0, 0, 0, 0],
                "confidence": float(probs[i][j]),
                "node_id": nodes[i].get("id", i),
                "solver_variable": f"x[{i},{cls}]",
            }
            field = CLASS_TO_FIELD.get(cls, cls)
            if cls in self.multi_fields:
                record.setdefault(field, []).append(item)  # aggregate into a list
            else:
                record[field] = item
        record["_solver"] = solver_meta or {
            "backend": "OR-Tools SCIP", "status": "OPTIMAL",
            "objective_value": float(objective) if objective is not None else None,
            "constraints": trace, "relaxed": relaxed, "fallback_used": False,
        }
        return record

    @staticmethod
    def _y(node) -> float:
        bbox = node.get("bbox") or [0, 0, 0, 0]
        return float(bbox[1]) if len(bbox) > 1 else 0.0
