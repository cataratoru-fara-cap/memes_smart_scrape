"""
Meme extraction pipeline — the student model at inference time.

    URL ──(render)──▶ nodes ──(GNN)──▶ per-node scores ──(priors)──▶
        ──(ILP solver)──▶ proof-carrying meme record + σ(P) drift

Mirrors ``src/pipeline_fixed.SmartScrapePipeline`` but for the meme field set.
Everything heavy is lazy:

  * No torch / no trained model  -> uniform GNN scores; heuristic priors + the
    ILP/greedy solver still produce a result (lets you smoke-test plumbing
    before installing torch).
  * No ortools                   -> solver uses its greedy fallback.
  * Renderer is injectable       -> pass nodes directly (or a fake renderer) to
    run fully offline.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Sequence

from .meme_config import MEME_CLASSES, MEME_MODEL_PATH, STABILITY_THRESHOLD, TITLE_ZONE_MAX_PX
from .meme_encoding import INPUT_DIM
from src.reasoning.meme_solver import MemeConstraintSolver, status_ok, year_ok

_YEAR_RE = re.compile(r"(?:18|19|20)\d{2}")

# Lowercased KYM type words used by the title/type priors.
_TYPE_VOCAB = {
    "image macro", "catchphrase", "exploitable", "copypasta", "reaction",
    "snowclone", "viral video", "slang", "hashtag", "pop culture reference",
    "photoshop", "remix", "parody", "challenge", "character",
}


class MemeExtractionPipeline:
    CLASSES = MEME_CLASSES

    def __init__(self, model_path: str = MEME_MODEL_PATH, renderer: Any = None,
                 use_priors: bool = True, title_zone_max_px: float | None = TITLE_ZONE_MAX_PX):
        self.model_path = model_path
        self.renderer = renderer
        self.use_priors = use_priors
        self.solver = MemeConstraintSolver(title_zone_max_px=title_zone_max_px)
        self._model = None
        self._model_trained = False
        self._load_model()

    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        if not os.path.exists(self.model_path):
            print(f"[MemePipeline] No model at '{self.model_path}' — "
                  "running uniform GNN + priors + ILP. Train with train_meme_gnn.py.")
            return
        try:
            import torch
            from .gnn_model import SmartScrapeGNN
            model = SmartScrapeGNN(input_dim=INPUT_DIM, hidden_dim=64, num_classes=len(self.CLASSES))
            model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
            model.eval()
            self._model = model
            self._model_trained = True
            print(f"[MemePipeline] Loaded meme GNN weights from '{self.model_path}'")
        except Exception as e:  # noqa: BLE001
            print(f"[MemePipeline] WARNING: could not load model ({e}) — using uniform scores.")

    # ------------------------------------------------------------------
    def run(self, url: Optional[str] = None,
            nodes: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Extract meme fields from a URL (rendered) or a pre-rendered node list."""
        if nodes is None:
            if url is None or self.renderer is None:
                raise ValueError("Provide nodes, or both a url and a renderer.")
            nodes = self.renderer.render(url)
        if not nodes:
            return {"_meta": {"error": "no nodes", "url": url}}

        probs = self._run_gnn(nodes)
        if self.use_priors:
            self._inject_priors(nodes, probs)

        page_height = self._page_height(nodes)
        record = self.solver.solve(nodes, probs, page_height)

        stability = self._stability(probs)
        record["_meta"] = {
            "url": url,
            "model_trained": self._model_trained,
            "stability_score": stability,
            "drift_alert": stability < STABILITY_THRESHOLD,
            "use_priors": self.use_priors,
            "num_nodes": len(nodes),
            "classes": self.CLASSES,
        }
        return record

    # ------------------------------------------------------------------
    def _run_gnn(self, nodes: List[Dict[str, Any]]) -> List[List[float]]:
        """Return an N x C score matrix (list of lists). Uniform if no model."""
        c = len(self.CLASSES)
        if self._model is not None:
            try:
                import torch
                from . import encoding
                from torch_geometric.data import Data
                feats = torch.stack([encoding.node_features(nd) for nd in nodes])
                edge_index = encoding.build_edges(nodes, k=3)
                with torch.no_grad():
                    logits = self._model(Data(x=feats, edge_index=edge_index))
                    return torch.exp(logits).tolist()
            except Exception as e:  # noqa: BLE001
                print(f"[MemePipeline] GNN inference failed ({e}) — uniform scores.")
        return [[1.0 / c] * c for _ in nodes]

    # ------------------------------------------------------------------
    def _inject_priors(self, nodes: List[Dict[str, Any]], probs: List[List[float]]) -> None:
        """Bounded heuristic nudges (|Δ| ≤ 0.3) on top of GNN scores."""
        idx = self.solver.cls_to_idx
        for i, node in enumerate(nodes):
            text = str(node.get("text", "")).strip()
            tag = str(node.get("tag", "")).lower()
            y = self._y(node)
            short = len(text) < 40

            if "year" in idx and short and year_ok(text) and len(text) <= 12:
                probs[i][idx["year"]] += 0.3
            if "status" in idx and short and status_ok(text):
                probs[i][idx["status"]] += 0.3
            if "type" in idx and self._norm(text) in _TYPE_VOCAB:
                probs[i][idx["type"]] += 0.3
            if "title" in idx and tag in ("h1", "h2") and y < 600:
                probs[i][idx["title"]] += 0.3

    # ------------------------------------------------------------------
    @staticmethod
    def _stability(probs: Sequence[Sequence[float]]) -> float:
        """σ(P) = mean over nodes of (top1 - top2) class probability."""
        if not probs:
            return 0.0
        total = 0.0
        for row in probs:
            top = sorted(row, reverse=True)
            total += (top[0] - top[1]) if len(top) > 1 else top[0]
        return total / len(probs)

    @staticmethod
    def _page_height(nodes: List[Dict[str, Any]]) -> float:
        h = 2000.0
        for node in nodes:
            bbox = node.get("bbox") or [0, 0, 0, 0]
            if len(bbox) >= 4:
                bottom = float(bbox[1]) + float(bbox[3])
                h = max(h, bottom)
        return h

    @staticmethod
    def _y(node) -> float:
        bbox = node.get("bbox") or [0, 0, 0, 0]
        return float(bbox[1]) if len(bbox) > 1 else 0.0

    @staticmethod
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip().lower())
