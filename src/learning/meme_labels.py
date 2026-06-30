"""
Teacher -> student label bridge (pure stdlib, fully testable offline).

The teacher (ScrapeGraph-AI) gives us *field values* per page. To train the
student GNN we need *per-node labels* on the rendered page. This module:

  1. pulls the node-localizable field values out of an annotation's ``meme``
     payload (``localizable_field_values``), and
  2. aligns each value to the single best-matching rendered node
     (``align_nodes_to_labels``), producing the weak supervision the GNN trains
     on — exactly analogous to the hand-labeled books pages, but generated
     automatically from the LLM annotations (distillation).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from .meme_config import SINGLETON_FIELDS

_WS = re.compile(r"\s+")
_YEAR = re.compile(r"(?:18|19|20)\d{2}")

# Minimum match score for a value to be aligned to a node at all.
MIN_MATCH = 0.45


def normalize_text(s: Any) -> str:
    """Lowercase, strip, collapse internal whitespace."""
    return _WS.sub(" ", str(s).strip().lower())


# --------------------------------------------------------------------------
# 1. Extract localizable values from an annotation payload
# --------------------------------------------------------------------------

def localizable_field_values(meme_payload: Dict[str, Any]) -> Dict[str, str]:
    """
    Map a ``meme`` payload (entry schema) to {field: value_string} for the
    student's singleton fields. Missing/empty values are omitted.
    """
    if not isinstance(meme_payload, dict):
        return {}
    out: Dict[str, str] = {}

    title = meme_payload.get("title")
    if title:
        out["title"] = str(title)

    entry_type = meme_payload.get("entry_type")
    if isinstance(entry_type, list) and entry_type:
        out["type"] = str(entry_type[0])
    elif isinstance(entry_type, str) and entry_type.strip():
        out["type"] = entry_type

    status = meme_payload.get("status")
    if status:
        out["status"] = str(status)

    origin = meme_payload.get("origin")
    if origin:
        out["origin"] = str(origin)

    year = meme_payload.get("year")
    if year and str(year) not in ("0", "None", ""):
        out["year"] = str(year)

    return {f: v for f, v in out.items() if f in SINGLETON_FIELDS}


# --------------------------------------------------------------------------
# 2. Align values to nodes
# --------------------------------------------------------------------------

def match_score(node_text: str, value: str, field: str) -> float:
    """
    Similarity in [0, 1] between a node's text and a target value.

    Rewards exact matches and tight containment (so the specific node beats a
    big container that merely includes the value). ``year`` matches on the
    presence of the 4-digit value.
    """
    nt, v = normalize_text(node_text), normalize_text(value)
    if not nt or not v:
        return 0.0

    if field == "year":
        return 1.0 if v in _YEAR.findall(nt) else 0.0

    if nt == v:
        return 1.0
    if v in nt:
        # Prefer shorter containers (closer to a leaf node holding just the value).
        return 0.5 + 0.5 * (len(v) / len(nt))
    if nt in v:
        return 0.5 * (len(nt) / len(v))

    # Token Jaccard as a weak fallback.
    a, b = set(nt.split()), set(v.split())
    if not a or not b:
        return 0.0
    return 0.4 * len(a & b) / len(a | b)


def align_nodes_to_labels(
    nodes: List[Dict[str, Any]], field_values: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Return a copy of ``nodes`` with a ``label`` on each (one of the student
    classes, default "other").

    Greedy global assignment: take the highest-scoring (field, node) pairs
    first, enforcing at most one node per field and one field per node, so each
    value lands on its best free node.
    """
    labeled = [dict(n) for n in nodes]
    for n in labeled:
        n.setdefault("label", "other")

    candidates = []
    for field, value in field_values.items():
        for idx, node in enumerate(labeled):
            s = match_score(node.get("text", ""), value, field)
            if s >= MIN_MATCH:
                candidates.append((s, field, idx))

    candidates.sort(key=lambda c: c[0], reverse=True)

    used_fields: set[str] = set()
    used_nodes: set[int] = set()
    for score, field, idx in candidates:
        if field in used_fields or idx in used_nodes:
            continue
        labeled[idx]["label"] = field
        used_fields.add(field)
        used_nodes.add(idx)

    return labeled


def label_page(
    nodes: List[Dict[str, Any]], meme_payload: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Convenience: extract values from a payload and label the nodes."""
    return align_nodes_to_labels(nodes, localizable_field_values(meme_payload))
