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

_WS = re.compile(r"\s+")
_YEAR = re.compile(r"(?:18|19|20)\d{2}")

# Minimum match score for a value to be aligned to a node at all.
MIN_MATCH = 0.45

# Singleton class -> how to read its value from a `meme` payload.
_SINGLETON_SRC = {
    "title": lambda p: p.get("title"),
    "type": lambda p: (p.get("entry_type") or [None])[0]
    if isinstance(p.get("entry_type"), list) else p.get("entry_type"),
    "status": lambda p: p.get("status"),
    "origin": lambda p: p.get("origin"),
    "year": lambda p: None if str(p.get("year")) in ("0", "None", "", "None")
    else p.get("year"),
    "parent_meme": lambda p: p.get("parent_meme"),
}

# Multi-label class -> the list field it reads from a `meme` payload.
_MULTI_SRC = {
    "tag": "tags",
    "alias": "aliases",
    "region": "region",
    "related": "related_memes",
}


def normalize_text(s: Any) -> str:
    """Lowercase, strip, collapse internal whitespace."""
    return _WS.sub(" ", str(s).strip().lower())


# --------------------------------------------------------------------------
# 1. Extract localizable values from an annotation payload
# --------------------------------------------------------------------------

def localizable_field_values(meme_payload: Dict[str, Any]) -> Dict[str, str]:
    """
    Map a ``meme`` payload (entry schema) to {singleton_class: value_string}.
    Missing/empty values are omitted.
    """
    if not isinstance(meme_payload, dict):
        return {}
    out: Dict[str, str] = {}
    for cls, getter in _SINGLETON_SRC.items():
        val = getter(meme_payload)
        if val not in (None, "", 0):
            out[cls] = str(val)
    return out


def multi_field_values(meme_payload: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Map a ``meme`` payload to {multi_class: [value_strings]} for the multi-label
    fields (tags/aliases/region/related_memes). Empty lists are omitted.
    """
    if not isinstance(meme_payload, dict):
        return {}
    out: Dict[str, List[str]] = {}
    for cls, field in _MULTI_SRC.items():
        raw = meme_payload.get(field)
        if isinstance(raw, list):
            vals = [str(v).strip() for v in raw if str(v).strip()]
        elif isinstance(raw, str) and raw.strip():
            vals = [raw.strip()]
        else:
            vals = []
        if vals:
            out[cls] = vals
    return out


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
    nodes: List[Dict[str, Any]],
    singleton_values: Dict[str, str],
    multi_values: Dict[str, List[str]] | None = None,
) -> List[Dict[str, Any]]:
    """
    Return a copy of ``nodes`` with a ``label`` on each (a student class,
    default "other").

    Greedy global assignment over all (class, value, node) candidates, highest
    score first, with these capacities:
      * each node is labelled at most once;
      * a singleton class is used at most once (one node per field);
      * a multi class may label many nodes, but each distinct *value* lands on
        one node (so a 3-tag page gets up to 3 'tag' nodes).
    """
    multi_values = multi_values or {}
    labeled = [dict(n) for n in nodes]
    for n in labeled:
        n.setdefault("label", "other")

    # (score, class, value_key, node_idx, is_singleton)
    candidates = []
    for cls, value in singleton_values.items():
        for idx, node in enumerate(labeled):
            s = match_score(node.get("text", ""), value, cls)
            if s >= MIN_MATCH:
                candidates.append((s, cls, None, idx, True))
    for cls, values in multi_values.items():
        for value in values:
            vkey = normalize_text(value)
            for idx, node in enumerate(labeled):
                s = match_score(node.get("text", ""), value, cls)
                if s >= MIN_MATCH:
                    candidates.append((s, cls, vkey, idx, False))

    candidates.sort(key=lambda c: c[0], reverse=True)

    used_singletons: set[str] = set()
    used_multi_values: set[tuple] = set()
    used_nodes: set[int] = set()
    for score, cls, vkey, idx, is_singleton in candidates:
        if idx in used_nodes:
            continue
        if is_singleton:
            if cls in used_singletons:
                continue
            used_singletons.add(cls)
        else:
            pair = (cls, vkey)
            if pair in used_multi_values:
                continue
            used_multi_values.add(pair)
        labeled[idx]["label"] = cls
        used_nodes.add(idx)

    return labeled


def label_page(
    nodes: List[Dict[str, Any]], meme_payload: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Convenience: extract singleton + multi values from a payload and label."""
    return align_nodes_to_labels(
        nodes,
        localizable_field_values(meme_payload),
        multi_field_values(meme_payload),
    )
