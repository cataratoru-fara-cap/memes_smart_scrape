"""
Heuristic section extractor (tier ③) — long-form fields the GNN can't localise.

The GNN picks single nodes; ``about`` / ``origin`` / ``spread`` are *sections*:
a run of paragraph nodes under a heading. This recovers them with no LLM and no
model — find the section heading, collect the paragraph nodes after it up to the
next heading, and join their text.

Pure stdlib; operates on the same ``{id,text,tag,bbox}`` nodes the renderer
produces (own-text only, so paragraphs aren't double-counted by containers).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

_WS = re.compile(r"\s+")


def _norm(s: Any) -> str:
    return _WS.sub(" ", str(s).strip().lower()).rstrip(":")


# Headings we actually extract -> output field name.
_TARGET = {
    "about": "about",
    "origin": "origin_description",
    "origins": "origin_description",
    "spread": "spread_description",
    "development": "spread_description",
    "online presence": "spread_description",
}

# Every heading that should *bound* a section (targets + other KYM sections), so
# collection for one section stops at the start of the next.
_BOUNDARIES = set(_TARGET) | {
    "meaning", "notable examples", "various examples", "search interest",
    "external references", "recent videos", "recent images", "editorial notes",
    "top comments", "related entries", "tags",
}

_PARAGRAPH_TAGS = {"p", "div", "span", "li", "blockquote", ""}
_HEADING_TAGS = {"h1", "h2", "h3", "h4"}
_MIN_PARA_LEN = 30


def _y(node: Dict[str, Any]) -> float:
    bbox = node.get("bbox") or [0, 0, 0, 0]
    return float(bbox[1]) if len(bbox) > 1 else 0.0


def _x(node: Dict[str, Any]) -> float:
    bbox = node.get("bbox") or [0, 0, 0, 0]
    return float(bbox[0]) if bbox else 0.0


def _is_heading(node: Dict[str, Any], norm_text: str) -> bool:
    tag = str(node.get("tag", "")).lower()
    return norm_text in _BOUNDARIES and (tag in _HEADING_TAGS or len(norm_text) <= 25)


def extract_sections(nodes: List[Dict[str, Any]], max_chars: int = 2000) -> Dict[str, str]:
    """
    Return {output_field: text} for whichever of about / origin_description /
    spread_description are found. Sections with no body text are omitted.
    """
    if not nodes:
        return {}

    # Reading order: top-to-bottom, then left-to-right.
    order = sorted(range(len(nodes)), key=lambda i: (_y(nodes[i]), _x(nodes[i])))

    # Heading positions within the reading order (pos -> output field or None).
    headings: List[tuple] = []
    for pos, i in enumerate(order):
        nt = _norm(nodes[i].get("text", ""))
        if _is_heading(nodes[i], nt):
            headings.append((pos, _TARGET.get(nt)))

    result: Dict[str, str] = {}
    for h, (pos, out_field) in enumerate(headings):
        if out_field is None:
            continue
        end = headings[h + 1][0] if h + 1 < len(headings) else len(order)
        parts: List[str] = []
        seen: set[str] = set()
        for p in range(pos + 1, end):
            node = nodes[order[p]]
            txt = str(node.get("text", "")).strip()
            tag = str(node.get("tag", "")).lower()
            if len(txt) >= _MIN_PARA_LEN and tag in _PARAGRAPH_TAGS and txt not in seen:
                parts.append(txt)
                seen.add(txt)
        text = " ".join(parts).strip()
        if text and out_field not in result:  # first occurrence wins
            result[out_field] = text[:max_chars]
    return result
