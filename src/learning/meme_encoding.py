"""
Meme label space + graph construction for the student GNN.

Reuses the canonical feature/edge primitives in ``encoding`` (so the meme model
sees graphs built identically to the books model) and only swaps in the meme
class space from ``meme_config``. torch / torch_geometric are imported lazily
inside ``page_to_graph`` so this module stays importable without them.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .meme_config import MEME_CLASSES

# Feature width is shared with the books model (text 128 + visual 4 + tag 14).
# Read it from ``encoding`` when torch is available; otherwise fall back to the
# fixed constant so this module imports offline (encoding imports torch at top).
try:
    from . import encoding as _encoding
    INPUT_DIM = _encoding.INPUT_DIM
except Exception:  # noqa: BLE001 - torch missing in offline/test envs
    INPUT_DIM = 146

MEME_LABEL_MAP: Dict[str, int] = {name: i for i, name in enumerate(MEME_CLASSES)}
_OTHER_IDX = MEME_LABEL_MAP["other"]
# Tolerate the discovery/annotation background spellings.
MEME_LABEL_MAP.setdefault("none", _OTHER_IDX)
MEME_LABEL_MAP.setdefault("", _OTHER_IDX)


def label_index(label: Any) -> int:
    """Map a label string to its class index ('other' for anything unknown)."""
    return MEME_LABEL_MAP.get(str(label), _OTHER_IDX)


def page_to_graph(nodes: List[Dict[str, Any]]):
    """Build (Data, labels) for a labeled meme page. Used by the trainer."""
    import torch
    from torch_geometric.data import Data
    from . import encoding

    feats = torch.stack([encoding.node_features(nd) for nd in nodes])
    labels = torch.tensor([label_index(nd.get("label")) for nd in nodes], dtype=torch.long)
    edge_index = encoding.build_edges(nodes, k=3)
    return Data(x=feats, edge_index=edge_index), labels
