"""
Configuration for the meme GNN+ILP *student* model.

The student is the meme-domain retarget of the books title/price extractor:
instead of {price, title, other} it classifies Know Your Meme info-box nodes
into the short, node-localizable fields below and an ILP solver enforces the
meme-specific constraints (uniqueness + value formats).

Long-form sections (about / origin / spread descriptions) are NOT student
targets — they stay the teacher's (ScrapeGraph-AI) job. The student handles
exactly the fields that map to a single page node.
"""

from __future__ import annotations

import os

# Per-node class space. "other" is background. Order is fixed: changing it
# invalidates a trained meme_model.pt (the GNN's output head is positional).
MEME_CLASSES = ["title", "type", "status", "origin", "year", "other"]

# Fields the GNN+ILP actually selects (everything except background).
SINGLETON_FIELDS = ["title", "type", "status", "origin", "year"]

# Valid Know Your Meme moderation statuses (lowercased, for the format rule).
STATUS_VALUES = {"confirmed", "submission", "deadpool", "researching"}

# Trained weights (produced by train_meme_gnn.py).
MEME_MODEL_PATH = os.getenv("MEME_MODEL_PATH", "meme_model.pt")

# Dataset / artifact paths.
LABELED_DATASET_PATH = os.getenv("MEME_LABELED_PATH", "data/labeled_memes.json")

# Optional header-zone constraint for the title (title must sit above this y).
# Disabled by default (None) because KYM layouts vary; enable per-run if useful.
TITLE_ZONE_MAX_PX = float(os.getenv("MEME_TITLE_ZONE_MAX_PX", "0")) or None

# Drift: average top1-top2 margin below this flags an unstable extraction.
STABILITY_THRESHOLD = float(os.getenv("MEME_STABILITY_THRESHOLD", "0.15"))
