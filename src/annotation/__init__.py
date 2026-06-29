"""Meme annotation pipeline (ScrapeGraph-AI based).

Turns Know Your Meme URL records into information-rich, MongoDB-ready
documents. See ``annotate_memes.py`` (repo root) for the CLI entry point.
"""

from .config import AnnotationConfig
from . import meme_schema, url_store, annotator

__all__ = ["AnnotationConfig", "meme_schema", "url_store", "annotator"]
