"""
Annotators turn a single URL into a normalised meme record.

* ``ScrapeGraphAnnotator`` — the real thing. Wraps ScrapeGraph-AI's
  ``SmartScraperGraph``: it fetches/renders the page and asks the configured
  LLM to fill the schema defined in ``meme_schema``. ScrapeGraph-AI is imported
  lazily so the rest of the pipeline works without it installed.

* ``MockAnnotator`` — deterministic, network-free fake used for ``--mock`` runs
  and tests. It produces schema-shaped output derived from the URL slug.

Both expose the same interface: ``annotate(url, template_type) -> dict``.
"""

from __future__ import annotations

from typing import Any, Dict, Protocol
from urllib.parse import urlparse

from . import meme_schema
from .config import AnnotationConfig


class Annotator(Protocol):
    def annotate(self, url: str, template_type: str) -> Dict[str, Any]: ...


# --------------------------------------------------------------------------
# Real annotator
# --------------------------------------------------------------------------

class ScrapeGraphAnnotator:
    """Extract meme data with ScrapeGraph-AI + the configured LLM."""

    def __init__(self, config: AnnotationConfig):
        self.config = config
        config.require_api_key()
        self._graph_cls = self._import_graph()
        self._graph_config = config.graph_config()

    @staticmethod
    def _import_graph():
        try:
            from scrapegraphai.graphs import SmartScraperGraph
        except ImportError as e:  # pragma: no cover - exercised only when dep missing
            raise RuntimeError(
                "scrapegraphai is not installed. Install it with "
                "`pip install -r requirements-annotation.txt` (and run "
                "`playwright install chromium` if needed), or use --mock."
            ) from e
        return SmartScraperGraph

    def annotate(self, url: str, template_type: str) -> Dict[str, Any]:
        prompt = meme_schema.build_prompt(template_type)
        # Some self-hosted/Ollama models behind an OpenAI-compatible gateway can't
        # do json-schema/function-calling; ANNOTATION_STRUCTURED_OUTPUT=false skips
        # the schema and relies on the prompt's "JSON only" + normalize().
        schema = (
            meme_schema.build_pydantic_schema(template_type)
            if self.config.structured_output else None
        )

        kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "source": url,
            "config": self._graph_config,
        }
        if schema is not None:
            kwargs["schema"] = schema

        graph = self._graph_cls(**kwargs)
        result = graph.run()

        # ScrapeGraph-AI may return a pydantic model, a dict, or a JSON string.
        if hasattr(result, "model_dump"):
            result = result.model_dump()
        elif isinstance(result, str):
            import json
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                result = {}
        if not isinstance(result, dict):
            result = {}

        result = meme_schema.normalize(result, template_type)
        if not result.get("title"):
            raise RuntimeError(
                "Annotation extracted no title — page is likely blocked or returned a "
                "non-content response (IP ban?). Set ANNOTATION_PROXY_URL in .env."
            )
        return result


# --------------------------------------------------------------------------
# Mock annotator (offline / tests)
# --------------------------------------------------------------------------

class MockAnnotator:
    """Network-free annotator that fabricates schema-shaped data from the URL."""

    def __init__(self, config: AnnotationConfig | None = None):
        self.config = config

    def annotate(self, url: str, template_type: str) -> Dict[str, Any]:
        slug = urlparse(url).path.rstrip("/").split("/")[-1] or "untitled"
        title = slug.replace("-", " ").replace("_", " ").strip().title()
        rec = meme_schema.empty_record(template_type)
        rec["title"] = title
        if meme_schema.spec_family(template_type) == "editorial":
            rec["summary"] = f"[mock] Summary for editorial '{title}'."
            rec["category"] = "Guide"
            rec["tags"] = ["mock"]
        else:
            rec["about"] = f"[mock] {title} is a meme. This is mock annotation output."
            rec["entry_type"] = ["Image Macro"]
            rec["status"] = "Confirmed"
            rec["tags"] = ["mock"]
        return meme_schema.normalize(rec, template_type)


def build_annotator(config: AnnotationConfig, mock: bool = False) -> Annotator:
    """Factory: return a mock or real annotator."""
    return MockAnnotator(config) if mock else ScrapeGraphAnnotator(config)
