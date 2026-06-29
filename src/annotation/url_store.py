"""
Input/output for the meme annotation pipeline.

Input: the ~40k Know Your Meme URL records, each shaped like::

    {
      "url": "https://knowyourmeme.com/editorials/guides/...",
      "Confirmed": true,
      "lastmod": "2026-06-23T12:16:57-04:00",
      "page_template_type": null,
      "last_scraped": null
    }

Records may live in a single JSON array file, a directory of such files, or a
JSON Lines file. ``last_scraped`` is used as the resume marker.

Output: one JSON object per line (JSONL) — each line is a ready-to-insert
MongoDB document combining the original metadata, the extracted ``meme``
payload, and annotation provenance.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Set


# --------------------------------------------------------------------------
# Loading input records
# --------------------------------------------------------------------------

def _load_json_file(path: Path) -> List[Dict[str, Any]]:
    """Load a .json (array or single object) or .jsonl file into a list."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported JSON structure in {path}")


def load_url_records(input_path: str) -> List[Dict[str, Any]]:
    """
    Load URL records from a file or a directory.

    * file  -> parsed directly (.json array/object or .jsonl)
    * dir   -> every *.json / *.jsonl inside is loaded and concatenated
    """
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    records: List[Dict[str, Any]] = []
    if p.is_dir():
        for child in sorted(p.iterdir()):
            if child.suffix in (".json", ".jsonl"):
                records.extend(_load_json_file(child))
    else:
        records.extend(_load_json_file(p))

    # Keep only records that actually carry a URL.
    return [r for r in records if isinstance(r, dict) and r.get("url")]


def iter_pending(
    records: Iterable[Dict[str, Any]],
    done_urls: Set[str],
    only_confirmed: bool = True,
    force: bool = False,
) -> Iterator[Dict[str, Any]]:
    """
    Yield records that still need annotation.

    A record is skipped when it is already in ``done_urls`` (resume), or when
    ``only_confirmed`` is set and ``Confirmed`` is falsy, or when it already has
    a ``last_scraped`` timestamp — unless ``force`` is True.
    """
    for r in records:
        url = r.get("url")
        if not url:
            continue
        if not force:
            if url in done_urls:
                continue
            if r.get("last_scraped"):
                continue
        if only_confirmed and not r.get("Confirmed", False):
            continue
        yield r


# --------------------------------------------------------------------------
# Output documents
# --------------------------------------------------------------------------

def url_id(url: str) -> str:
    """Stable Mongo _id derived from the URL (sha1 hex)."""
    return hashlib.sha1(url.encode("utf-8")).hexdigest()


def build_document(
    record: Dict[str, Any],
    template_type: str,
    payload: Dict[str, Any],
    *,
    model: str,
    provider: str,
    error: str | None = None,
) -> Dict[str, Any]:
    """
    Assemble a MongoDB-ready document from the source record + extracted payload.
    """
    now = datetime.now(timezone.utc).isoformat()
    doc: Dict[str, Any] = {
        "_id": url_id(record["url"]),
        "url": record["url"],
        "confirmed": bool(record.get("Confirmed", False)),
        "lastmod": record.get("lastmod"),
        "page_template_type": template_type,
        "last_scraped": now,
        "meme": payload,
        "_annotation": {
            "provider": provider,
            "model": model,
            "schema_family": "editorial" if template_type == "editorial" else "entry",
            "ok": error is None,
            "error": error,
            "annotated_at": now,
        },
    }
    return doc


class AnnotationWriter:
    """Append-only JSONL writer with resume support."""

    def __init__(self, output_path: str):
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = None

    def existing_urls(self) -> Set[str]:
        """URLs already present in the output file (for resume)."""
        done: Set[str] = set()
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        done.add(json.loads(line)["url"])
                    except (json.JSONDecodeError, KeyError):
                        continue
        return done

    def __enter__(self) -> "AnnotationWriter":
        self._fh = self.path.open("a", encoding="utf-8")
        return self

    def write(self, doc: Dict[str, Any]) -> None:
        assert self._fh is not None, "AnnotationWriter must be used as a context manager"
        self._fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def __exit__(self, *exc) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
