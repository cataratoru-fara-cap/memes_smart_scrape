"""
Storage abstraction shared by the discovery script and the annotation pipeline.

The two stages communicate through a single ``urls`` collection of discovery
records (one per Know Your Meme URL) plus an ``annotations`` collection holding
the extracted, MongoDB-ready documents:

    discovery  --upsert-->  urls  --ingest-->  annotation  --write-->  annotations
                                                       `--mark last_scraped--`

This module defines:
  * pure helpers (``url_doc_id``, ``merge_discovery``) with no I/O,
  * the ``Store`` protocol both backends implement,
  * ``InMemoryStore`` — a dependency-free backend used by tests and as a safe
    fallback when pymongo is unavailable.

The pymongo-backed implementation lives in ``src/db/mongo.py``.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, Iterator, List, Protocol, Set

URLS_COLLECTION = "urls"
ANNOTATIONS_COLLECTION = "annotations"

# Fields a discovery run is allowed to overwrite on an existing record.
# Notably absent: ``last_scraped`` (owned by the annotation stage) and
# ``Confirmed`` (merged specially — never downgraded).
_DISCOVERY_FIELDS = ("url", "namespace", "lastmod", "page_template_type")


def url_doc_id(url: str) -> str:
    """Stable document id derived from the URL (sha1 hex)."""
    return hashlib.sha1(url.encode("utf-8")).hexdigest()


def merge_discovery(old: Dict[str, Any] | None, new: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge a freshly discovered record into an existing one (pure function).

    Rules:
      * ``last_scraped`` already in the store is never overwritten (annotation
        progress must not be lost). Only when the store has no value yet does
        the incoming one apply — so an initial import of the existing JSON keeps
        whatever progress it already recorded.
      * ``Confirmed`` is monotonic: once True it stays True. This lets the
        sitemap (confirmed) and the crawl (submissions/deadpool) cooperate
        without the crawl downgrading a confirmed entry.
      * ``lastmod`` is updated only when the new value is non-null, and is
        forced to None for non-confirmed entries (only confirmed memes carry a
        lastmod).
      * Other discovery fields overwrite.
    """
    merged: Dict[str, Any] = dict(old) if old else {}
    merged["url"] = new["url"]
    merged["_id"] = url_doc_id(new["url"])

    for field in _DISCOVERY_FIELDS:
        if field == "lastmod":
            if new.get("lastmod") is not None:
                merged["lastmod"] = new["lastmod"]
            else:
                merged.setdefault("lastmod", None)
        elif field in new:
            merged[field] = new[field]

    merged["Confirmed"] = bool((old or {}).get("Confirmed", False) or new.get("Confirmed", False))
    # Invariant: only confirmed entries carry a lastmod; non-confirmed -> None.
    if not merged["Confirmed"]:
        merged["lastmod"] = None
    old_last_scraped = (old or {}).get("last_scraped")
    merged["last_scraped"] = old_last_scraped if old_last_scraped is not None else new.get("last_scraped")
    return merged


def _record_view(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Return the discovery-record view of a stored doc (drop internal keys)."""
    return {k: v for k, v in doc.items() if k != "_id"}


class Store(Protocol):
    """Common interface for the discovery sink and the annotation source/sink."""

    def upsert_urls(self, records: Iterable[Dict[str, Any]]) -> Dict[str, int]: ...
    def iter_pending(self, only_confirmed: bool = True, force: bool = False
                     ) -> Iterator[Dict[str, Any]]: ...
    def annotated_urls(self) -> Set[str]: ...
    def save_annotation(self, doc: Dict[str, Any]) -> None: ...
    def iter_annotations(self) -> Iterator[Dict[str, Any]]: ...
    def count_urls(self) -> int: ...
    def close(self) -> None: ...


class InMemoryStore:
    """Dict-backed Store. No external dependencies — used in tests/fallback."""

    def __init__(self) -> None:
        self.urls: Dict[str, Dict[str, Any]] = {}
        self.annotations: Dict[str, Dict[str, Any]] = {}

    def upsert_urls(self, records: Iterable[Dict[str, Any]]) -> Dict[str, int]:
        stats = {"added": 0, "updated": 0}
        for rec in records:
            url = rec.get("url")
            if not url:
                continue
            _id = url_doc_id(url)
            old = self.urls.get(_id)
            self.urls[_id] = merge_discovery(old, rec)
            stats["updated" if old else "added"] += 1
        return stats

    def iter_pending(self, only_confirmed: bool = True, force: bool = False
                     ) -> Iterator[Dict[str, Any]]:
        for doc in self.urls.values():
            if not force and doc.get("last_scraped"):
                continue
            if only_confirmed and not doc.get("Confirmed", False):
                continue
            yield _record_view(doc)

    def annotated_urls(self) -> Set[str]:
        return {d["url"] for d in self.urls.values() if d.get("last_scraped")}

    def save_annotation(self, doc: Dict[str, Any]) -> None:
        _id = doc.get("_id") or url_doc_id(doc["url"])
        self.annotations[_id] = doc
        url_doc = self.urls.get(_id)
        if url_doc is not None:
            url_doc["last_scraped"] = doc.get("last_scraped")
            url_doc["page_template_type"] = doc.get("page_template_type")

    def iter_annotations(self) -> Iterator[Dict[str, Any]]:
        yield from self.annotations.values()

    def count_urls(self) -> int:
        return len(self.urls)

    def close(self) -> None:  # nothing to release
        pass
