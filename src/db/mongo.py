"""
MongoDB-backed Store (pymongo). Imported lazily so the rest of the codebase
works without pymongo installed.

Connection settings come from the environment:
    MONGODB_URI   (default: mongodb://localhost:27017)
    MONGODB_DB    (default: memes)

Collections (see src/db/store.py):
    urls         — discovery records, _id = sha1(url)
    annotations  — extracted documents, _id = sha1(url)
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Iterator, Set

from .store import (
    ANNOTATIONS_COLLECTION,
    URLS_COLLECTION,
    InMemoryStore,
    merge_discovery,
    url_doc_id,
)


def _env_uri() -> str:
    return os.getenv("MONGODB_URI", "mongodb://localhost:27017")


def _env_db() -> str:
    return os.getenv("MONGODB_DB", "memes")


class MongoStore:
    """pymongo implementation of the Store protocol."""

    def __init__(self, uri: str | None = None, db_name: str | None = None):
        try:
            from pymongo import MongoClient, UpdateOne  # noqa: F401
        except ImportError as e:  # pragma: no cover - only when dep missing
            raise RuntimeError(
                "pymongo is not installed. `pip install pymongo` "
                "(or use the JSON file backend / InMemoryStore)."
            ) from e

        self._UpdateOne = UpdateOne
        self.client = MongoClient(uri or _env_uri())
        self.db = self.client[db_name or _env_db()]
        self.urls = self.db[URLS_COLLECTION]
        self.annotations = self.db[ANNOTATIONS_COLLECTION]
        # Idempotent indexes.
        self.urls.create_index("url", unique=True)
        self.urls.create_index("Confirmed")
        self.urls.create_index("last_scraped")

    def upsert_urls(self, records: Iterable[Dict[str, Any]]) -> Dict[str, int]:
        ops = []
        ids = []
        new_records: Dict[str, Dict[str, Any]] = {}
        for rec in records:
            url = rec.get("url")
            if not url:
                continue
            _id = url_doc_id(url)
            ids.append(_id)
            new_records[_id] = rec

        # Fetch existing docs in one round-trip so Confirmed/last_scraped merge
        # correctly (pure merge_discovery, identical to InMemoryStore).
        existing = {d["_id"]: d for d in self.urls.find({"_id": {"$in": ids}})} if ids else {}

        stats = {"added": 0, "updated": 0}
        for _id, rec in new_records.items():
            old = existing.get(_id)
            merged = merge_discovery(old, rec)
            ops.append(self._UpdateOne({"_id": _id}, {"$set": merged}, upsert=True))
            stats["updated" if old else "added"] += 1

        if ops:
            self.urls.bulk_write(ops, ordered=False)
        return stats

    def iter_pending(self, only_confirmed: bool = True, force: bool = False
                     ) -> Iterator[Dict[str, Any]]:
        query: Dict[str, Any] = {}
        if not force:
            query["last_scraped"] = None
        if only_confirmed:
            query["Confirmed"] = True
        for doc in self.urls.find(query):
            doc.pop("_id", None)
            yield doc

    def annotated_urls(self) -> Set[str]:
        return {d["url"] for d in self.urls.find({"last_scraped": {"$ne": None}}, {"url": 1})}

    def save_annotation(self, doc: Dict[str, Any]) -> None:
        _id = doc.get("_id") or url_doc_id(doc["url"])
        doc["_id"] = _id
        self.annotations.replace_one({"_id": _id}, doc, upsert=True)
        self.urls.update_one(
            {"_id": _id},
            {"$set": {
                "last_scraped": doc.get("last_scraped"),
                "page_template_type": doc.get("page_template_type"),
            }},
        )

    def iter_annotations(self) -> Iterator[Dict[str, Any]]:
        yield from self.annotations.find({})

    def count_urls(self) -> int:
        return self.urls.count_documents({})

    def close(self) -> None:
        self.client.close()


def get_store(uri: str | None = None, db_name: str | None = None):
    """
    Return a MongoStore, or raise with a clear message if pymongo is missing.
    Tests use InMemoryStore directly; production code calls this.
    """
    return MongoStore(uri=uri, db_name=db_name)


__all__ = ["MongoStore", "get_store", "InMemoryStore"]
