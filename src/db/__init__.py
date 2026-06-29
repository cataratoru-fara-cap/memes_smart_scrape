"""Shared storage layer (MongoDB hub) for discovery + annotation.

``InMemoryStore`` is always importable (pure stdlib). ``MongoStore`` /
``get_store`` require pymongo and are imported lazily on use.
"""

from .store import (
    ANNOTATIONS_COLLECTION,
    URLS_COLLECTION,
    InMemoryStore,
    Store,
    merge_discovery,
    url_doc_id,
)

__all__ = [
    "InMemoryStore",
    "Store",
    "url_doc_id",
    "merge_discovery",
    "URLS_COLLECTION",
    "ANNOTATIONS_COLLECTION",
]
