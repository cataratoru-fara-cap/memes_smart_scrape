"""
SmartScrape — Meme Annotation Pipeline (ScrapeGraph-AI)

Reads Know Your Meme URL records, extracts information-rich structured data with
ScrapeGraph-AI + an LLM, and writes MongoDB-ready JSONL documents. Resumable:
already-annotated URLs (present in the output, or carrying ``last_scraped``) are
skipped automatically.

Examples
--------
    # Offline smoke test — no API key, no network:
    python annotate_memes.py --mock --limit 5

    # Real run against your data with OpenAI (set OPENAI_API_KEY in .env):
    python annotate_memes.py --input data/meme_urls.json --output data/annotations.jsonl

    # Swap provider/model on the fly:
    python annotate_memes.py --provider anthropic --model claude-haiku-4-5

Load into MongoDB afterwards:
    mongoimport --db memes --collection entries --file data/annotations.jsonl
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from src.annotation.config import AnnotationConfig
from src.annotation import meme_schema, url_store
from src.annotation.annotator import build_annotator


class _RateLimiter:
    """Simple thread-safe minimum-interval limiter (0 = unlimited)."""

    def __init__(self, per_min: int):
        self._interval = 60.0 / per_min if per_min and per_min > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        if self._interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            sleep_for = max(0.0, self._next - now)
            self._next = max(now, self._next) + self._interval
        if sleep_for > 0:
            time.sleep(sleep_for)


def _annotate_one(annotator, record: Dict[str, Any], cfg: AnnotationConfig,
                  limiter: _RateLimiter) -> Dict[str, Any]:
    """Annotate a single record with retries/backoff; never raises."""
    url = record["url"]
    template_type = record.get("page_template_type") or meme_schema.detect_template_type(url)

    last_err = None
    for attempt in range(1, cfg.max_retries + 1):
        try:
            limiter.wait()
            payload = annotator.annotate(url, template_type)
            return url_store.build_document(
                record, template_type, payload,
                model=cfg.model, provider=cfg.provider,
            )
        except Exception as e:  # noqa: BLE001 - we want to capture any failure
            last_err = e
            if attempt < cfg.max_retries:
                time.sleep(2 ** attempt)  # 2s, 4s, 8s, ...

    # All retries failed — emit an error document so we don't retry forever.
    return url_store.build_document(
        record, template_type, meme_schema.empty_record(template_type),
        model=cfg.model, provider=cfg.provider, error=str(last_err),
    )


class _FileSink:
    """Write annotation docs to a JSONL file (resume via existing_urls)."""

    def __init__(self, output_path: str):
        self.writer = url_store.AnnotationWriter(output_path)

    def existing_urls(self):
        return self.writer.existing_urls()

    def __enter__(self):
        self.writer.__enter__()
        return self

    def write(self, doc: Dict[str, Any]) -> None:
        self.writer.write(doc)

    def __exit__(self, *exc) -> None:
        self.writer.__exit__(*exc)


class _MongoSink:
    """Write annotation docs to MongoDB (annotations coll + mark last_scraped)."""

    def __init__(self, store):
        self.store = store

    def __enter__(self):
        return self

    def write(self, doc: Dict[str, Any]) -> None:
        self.store.save_annotation(doc)

    def __exit__(self, *exc) -> None:
        self.store.close()


def _gather(cfg: AnnotationConfig, force: bool):
    """Resolve (pending_records, sink, source_label) for the configured source."""
    if cfg.source == "mongo":
        from src.db.mongo import get_store
        store = get_store()
        pending = list(store.iter_pending(only_confirmed=cfg.only_confirmed, force=force))
        return pending, _MongoSink(store), f"mongo({store.count_urls()} urls)"

    records = url_store.load_url_records(cfg.input_path)
    sink = _FileSink(cfg.output_path)
    done = sink.existing_urls()
    pending = list(
        url_store.iter_pending(records, done, only_confirmed=cfg.only_confirmed, force=force)
    )
    return pending, sink, f"file:{cfg.input_path} ({len(records)} urls)"


def run(cfg: AnnotationConfig, *, mock: bool, limit: int, force: bool) -> int:
    pending, sink, source_label = _gather(cfg, force)

    if limit and limit > 0:
        pending = pending[:limit]

    print(f"[annotate] source={source_label}  pending={len(pending)}")
    print(f"[annotate] provider={cfg.provider}  model={cfg.model}  "
          f"mock={mock}  concurrency={cfg.concurrency}")
    if not pending:
        print("[annotate] Nothing to do.")
        return 0

    annotator = build_annotator(cfg, mock=mock)
    limiter = _RateLimiter(cfg.rate_limit_per_min)

    ok = fail = 0
    with sink:
        with ThreadPoolExecutor(max_workers=max(1, cfg.concurrency)) as pool:
            futures = {
                pool.submit(_annotate_one, annotator, r, cfg, limiter): r
                for r in pending
            }
            for i, fut in enumerate(as_completed(futures), 1):
                doc = fut.result()
                sink.write(doc)
                if doc["_annotation"]["ok"]:
                    ok += 1
                else:
                    fail += 1
                    print(f"  ! {doc['url']} -> {doc['_annotation']['error']}", file=sys.stderr)
                if i % 25 == 0 or i == len(pending):
                    print(f"[annotate] {i}/{len(pending)} done  (ok={ok} fail={fail})")

    dest = "mongo" if cfg.source == "mongo" else cfg.output_path
    print(f"[annotate] Finished: ok={ok} fail={fail}  ->  {dest}")
    return 0 if fail == 0 else 1


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Annotate Know Your Meme URLs with ScrapeGraph-AI.")
    p.add_argument("--source", choices=["file", "mongo"],
                   help="Where to read URL records from (default: file / $ANNOTATION_SOURCE).")
    p.add_argument("--input", help="Input JSON/JSONL file or directory of records (file source).")
    p.add_argument("--output", help="Output JSONL path, MongoDB-ready (file source).")
    p.add_argument("--provider", help="LLM provider (openai|anthropic|google|ollama).")
    p.add_argument("--model", help="LLM model name.")
    p.add_argument("--concurrency", type=int, help="Parallel workers.")
    p.add_argument("--limit", type=int, default=0, help="Annotate at most N records (0 = all).")
    p.add_argument("--force", action="store_true", help="Re-annotate even if already done.")
    p.add_argument("--mock", action="store_true", help="Offline mock annotator (no API/network).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = AnnotationConfig()
    # CLI overrides env/defaults.
    if args.source:
        cfg.source = args.source
    if args.input:
        cfg.input_path = args.input
    if args.output:
        cfg.output_path = args.output
    if args.provider:
        cfg.provider = args.provider.lower()
        cfg.__post_init__()  # re-resolve api key for the new provider
    if args.model:
        cfg.model = args.model
    if args.concurrency:
        cfg.concurrency = args.concurrency
    return run(cfg, mock=args.mock, limit=args.limit, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
