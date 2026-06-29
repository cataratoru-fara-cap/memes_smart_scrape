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


def run(cfg: AnnotationConfig, *, mock: bool, limit: int, force: bool) -> int:
    records = url_store.load_url_records(cfg.input_path)
    writer = url_store.AnnotationWriter(cfg.output_path)
    done = writer.existing_urls()

    pending: List[Dict[str, Any]] = list(
        url_store.iter_pending(records, done, only_confirmed=cfg.only_confirmed, force=force)
    )
    if limit and limit > 0:
        pending = pending[:limit]

    print(f"[annotate] input={cfg.input_path}  total={len(records)}  "
          f"already_done={len(done)}  pending={len(pending)}")
    print(f"[annotate] provider={cfg.provider}  model={cfg.model}  "
          f"mock={mock}  concurrency={cfg.concurrency}")
    if not pending:
        print("[annotate] Nothing to do.")
        return 0

    annotator = build_annotator(cfg, mock=mock)
    limiter = _RateLimiter(cfg.rate_limit_per_min)

    ok = fail = 0
    with writer:
        with ThreadPoolExecutor(max_workers=max(1, cfg.concurrency)) as pool:
            futures = {
                pool.submit(_annotate_one, annotator, r, cfg, limiter): r
                for r in pending
            }
            for i, fut in enumerate(as_completed(futures), 1):
                doc = fut.result()
                writer.write(doc)
                if doc["_annotation"]["ok"]:
                    ok += 1
                else:
                    fail += 1
                    print(f"  ! {doc['url']} -> {doc['_annotation']['error']}", file=sys.stderr)
                if i % 25 == 0 or i == len(pending):
                    print(f"[annotate] {i}/{len(pending)} done  (ok={ok} fail={fail})")

    print(f"[annotate] Finished: ok={ok} fail={fail}  ->  {cfg.output_path}")
    return 0 if fail == 0 else 1


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Annotate Know Your Meme URLs with ScrapeGraph-AI.")
    p.add_argument("--input", help="Input JSON/JSONL file or directory of records.")
    p.add_argument("--output", help="Output JSONL path (MongoDB-ready).")
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
