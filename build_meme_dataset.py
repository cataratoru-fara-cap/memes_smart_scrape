"""
Build the student training set: annotations -> rendered + aligned labeled pages.

For each annotation document (teacher output) this renders the page to DOM
nodes and aligns the extracted field values onto those nodes (weak supervision),
producing the labeled corpus ``train_meme_gnn.py`` consumes — the meme analogue
of the hand-labeled books pages.

    annotations.jsonl / mongo `annotations`  ──▶  data/labeled_memes.json

Resumable: URLs already present in the output are skipped. Rendering needs
Playwright (and network); set ANNOTATION_PROXY_URL if your IP is banned.

Usage:
    python build_meme_dataset.py --annotations data/annotations.jsonl
    python build_meme_dataset.py --source mongo --limit 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List

from src.learning.meme_config import LABELED_DATASET_PATH
from src.learning.meme_labels import label_page


def iter_annotation_docs(source: str, annotations_path: str) -> Iterator[Dict[str, Any]]:
    """Yield annotation documents from a JSONL file or the mongo collection."""
    if source == "mongo":
        from src.db.mongo import get_store
        store = get_store()
        try:
            yield from store.iter_annotations()
        finally:
            store.close()
        return

    p = Path(annotations_path)
    if not p.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)


def load_existing(out_path: Path) -> List[Dict[str, Any]]:
    if not out_path.exists():
        return []
    data = json.loads(out_path.read_text(encoding="utf-8"))
    return data.get("pages", data) if isinstance(data, dict) else data


def save_dataset(out_path: Path, pages: List[Dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "pages": len(pages),
            "labeled_nodes": sum(
                sum(1 for n in p["nodes"] if n.get("label", "other") != "other")
                for p in pages
            ),
        },
        "pages": pages,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    from src.learning.kym_render import KymRenderer

    out_path = Path(args.output)
    pages = load_existing(out_path)
    done = {p["url"] for p in pages}

    renderer = KymRenderer(headless=not args.headful)

    docs = list(iter_annotation_docs(args.source, args.annotations))
    todo = [
        d for d in docs
        if d.get("url") and d["url"] not in done
        and d.get("_annotation", {}).get("ok", True)
        and isinstance(d.get("meme"), dict)
    ]
    if args.limit:
        todo = todo[: args.limit]

    print(f"[dataset] annotations={len(docs)}  already_built={len(done)}  todo={len(todo)}")

    ok = fail = 0
    for i, doc in enumerate(todo, 1):
        url = doc["url"]
        try:
            nodes = renderer.render(url)
            labeled = label_page(nodes, doc["meme"])
            n_labeled = sum(1 for n in labeled if n.get("label", "other") != "other")
            pages.append({
                "url": url,
                "page_template_type": doc.get("page_template_type"),
                "nodes": labeled,
            })
            ok += 1
            print(f"  [{i}/{len(todo)}] {url}  ({len(labeled)} nodes, {n_labeled} labeled)")
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"  [{i}/{len(todo)}] FAIL {url}: {e}")
        if i % 10 == 0:
            save_dataset(out_path, pages)  # periodic checkpoint

    save_dataset(out_path, pages)
    print(f"[dataset] done: ok={ok} fail={fail}  total_pages={len(pages)}  ->  {out_path}")
    return 0 if fail == 0 else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build the meme student training set.")
    p.add_argument("--source", choices=["file", "mongo"], default="file")
    p.add_argument("--annotations", default="data/annotations.jsonl",
                   help="Annotations JSONL (file source).")
    p.add_argument("--output", default=LABELED_DATASET_PATH, help="Output labeled dataset JSON.")
    p.add_argument("--limit", type=int, default=0, help="Process at most N new pages.")
    p.add_argument("--headful", action="store_true", help="Show the browser (debug).")
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
