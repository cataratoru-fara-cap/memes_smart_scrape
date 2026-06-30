"""
Run the meme student model on one page and print the proof-carrying record.

    # Live (renders the page; needs Playwright + network):
    python infer_meme.py --url https://knowyourmeme.com/memes/doge

    # Offline (feed pre-rendered nodes; no browser/torch/ortools required):
    python infer_meme.py --nodes some_nodes.json

``--nodes`` expects a JSON list of {id, text, tag, bbox} dicts — handy for
debugging the GNN+ILP layers without a browser.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.learning.meme_config import MEME_MODEL_PATH
from src.learning.meme_pipeline import MemeExtractionPipeline


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Run the meme student model on a page.")
    p.add_argument("--url", help="Page URL to render and extract.")
    p.add_argument("--nodes", help="JSON file of pre-rendered nodes (offline).")
    p.add_argument("--model", default=MEME_MODEL_PATH, help="Trained meme_model.pt path.")
    p.add_argument("--no-priors", action="store_true", help="Disable heuristic priors.")
    args = p.parse_args(argv)

    if not args.url and not args.nodes:
        p.error("provide --url or --nodes")

    nodes = None
    renderer = None
    if args.nodes:
        nodes = json.loads(Path(args.nodes).read_text(encoding="utf-8"))
    else:
        from src.learning.kym_render import KymRenderer
        renderer = KymRenderer()

    pipeline = MemeExtractionPipeline(
        model_path=args.model, renderer=renderer, use_priors=not args.no_priors,
    )
    record = pipeline.run(url=args.url, nodes=nodes)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
