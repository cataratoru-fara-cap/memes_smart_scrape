"""
Train the meme student GNN.

Reads the labeled dataset from build_meme_dataset.py and trains the same
``SmartScrapeGNN`` architecture used for books, with the meme class space
(title/type/status/origin/year/other). Output: meme_model.pt.

    python train_meme_gnn.py --data data/labeled_memes.json --epochs 60

Requires torch + torch-geometric (requirements-model.txt).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.learning.meme_config import MEME_CLASSES, MEME_MODEL_PATH, LABELED_DATASET_PATH


def _load_pages(path: str):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    pages = data.get("pages", data) if isinstance(data, dict) else data
    return [p for p in pages if p.get("nodes")]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train the meme student GNN.")
    parser.add_argument("--data", default=LABELED_DATASET_PATH)
    parser.add_argument("--output", default=MEME_MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    import torch
    import torch.nn.functional as F
    from sklearn.model_selection import train_test_split

    from src.learning import meme_encoding
    from src.learning.gnn_model import SmartScrapeGNN

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    pages = _load_pages(args.data)
    if not pages:
        print(f"No labeled pages in {args.data}. Run build_meme_dataset.py first.")
        return 1

    graphs, labels_list = [], []
    for p in pages:
        g, y = meme_encoding.page_to_graph(p["nodes"])
        graphs.append(g)
        labels_list.append(y)

    n_classes = len(MEME_CLASSES)
    total_nodes = sum(int(y.shape[0]) for y in labels_list)
    print(f"Pages: {len(graphs)} | Nodes: {total_nodes} | Classes: {n_classes} | "
          f"Feat dim: {meme_encoding.INPUT_DIM}")

    positions = list(range(len(graphs)))
    tr, val = train_test_split(positions, test_size=0.2, random_state=args.seed) \
        if len(positions) > 4 else (positions, positions)
    print(f"Train: {len(tr)} pages | Val: {len(val)} pages\n")

    # Background ('other') dominates; upweight the 5 real fields.
    class_weight = torch.tensor([5.0] * (n_classes - 1) + [1.0])

    model = SmartScrapeGNN(input_dim=meme_encoding.INPUT_DIM, hidden_dim=64, num_classes=n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    field_idx = list(range(n_classes - 1))  # all but 'other'

    def field_recall(split):
        model.eval()
        tp = {c: 0 for c in field_idx}
        fn = {c: 0 for c in field_idx}
        with torch.no_grad():
            for i in split:
                preds = model(graphs[i]).argmax(dim=1)
                y = labels_list[i]
                for c in field_idx:
                    tp[c] += int(((preds == c) & (y == c)).sum())
                    fn[c] += int(((preds != c) & (y == c)).sum())
        recalls = [tp[c] / (tp[c] + fn[c]) for c in field_idx if (tp[c] + fn[c]) > 0]
        return sum(recalls) / len(recalls) if recalls else 0.0

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for i in tr:
            optimizer.zero_grad()
            out = model(graphs[i])
            loss = F.nll_loss(out, labels_list[i], weight=class_weight)
            loss.backward()
            optimizer.step()
            total_loss += float(loss)
        if epoch % 10 == 0 or epoch == args.epochs:
            score = field_recall(val)
            print(f"Epoch {epoch:3d} | Loss {total_loss/max(1,len(tr)):.3f} | "
                  f"mean field recall {score:.3f}")
            if score >= best:
                best = score
                torch.save(model.state_dict(), args.output)

    if best < 0:  # fewer than 10 epochs path safety
        torch.save(model.state_dict(), args.output)
    print(f"\nBest mean field recall: {best:.3f}\nModel saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
