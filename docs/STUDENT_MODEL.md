# 🎓 Meme GNN+ILP Student Model

The **student** is a neuro-symbolic extractor (GNN node scoring + ILP
constraints, following the SmartScrape approach) for Know Your Meme. It learns
from the **teacher** (ScrapeGraph-AI annotations) so that, once trained, it
extracts the structured info-box fields **without an LLM** — cheaply, fast, and
auditably.

```
annotations (teacher)                       trained student
        │                                         ▲
        ▼                                         │
build_meme_dataset.py ──▶ labeled_memes.json ──▶ train_meme_gnn.py ──▶ meme_model.pt
   render + align            per-node labels         GNN training            │
                                                                             ▼
                                              infer_meme.py / MemeExtractionPipeline
                                              URL ▶ render ▶ GNN ▶ priors ▶ ILP ▶ record
```

---

## What it extracts

The student covers three tiers of fields, by how each maps onto the page:

**① Singletons — one node each (GNN class + ILP uniqueness):**

```
title · type · status · origin · year · parent_meme
```

**② Multi-label — many nodes each (GNN class, no uniqueness):**

```
tag → tags · alias → aliases · region · related → related_memes
```

Each node the GNN tags with a multi class becomes one item; the served record
aggregates them into a list (e.g. all `tag` nodes → `tags: [...]`).

**③ Sections — long-form prose (heuristic, no GNN/LLM):**

```
about · origin_description · spread_description
```

These are whole sections (a run of paragraphs under a heading), so a separate
heading-anchored extractor recovers them — see `src/learning/meme_sections.py`.

The per-node class space is therefore:

```
title type status origin year parent_meme   (singletons)
tag alias region related                     (multi-label)
other                                        (background)
```

Still teacher-only (not node-localizable): `search_interest`, `notable_examples`,
`external_references`, `nsfw`.

The same `SmartScrapeGNN` architecture (2-layer GCN, `src/learning/gnn_model.py`)
and feature/edge encoder (`src/learning/encoding.py`) are reused — only the class
space and the constraints change.

---

## The ILP constraints (`src/reasoning/meme_solver.py`)

Maximise the GNN's per-node scores subject to:

| Constraint | Rule |
|---|---|
| **Integrity** | each node gets exactly one class |
| **Uniqueness** | ≤ 1 node per **singleton** field (title/type/status/origin/year/parent_meme); multi-label fields have no cap |
| **Format — year** | a node can be `year` only if its text contains a 4-digit year |
| **Format — status** | a node can be `status` only if its text is a KYM status word (`confirmed`/`submission`/`deadpool`/`researching`) |
| **Title zone** *(optional)* | title must sit above `MEME_TITLE_ZONE_MAX_PX` |

OR-Tools (SCIP) solves it; on infeasibility the zone constraint is relaxed, then
a pure-Python **greedy fallback** runs. The fallback needs no third-party
packages, so the solver works (degraded) even before you install OR-Tools.

---

## How the teacher labels the student (`src/learning/meme_labels.py`)

Hand-labeling 40k pages is infeasible, so labels are generated automatically:

1. `localizable_field_values()` pulls the singleton values and
   `multi_field_values()` pulls the list fields (tags/aliases/region/related)
   from an annotation's `meme` payload.
2. `align_nodes_to_labels()` assigns each value to its best-matching node (exact
   / tight-containment / token overlap; `year` matches on the 4-digit value).
   A greedy global assignment caps singletons at one node each, while a multi
   class may claim several nodes (one per distinct value).

This is weak supervision / distillation — the meme analogue of the hand-labeled
books pages.

---

## Usage

```bash
pip install -r requirements-model.txt
playwright install chromium      # for live rendering
```

**1. Build the training set** (needs annotations from the teacher; renders pages):

```bash
python build_meme_dataset.py --annotations data/annotations.jsonl
# or from MongoDB:
python build_meme_dataset.py --source mongo --limit 200
```

Output: `data/labeled_memes.json` (`{url, page_template_type, nodes:[{…,label}]}`).
Resumable — re-runs skip URLs already built.

**2. Train**:

```bash
python train_meme_gnn.py --data data/labeled_memes.json --epochs 60
```

Output: `meme_model.pt`. Background dominates, so the five real fields are
up-weighted; the script reports mean per-field recall on a held-out split.

**3. Infer**:

```bash
# Live (render + extract):
python infer_meme.py --url https://knowyourmeme.com/memes/doge

# Offline (debug the GNN+ILP on pre-rendered nodes — no browser/torch/ortools):
python infer_meme.py --nodes nodes.json
```

Output is a proof-carrying record: each field has its `text`, `bbox`,
`confidence`, `node_id`, plus a `_solver` block (backend, constraints, relaxed,
fallback) and `_meta` (model_trained, `stability_score` = σ(P), drift_alert).

---

## Graceful degradation (test before you can render)

Everything heavy is lazy, so you can validate the plumbing incrementally:

| Missing | Behaviour |
|---|---|
| trained model / torch | uniform GNN scores; **priors + ILP still produce a result** |
| OR-Tools | solver uses its greedy fallback |
| Playwright | pass `--nodes` to run on pre-rendered nodes |

The offline test suite (`tests/test_meme_student.py`) covers value extraction,
node alignment, the solver's format/uniqueness rules, and an end-to-end pipeline
run with uniform scores — all with no third-party deps:

```bash
python -m unittest tests.test_meme_student -v
```

---

## Files

| Path | Role |
|---|---|
| `src/learning/meme_config.py` | class space (singleton/multi), paths, thresholds |
| `src/learning/meme_labels.py` | singleton + multi value extraction & node alignment (pure) |
| `src/learning/meme_encoding.py` | meme label space + graph builder |
| `src/learning/meme_sections.py` | tier ③ heading-anchored section extractor (pure) |
| `src/learning/kym_render.py` | Playwright DOM → nodes (proxy-aware) |
| `src/learning/meme_pipeline.py` | inference pipeline (render→GNN→priors→ILP→sections) |
| `src/reasoning/meme_solver.py` | spec-driven ILP (singleton uniqueness + multi) + greedy fallback |
| `build_meme_dataset.py` · `train_meme_gnn.py` · `infer_meme.py` | CLIs |
