# 🕸️ SmartScrape

**A Neuro-Symbolic Framework for Web Information Extraction**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

SmartScrape combines a **Graph Neural Network (GNN)** for node scoring with an **Integer Linear Programming (ILP)** constraint solver to extract structured information from web pages — reliably, auditably, and without breaking when page templates change.

### Two parts

This repo holds two complementary halves:

1. **Neuro-symbolic extractor demo** (GNN + ILP) — the original SmartScrape,
   extracting `title`/`price` from books.toscrape.com. See the sections below.
2. **Know Your Meme data pipeline** — discovers ~40k meme URLs, annotates them
   into information-rich records with [ScrapeGraph-AI](https://github.com/ScrapeGraphAI/Scrapegraph-ai),
   and stores them in MongoDB as the training corpus the GNN+ILP model
   (retargeted from books to memes) will learn from.

```
kym_discover.py ──▶ urls (JSON/MongoDB) ──▶ annotate_memes.py ──▶ annotations ──▶ GNN+ILP student
  sitemaps +          discovery index         ScrapeGraph-AI + LLM   rich meme       (train + serve
  listing crawl                               (teacher)             records          meme info)
```

Pipeline guides: **[docs/DISCOVERY.md](docs/DISCOVERY.md)** ·
**[docs/MEME_ANNOTATION.md](docs/MEME_ANNOTATION.md)** ·
**[docs/STUDENT_MODEL.md](docs/STUDENT_MODEL.md)**.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🧠 **GNN Inference** | 2-layer GCN encodes DOM nodes using text, visual geometry, and tag features |
| 🔒 **ILP Constraint Solver** | Enforces uniqueness, footer exclusion, product zone, and format constraints |
| 📊 **Proof-Carrying Output** | Every record includes which constraints were applied and whether any were violated |
| 📡 **Drift Detection** | Stability metric σ(P) monitors extraction confidence and triggers active learning |
| 🔬 **Ablation Study** | Toggle between ILP and Greedy modes to see the constraint benefit live |
| 🏷️ **Annotation Tool** | Built-in Streamlit labeling UI to build your own training dataset |

---

## 🏗️ Architecture

```
URL
 │
 ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────┐
│  FitLayout API  │────▶│  Feature Graph   │────▶│  GNN (GCN)  │
│ (Puppeteer/VIPS)│     │  Builder (KNN)   │     │  147-dim    │
└─────────────────┘     └──────────────────┘     └──────┬──────┘
                                                         │ scores
                                                         ▼
                                                ┌─────────────────┐
                                                │   ILP Solver    │
                                                │  (OR-Tools)     │
                                                │  Γ1 Uniqueness  │
                                                │  Γ2 Footer trap │
                                                │  Γ3 Product zone│
                                                │  Γ4 Format      │
                                                └────────┬────────┘
                                                         │
                                                         ▼
                                               ┌──────────────────┐
                                               │  Proof-Carrying  │
                                               │  Record + σ(P)   │
                                               └──────────────────┘
```

---

## 📈 Results

Evaluated on **51 annotated pages** from books.toscrape.com:

| Method | Title Acc. | Price Acc. | Both Correct | Violations |
|---|---|---|---|---|
| Greedy (no Γ) | 83% | 80% | 77% | 4 |
| **SmartScrape ILP (Γ)** | **97%** | **97%** | **93%** | **0** |

The ILP solver eliminates all constraint violations and improves accuracy by +14 pp (title) and +17 pp (price) over the greedy baseline.

---

## 🚀 Quick Start

### 1. Clone and install

```bash
git clone https://github.com/yourusername/smartscrape.git
cd smartscrape
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux / Mac
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your FitLayout token:
# FITLAYOUT_TOKEN=Bearer eyJ...
```

### 3. Run the demo

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## 🧬 Know Your Meme Pipeline (discovery → annotation)

Retargeting SmartScrape from books to **Know Your Meme**? Two stages feed the
model, joined by a MongoDB hub:

```
kym_discover.py ──> urls (JSON / MongoDB) ──> annotate_memes.py ──> annotations
   (sitemaps +                                  (ScrapeGraph-AI + LLM:
    listing crawl)                               information-rich extraction)
```

**1. Discovery** — find (nearly) every entry URL:

```bash
pip install -r requirements-discovery.txt
python kym_discover.py --mongo        # sitemaps + status-listing crawl -> MongoDB
```

**2. Annotation** — extract rich meme data (the *teacher* labeling the corpus the
GNN+ILP *student* will train on), via
[ScrapeGraph-AI](https://github.com/ScrapeGraphAI/Scrapegraph-ai):

```bash
pip install -r requirements-annotation.txt
python annotate_memes.py --mock --input data/meme_urls.sample.json --limit 5  # offline demo
python annotate_memes.py --source mongo                                        # ingest from MongoDB
```

**3. Student model** — the SmartScrape GNN+ILP extractor retargeted to memes.
Trains on the teacher's annotations, then extracts the structured info-box
fields (title/type/status/origin/year) **without an LLM** — cheap, fast, auditable:

```bash
pip install -r requirements-model.txt && playwright install chromium
python build_meme_dataset.py --annotations data/annotations.jsonl   # render + align -> labels
python train_meme_gnn.py                                            # -> meme_model.pt
python infer_meme.py --url https://knowyourmeme.com/memes/doge      # proof-carrying record
```

All stages are resumable, crash-safe, and provider-agnostic
(OpenAI/Anthropic/Google/Ollama). Guides:
**[docs/DISCOVERY.md](docs/DISCOVERY.md)** ·
**[docs/MEME_ANNOTATION.md](docs/MEME_ANNOTATION.md)** ·
**[docs/STUDENT_MODEL.md](docs/STUDENT_MODEL.md)**.

---

## 🏋️ Training Your Own Model

### Step 1 — Annotate pages

```bash
streamlit run annotate.py
```

Label each page node as `title`, `price`, or `other`. Saves to `data/labeled.json`.
Aim for at least 20-50 pages for good results.

### Step 2 — Train the GNN

```bash
python train_gnn.py
```

Output: `model.pt` — copy this to the project root.

### Step 3 — Run with trained model

```bash
streamlit run app.py
```

The header will show **Model trained: True** when the model is loaded correctly.

---

## 📁 Project Structure

```
memes_smart_scrape/
│
├── Meme data pipeline ─────────────────────────────────────────────
│   ├── kym_discover.py             # Phase A: discover KYM URLs (sitemaps + crawl)
│   ├── annotate_memes.py           # Phase B: annotate URLs -> rich records (CLI)
│   ├── build_meme_dataset.py       # Phase C: render + align annotations -> labels
│   ├── train_meme_gnn.py           # Phase C: train the student -> meme_model.pt
│   ├── infer_meme.py               # Phase C: run the student on a page (CLI)
│   ├── requirements-discovery.txt  # deps for discovery (requests, bs4, lxml, pymongo)
│   ├── requirements-annotation.txt # deps for annotation (scrapegraphai, pydantic, pymongo)
│   ├── requirements-model.txt      # deps for the student (torch, torch-geometric, ortools, playwright)
│   ├── docs/
│   │   ├── DISCOVERY.md            # discovery guide
│   │   ├── MEME_ANNOTATION.md      # annotation guide
│   │   └── STUDENT_MODEL.md        # GNN+ILP student guide
│   ├── data/
│   │   └── meme_urls.sample.json   # sample discovery records (for --mock)
│   └── src/
│       ├── db/                     # MongoDB hub shared by all phases
│       │   ├── store.py            # pure merge logic + InMemoryStore (dep-free)
│       │   └── mongo.py            # pymongo MongoStore (urls + annotations)
│       ├── annotation/
│       │   ├── meme_schema.py      # extracted fields (entry vs editorial)
│       │   ├── annotator.py        # ScrapeGraph-AI wrapper + MockAnnotator
│       │   ├── url_store.py        # file IO, resume, Mongo-ready documents
│       │   └── config.py           # env-driven, swappable LLM provider
│       ├── learning/               # student model (meme retarget)
│       │   ├── meme_config.py      # class space, paths, thresholds
│       │   ├── meme_labels.py      # teacher->student label alignment (pure)
│       │   ├── meme_encoding.py    # meme label space + graph builder
│       │   ├── meme_pipeline.py    # inference: render->GNN->priors->ILP
│       │   └── kym_render.py       # Playwright DOM -> nodes (proxy-aware)
│       └── reasoning/
│           └── meme_solver.py      # spec-driven ILP + greedy fallback
│
├── Books demo (original SmartScrape) ──────────────────────────────
│   ├── app.py                      # Streamlit demo application
│   ├── annotate.py                 # Streamlit labeling UI
│   ├── train_gnn.py                # GNN training script
│   ├── config.py                   # FitLayout / GNN configuration
│   ├── model.pt                    # trained GNN weights (not in repo)
│   ├── data/labeled_2.json         # annotated training pages
│   └── src/
│       ├── pipeline_fixed.py       # main extraction pipeline
│       ├── integration/fitlayout.py# FitLayout API client
│       ├── learning/               # gnn_model, features, graph_builder, drift_monitor
│       └── reasoning/              # solver_fixed (ILP), engine (greedy baseline)
│
└── tests/                          # offline test suite (unittest)
    ├── test_discovery.py           # URL classification, sitemap, taxonomy
    ├── test_db_store.py            # storage layer + discovery->annotation flow
    ├── test_meme_annotation.py     # schema, resume, mock pipeline
    └── test_meme_student.py        # alignment, ILP solver, inference pipeline
```

---

## 🔧 Configuration

All variables live in `.env` (copy from `.env.example`).

**Books demo (GNN + ILP)**

| Variable | Default | Description |
|---|---|---|
| `FITLAYOUT_TOKEN` | — | FitLayout API Bearer token (required) |
| `FITLAYOUT_API_URL` | `https://layout.fit.vutbr.cz/api` | FitLayout endpoint |
| `SMARTSCRAPE_MODEL_PATH` | `model.pt` | Path to trained GNN weights |
| `FOOTER_THRESHOLD` | `0.80` | Footer zone cutoff (fraction of page height) |
| `STABILITY_THRESHOLD` | `0.60` | σ(P) threshold for drift detection |

**Meme annotation pipeline**

| Variable | Default | Description |
|---|---|---|
| `ANNOTATION_LLM_PROVIDER` | `openai` | LLM provider: `openai` \| `anthropic` \| `google` \| `ollama` |
| `ANNOTATION_LLM_MODEL` | `gpt-4o-mini` | LLM model name |
| `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY`) | — | Key for the chosen provider |
| `ANNOTATION_SOURCE` | `file` | Where URLs come from: `file` \| `mongo` |
| `ANNOTATION_INPUT_PATH` | `data/meme_urls.sample.json` | Input records (file source) |
| `ANNOTATION_OUTPUT_PATH` | `data/annotations.jsonl` | Output JSONL (file source) |
| `ANNOTATION_CONCURRENCY` | `4` | Parallel annotation workers |
| `ANNOTATION_ONLY_CONFIRMED` | `true` | Only annotate `Confirmed` records |

**MongoDB hub**

| Variable | Default | Description |
|---|---|---|
| `MONGODB_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `MONGODB_DB` | `memes` | Database holding `urls` + `annotations` collections |

---

## 🎮 Demo Scenarios

Once the app is running, you can explore:

1. **Live Extraction** — paste any `books.toscrape.com` URL and watch the pipeline extract title and price with bounding-box overlay
2. **Constraint Proof** — inspect the proof object showing active constraints Γ1–Γ4
3. **Drift Simulation** — use the chaos engineering slider to inject noise and watch σ(P) drop below the threshold
4. **Ablation Study** — toggle ILP ↔ Greedy in the sidebar and see accuracy change in real time
5. **Batch Evaluation** — run the benchmark over all 51 labeled pages and view F1 scores and σ(P) distribution

---

## 🧮 Formal Constraints

The ILP solver enforces:

```prolog
Γ1 UNIQUENESS:    ∑ x[i, Price] ≤ 1,   ∑ x[i, Title] ≤ 1
Γ2 FOOTER TRAP:   y(n) > 0.8 × PageHeight  ⇒  x[n, Title] = x[n, Price] = 0
Γ3 PRODUCT ZONE:  y(n) > 500px           ⇒  x[n, Title] = x[n, Price] = 0
Γ4 FORMAT:        x[n, Price] = 1        ⇒  HasCurrency(n) ∧ IsNumeric(n)
```

---

## 🧪 Testing

The meme pipeline ships with an offline test suite — no API key, network, or
running MongoDB required:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

- `tests/test_discovery.py` — URL classification, sitemap parsing, taxonomy inference.
- `tests/test_db_store.py` — storage layer + the discovery → annotation flow (via `InMemoryStore`).
- `tests/test_meme_annotation.py` — schema normalisation, resume logic, end-to-end mock run.
- `tests/test_meme_student.py` — teacher→student alignment, ILP solver constraints, inference pipeline.

A few HTML-parsing tests auto-skip when `beautifulsoup4` isn't installed. The
annotation CLI also has a `--mock` mode (`python annotate_memes.py --mock --limit 5`)
that exercises the full pipeline with a network-free fake annotator.

---

## 📄 Citation

If you use SmartScrape in your research, please cite:

```bibtex
@inproceedings{imanov2026smartscrape,
  title     = {SmartScrape: A Neuro-Symbolic Web Information Extraction Demo},
  author    = {Imanov, Ganjali},
  booktitle = {Proceedings of the 26th International Conference on Web Engineering (ICWE)},
  year      = {2026},
  address   = {Lyon, France},
  publisher = {Springer}
}
```

---

## 🙏 Acknowledgements

This work was developed at the Faculty of Information Technology, Brno University of Technology.
Thanks to doc. Ing. Radek Burget, Ph.D. for supervision and to Prof. RNDr. Alexandr Meduna, CSc. for guidance.

The system integrates with [FitLayout](https://github.com/FitLayout/FitLayout) — a web page segmentation and analysis framework developed at BUT FIT.

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.
