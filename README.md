# 🧠 MemeSmartScrape

**Neuro-symbolic information extraction for Know Your Meme.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

MemeSmartScrape builds a structured, machine-learning-ready dataset of memes from
[knowyourmeme.com](https://knowyourmeme.com). It is a reimplementation of the
SmartScrape neuro-symbolic approach — a **Graph Neural Network (GNN)** that scores
page nodes combined with an **Integer Linear Programming (ILP)** constraint solver
— applied to memes instead of e-commerce pages: reliable, auditable, and robust to
template changes.

## How it works

Three stages, joined by a MongoDB hub:

```
kym_discover.py ──▶ urls (JSON/MongoDB) ──▶ annotate_memes.py ──▶ annotations ──▶ GNN+ILP student
  discovery:          discovery index        teacher:               rich meme       student:
  sitemaps +                                  ScrapeGraph-AI + LLM    records         train + serve
  listing crawl                              (information-rich)                       meme info, no LLM
```

1. **Discovery** — crawl KYM (sitemaps + status listings) for ~40k entry URLs.
2. **Annotation (teacher)** — an LLM, via
   [ScrapeGraph-AI](https://github.com/ScrapeGraphAI/Scrapegraph-ai), extracts
   information-rich records. Expensive but accurate; run once to label the corpus.
3. **Student model** — the GNN+ILP extractor distills the teacher's labels into a
   cheap, fast, auditable model that recovers the structured fields **without an
   LLM**. Every output is *proof-carrying*: it records which constraints were
   applied, plus a stability score σ(P).

Why neuro-symbolic? The GNN learns *which node looks like* each field; the ILP
then enforces hard rules (one title, `year` is a 4-digit number, `status` is a
valid KYM status…) so outputs stay consistent and explainable rather than relying
on the network alone.

Stage guides: **[docs/DISCOVERY.md](docs/DISCOVERY.md)** ·
**[docs/MEME_ANNOTATION.md](docs/MEME_ANNOTATION.md)** ·
**[docs/STUDENT_MODEL.md](docs/STUDENT_MODEL.md)** ·
**[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)**.

---

## 🚀 Quick start

```bash
cp .env.example .env          # set an LLM key (OPENAI_API_KEY) for real runs

# Try the pipeline offline — no key, no network:
python annotate_memes.py --mock --input data/meme_urls.sample.json --limit 5
python -m unittest discover -s tests -p "test_*.py"
```

**1. Discovery** — build the URL index:

```bash
pip install -r requirements-discovery.txt
python kym_discover.py --mongo          # sitemaps + status-listing crawl -> MongoDB
```

**2. Annotation (teacher)** — extract rich records:

```bash
pip install -r requirements-annotation.txt && playwright install chromium
python annotate_memes.py --source mongo                 # ingest from MongoDB
# or file-based:  python annotate_memes.py --input data/kym_urls.json
```

**3. Student model** — train and serve the GNN+ILP extractor:

```bash
pip install -r requirements-model.txt && playwright install chromium
python build_meme_dataset.py --annotations data/annotations.jsonl   # render + align -> labels
python train_meme_gnn.py                                            # -> meme_model.pt
python infer_meme.py --url https://knowyourmeme.com/memes/doge      # proof-carrying record
```

All stages are resumable, crash-safe, and provider-agnostic
(OpenAI / Anthropic / Google / Ollama). New here? Follow
**[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)** end to end.

---

## 🧬 What the student extracts

Per-node classification over the meme fields, with ILP constraints:

| Tier | Fields | Mechanism |
|---|---|---|
| **Singletons** | title · type · status · origin · year · parent_meme | GNN class + ILP uniqueness/format |
| **Multi-label** | tags · aliases · region · related_memes | GNN class, no uniqueness (aggregated to lists) |
| **Sections** | about · origin_description · spread_description | heuristic heading-anchored extractor (no GNN/LLM) |

The teacher additionally captures `search_interest`, `notable_examples`,
`external_references`, `nsfw` (not node-localizable — see
[docs/STUDENT_MODEL.md](docs/STUDENT_MODEL.md)).

---

## 📁 Project structure

```
memes_smart_scrape/
├── kym_discover.py             # stage 1: discover KYM URLs (sitemaps + crawl)
├── annotate_memes.py           # stage 2: LLM annotation -> rich records (CLI)
├── build_meme_dataset.py       # stage 3: render + align annotations -> labels
├── train_meme_gnn.py           # stage 3: train the student -> meme_model.pt
├── infer_meme.py               # stage 3: run the student on a page (CLI)
├── requirements-discovery.txt  # requests, bs4, lxml, pymongo
├── requirements-annotation.txt # scrapegraphai, pydantic, pymongo
├── requirements-model.txt      # torch, torch-geometric, ortools, playwright
├── data/meme_urls.sample.json  # sample discovery records (for --mock)
├── docs/                       # DISCOVERY · MEME_ANNOTATION · STUDENT_MODEL · GETTING_STARTED
├── tests/                      # offline unittest suite (stdlib only)
└── src/
    ├── db/                     # MongoDB hub shared by all stages
    │   ├── store.py            #   pure merge logic + InMemoryStore (dep-free)
    │   └── mongo.py            #   pymongo MongoStore (urls + annotations)
    ├── net/
    │   └── proxy_pool.py       # rotating free-proxy pool (proxifly) + failover
    ├── annotation/             # teacher (ScrapeGraph-AI)
    │   ├── meme_schema.py      #   extracted fields (entry vs editorial)
    │   ├── annotator.py        #   ScrapeGraph-AI wrapper + MockAnnotator
    │   ├── url_store.py        #   file IO, resume, Mongo-ready documents
    │   └── config.py           #   env-driven, swappable LLM provider
    ├── learning/               # student model
    │   ├── meme_config.py      #   class space (singleton/multi), paths, thresholds
    │   ├── meme_labels.py      #   teacher->student label alignment (pure)
    │   ├── meme_encoding.py    #   meme label space + graph builder
    │   ├── meme_sections.py    #   section extractor (about/origin/spread)
    │   ├── meme_pipeline.py    #   inference: render->GNN->priors->ILP->sections
    │   ├── kym_render.py       #   Playwright DOM -> nodes (proxy-aware)
    │   ├── encoding.py         #   shared node features + graph edges
    │   └── gnn_model.py        #   2-layer GCN node classifier
    └── reasoning/
        └── meme_solver.py      #   spec-driven ILP + greedy fallback
```

---

## 🔧 Configuration

All settings live in `.env` (copy from `.env.example`).

**Annotation (teacher)**

| Variable | Default | Description |
|---|---|---|
| `ANNOTATION_LLM_PROVIDER` | `openai` | `openai` \| `anthropic` \| `google` \| `ollama` |
| `ANNOTATION_LLM_MODEL` | `gpt-4o-mini` | LLM model name |
| `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY`) | — | Key for the chosen provider |
| `ANNOTATION_SOURCE` | `file` | Where URLs come from: `file` \| `mongo` |
| `ANNOTATION_CONCURRENCY` | `4` | Parallel annotation workers |
| `ANNOTATION_ONLY_CONFIRMED` | `true` | Only annotate `Confirmed` records |
| `ANNOTATION_PROXY_URL` | — | Proxy for annotation **and** rendering (use if IP-banned) |

**MongoDB hub**

| Variable | Default | Description |
|---|---|---|
| `MONGODB_URI` | `mongodb://localhost:27017` | Connection string |
| `MONGODB_DB` | `memes` | DB holding the `urls` + `annotations` collections |

**Student model**

| Variable | Default | Description |
|---|---|---|
| `MEME_MODEL_PATH` | `meme_model.pt` | Trained GNN weights |
| `MEME_LABELED_PATH` | `data/labeled_memes.json` | Aligned training set |
| `MEME_STABILITY_THRESHOLD` | `0.15` | Drift alert when mean σ(P) margin falls below this |

---

## 🧪 Testing

The full suite is offline — no API key, network, or running MongoDB:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

- `tests/test_discovery.py` — URL classification, sitemap parsing, taxonomy inference.
- `tests/test_db_store.py` — storage layer + the discovery → annotation flow (via `InMemoryStore`).
- `tests/test_meme_annotation.py` — schema normalisation, resume logic, end-to-end mock run.
- `tests/test_meme_student.py` — teacher→student alignment, ILP solver constraints, inference pipeline.

Heavy dependencies (torch, ortools, scrapegraphai, playwright, pymongo) are
imported lazily, so the tests run on stdlib alone; a few HTML-parsing tests
auto-skip when `beautifulsoup4` isn't installed.

---

## 🙏 Credits

The neuro-symbolic GNN + ILP design follows the **SmartScrape** approach to web
information extraction; this project reimplements and adapts it for Know Your Meme.

## 📜 License

MIT License — see [LICENSE](LICENSE).
