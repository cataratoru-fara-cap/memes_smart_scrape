# 🧬 Meme Dataset Annotation (ScrapeGraph-AI)

This document describes how SmartScrape's strategy is retargeted from
books.toscrape.com (title/price) to **knowyourmeme.com** (rich meme data),
and how the **ScrapeGraph-AI** annotation pipeline produces the training corpus.

---

## Why this pipeline exists

We have ~40,000 Know Your Meme URLs stored as records like:

```json
{
  "url": "https://knowyourmeme.com/editorials/guides/what-happened-to-chimptopia-...",
  "Confirmed": true,
  "lastmod": "2026-06-23T12:16:57-04:00",
  "page_template_type": null,
  "last_scraped": null
}
```

The end goal is an ML model that maps **page content → structured meme info**.
SmartScrape already does this for books with a *neuro-symbolic* design
(GNN scores nodes → ILP picks a globally consistent, auditable assignment).
But that model needs **ground-truth labels**, and hand-labeling 40k pages is
infeasible.

So we use a **teacher → student** strategy:

| Stage | Tool | Output |
|---|---|---|
| **1. Annotate** (teacher) | ScrapeGraph-AI + an LLM | Rich JSON per page (this pipeline) |
| **2. Train** (student) | GNN + ILP (SmartScrape) | A cheap, fast, auditable extractor |
| **3. Serve** | MongoDB + model | Query/extract meme info at scale |

The LLM is expensive but accurate; we pay it **once** to label the corpus, then
distill that knowledge into the cheap GNN+ILP model for production extraction.

---

## What gets extracted (information-rich schema)

The schema lives in [`src/annotation/meme_schema.py`](../src/annotation/meme_schema.py)
and is the single source of truth. Two families, chosen automatically from the URL:

**Entry pages** (`/memes/`, `/cultures/`, `/subcultures/`, `/people/`, `/events/`, …):
`title`, `aliases`, `entry_type`, `status`, `origin`, `year`, `region`, `tags`,
`about`, `origin_description`, `spread_description`, `notable_examples`,
`related_memes`, `parent_meme`, `search_interest`, `external_references`, `nsfw`.

**Editorial pages** (`/editorials/`):
`title`, `subtitle`, `author`, `published_date`, `category`, `summary`,
`body_excerpt`, `tags`, `mentioned_memes`, `related_articles`.

To add/change a field, edit the spec lists in `meme_schema.py` — the prompt,
the pydantic schema, and normalisation all derive from it automatically.

---

## Usage

### 1. Install

```bash
pip install -r requirements-annotation.txt
playwright install chromium      # ScrapeGraph-AI renders pages with Playwright
cp .env.example .env             # then set OPENAI_API_KEY (or another provider)
```

### 2. Offline smoke test (no key, no network)

```bash
python annotate_memes.py --mock --input data/meme_urls.sample.json --limit 5
```

### 3. Real run

```bash
python annotate_memes.py \
  --input  data/meme_urls.json \
  --output data/annotations.jsonl \
  --concurrency 4
```

### 3b. Ingest from MongoDB (recommended at scale)

When discovery has populated MongoDB (`kym_discover.py --mongo`, see
[DISCOVERY.md](DISCOVERY.md)), annotate straight from it — no JSON file needed:

```bash
python annotate_memes.py --source mongo
```

This reads pending records (`Confirmed: true`, `last_scraped: null`) from the
`urls` collection, writes extracted docs to the `annotations` collection, and
stamps `last_scraped` back on `urls`. Both stages stay in sync and resumable.

Swap the LLM provider/model from the CLI or `.env`:

```bash
python annotate_memes.py --provider anthropic --model claude-haiku-4-5
python annotate_memes.py --provider ollama   --model llama3        # local, free
```

### Key flags

| Flag | Meaning |
|---|---|
| `--mock` | Offline fake annotator (tests/dev) |
| `--limit N` | Annotate at most N pending records |
| `--force` | Re-annotate even already-done URLs |
| `--concurrency N` | Parallel workers |

---

## Resumability & scale (40k pages)

The pipeline is **append-only and resumable** — safe to stop/restart:

- A record is **skipped** if its URL is already in the output JSONL, or it
  carries a `last_scraped` timestamp (the resume marker in your source data).
- Only `Confirmed: true` records are processed by default
  (`ANNOTATION_ONLY_CONFIRMED`).
- Output is flushed + `fsync`'d per line, so a crash never corrupts prior work.
- Failures are retried with exponential backoff; if they still fail an **error
  document** is written (so the URL isn't retried forever) with the reason in
  `_annotation.error`.
- Use `ANNOTATION_RATE_LIMIT_PER_MIN` to stay under provider rate limits.

**Cost control tip:** start with `--limit 50` on a sample, inspect the output
quality, tune the schema/prompt, *then* run the full corpus.

---

## Output: MongoDB-ready documents

Each line of `annotations.jsonl` is one document:

```json
{
  "_id": "<sha1(url)>",
  "url": "...",
  "confirmed": true,
  "lastmod": "...",
  "page_template_type": "meme",
  "last_scraped": "2026-06-29T13:16:22+00:00",
  "meme": { ...the extracted schema... },
  "_annotation": { "provider": "openai", "model": "gpt-4o-mini", "ok": true, "error": null }
}
```

Load it:

```bash
mongoimport --db memes --collection entries --file data/annotations.jsonl
```

`_id` is a stable hash of the URL, so re-imports upsert cleanly instead of
duplicating.

---

## Next step: training the student model

Once `annotations.jsonl` exists, the SmartScrape GNN+ILP stack is retargeted by:

1. Replacing the `["price", "title", "other"]` field set with the meme schema
   fields as extraction targets.
2. Aligning rendered page nodes (FitLayout / Playwright DOM) to the LLM-extracted
   values to produce per-node labels — the analogue of `data/labeled_2.json`.
3. Reusing `train_gnn.py` / `src/learning/*` to train, and `src/reasoning/*`
   (ILP) to enforce meme-specific constraints (e.g. exactly one `title`, `year`
   must be a 4-digit number, `status ∈ {Confirmed, Submission, Deadpool}`).

That stage is intentionally **not** built yet — it depends on this annotated
corpus existing first.
