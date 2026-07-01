# 🧬 Meme Dataset Annotation (ScrapeGraph-AI)

This document describes how the **ScrapeGraph-AI** annotation pipeline turns
Know Your Meme URLs into information-rich records — the training corpus for the
GNN+ILP student.

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

### Self-hosted Open WebUI / Ollama gateway

Point the `openai` provider at the gateway's OpenAI-compatible surface via
`base_url`. The pipeline calls `POST {base_url}/chat/completions` with
`Authorization: Bearer $OPENAI_API_KEY`.

> On the lab's Open WebUI gateway the `/openai/*` routes are **disabled**
> (`{"detail":"OpenAI API is disabled"}`); use the Ollama gateway's own
> OpenAI-compatible endpoints under **`/ollama/v1`** instead.

```bash
# .env
ANNOTATION_LLM_PROVIDER=openai
ANNOTATION_LLM_MODEL=llama3.3:70b               # id from GET /ollama/v1/models
ANNOTATION_LLM_BASE_URL=https://<lab-host>/ollama/v1
OPENAI_API_KEY=<gateway Bearer token>
ANNOTATION_STRUCTURED_OUTPUT=false              # Ollama models: prompt-only JSON
```

Check connectivity and that your model name resolves before a real run:

```bash
python annotate_memes.py --verify
```

**Choosing a model.** For rich JSON extraction prefer a strong *instruction*
model with a long context: `llama3.3:70b` (best here), or lighter
`qwen2.5:32b` / `mistral-small3.2:24b`. Avoid **reasoning** models
(`deepseek-r1:*`, the `qwen3:*` family) — they emit `<think>…</think>` traces
that corrupt JSON output. Skip the `*-embed*` / `llava` (vision) / `codellama`
entries; this task needs a text chat model. Our node encoder uses a char-n-gram
hash, so the embedding models aren't used at all.

> **Structured output.** By default the annotator sends a JSON schema
> (function-calling / json_schema); Ollama models generally don't support that
> and will error, so set `ANNOTATION_STRUCTURED_OUTPUT=false` to fall back to
> prompt-only JSON — the prompt already requires a strict JSON object and
> `meme_schema.normalize()` coerces it. Local models also extract less reliably
> than hosted `gpt-4o-mini`, so spot-check a `--limit 20` run before scaling up.

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

## Next step: the student model

Once `annotations.jsonl` (or the MongoDB `annotations` collection) exists, the
GNN+ILP **student** distills these labels into a cheap, LLM-free extractor:

```bash
python build_meme_dataset.py --annotations data/annotations.jsonl   # render + align -> labels
python train_meme_gnn.py                                            # -> meme_model.pt
python infer_meme.py --url https://knowyourmeme.com/memes/doge      # proof-carrying record
```

It renders each annotated page to DOM nodes, aligns the extracted values onto
those nodes (weak supervision), trains the GNN, and enforces meme constraints in
the ILP (e.g. one `title`, `year` is a 4-digit number, `status` is a valid KYM
status). Full details in **[STUDENT_MODEL.md](STUDENT_MODEL.md)**.
