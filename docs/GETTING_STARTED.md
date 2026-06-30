# Getting Started: Live Annotation Run

A step-by-step guide to going from a fresh clone to a real LLM-backed
annotation run. Follow the stages in order; each one is a working checkpoint.

---

## Stage 0 — Baseline check

Run the offline test suite before touching any live service:

```bash
python -m unittest tests.test_discovery tests.test_meme_annotation tests.test_db_store -v
```

All 35 tests should pass with no warnings. If any fail, stop here and fix them
first — the live pipeline depends on the same code paths.

---

## Stage 1 — Install dependencies

Two separate requirements files mirror the two pipeline stages.

**Discovery** (crawl KYM for URLs):

```bash
pip install -r requirements-discovery.txt
```

Installs: `requests`, `beautifulsoup4`, `lxml`, `pymongo` (optional MongoDB sink).
All likely already present — the command is idempotent.

**Annotation** (LLM extraction):

```bash
pip install -r requirements-annotation.txt
playwright install chromium
```

Installs: `scrapegraphai`, `pydantic`, `python-dotenv`, `pymongo`.
`playwright install chromium` downloads the headless browser ScrapeGraph-AI uses
to render pages before passing them to the LLM. Run it once; it takes ~200 MB.

---

## Stage 2 — Configure `.env`

A [.env](.env) file was already created from the template. Open it and set
**one** API key matching your chosen provider:

```bash
# OpenAI (default, recommended for first run — gpt-4o-mini is cheap)
ANNOTATION_LLM_PROVIDER=openai
ANNOTATION_LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...

# --- OR ---

# Anthropic
ANNOTATION_LLM_PROVIDER=anthropic
ANNOTATION_LLM_MODEL=claude-haiku-4-5
ANTHROPIC_API_KEY=sk-ant-...

# --- OR ---

# Ollama (local, free — run `ollama serve` first)
ANNOTATION_LLM_PROVIDER=ollama
ANNOTATION_LLM_MODEL=llama3
# No API key needed
```

Everything else in `.env` has safe defaults. The only other value worth
reviewing before a first run:

```bash
ANNOTATION_ONLY_CONFIRMED=true   # skip Submission/Deadpool/Researching entries
ANNOTATION_CONCURRENCY=4         # parallel workers; drop to 1 if hitting rate limits
ANNOTATION_RATE_LIMIT_PER_MIN=0  # 0 = unlimited; set e.g. 20 to stay under quotas
```

---

## Stage 3 — First live run (sample file, 3 URLs)

The repo ships a 4-record sample (`data/meme_urls.sample.json`; 3 are
`Confirmed: true` so one is skipped by default). This is the cheapest possible
live test — three real LLM calls, three real page renders.

```bash
python annotate_memes.py --input data/meme_urls.sample.json --output data/annotations_sample.jsonl
```

Expected console output:

```
[annotate] source=file:data/meme_urls.sample.json (4 urls)  pending=3
[annotate] provider=openai  model=gpt-4o-mini  mock=False  concurrency=4
[annotate] 3/3 done  (ok=3 fail=0)
[annotate] Finished: ok=3 fail=0  ->  data/annotations_sample.jsonl
```

Inspect the result:

```bash
python -c "
import json, pathlib
for line in pathlib.Path('data/annotations_sample.jsonl').read_text().splitlines():
    doc = json.loads(line)
    print(doc['page_template_type'], doc['url'])
    inner = doc.get('meme') or {}
    print('  title:', inner.get('title'))
    print('  tags: ', inner.get('tags'))
    print()
"
```

If you see titles and tags extracted correctly, the pipeline is working.

---

## Stage 4 — Discover URLs (optional, if you need more than the sample)

Skip this stage if you already have a URL list. Run it to crawl KYM sitemaps
and collect the full ~40k entry index.

```bash
# Quick test — sitemaps only, no listing crawl, stops after the index:
python kym_discover.py --sitemap-only --output data/kym_urls_sitemaps.json

# Full run — sitemaps + all status-listing pages (confirmed + submissions + deadpool):
python kym_discover.py --output data/kym_urls.json
```

The full run paginates listing pages and takes several minutes. It is resumable:
re-run with the same `--output` path and it picks up where it left off (existing
confirmed entries are never downgraded).

Output is a JSON array of records:

```json
{ "url": "...", "namespace": "memes", "Confirmed": true, "lastmod": "...", "last_scraped": null }
```

Pipe that directly into the annotation pipeline:

```bash
python annotate_memes.py --input data/kym_urls.json --output data/annotations.jsonl --limit 50
```

`--limit 50` annotates only the first 50 pending records. Always start small,
inspect quality, then remove the flag for the full corpus.

---

## Stage 5 — Scale up with MongoDB (recommended for the full corpus)

For a large run (thousands of URLs) the MongoDB hub keeps discovery and
annotation in sync and makes both stages resumable without managing JSON files.

### 5a. Install MongoDB

```bash
sudo apt install -y mongodb-server-core
sudo systemctl start mongod
sudo systemctl enable mongod       # start on boot
mongosh --eval "db.runCommand({ping:1})"   # verify
```

### 5b. Populate URLs from discovery

```bash
# Sitemaps only (fast, ~40k confirmed entries):
python kym_discover.py --sitemap-only --mongo --no-file

# Or import your existing JSON file:
python kym_discover.py --resume --sitemap-only --mongo
```

### 5c. Annotate from MongoDB

```bash
# Dry-run with mock first:
python annotate_memes.py --source mongo --mock --limit 10

# Real run — reads from `urls` collection, writes to `annotations` collection:
python annotate_memes.py --source mongo --limit 100   # start small
python annotate_memes.py --source mongo               # full corpus when satisfied
```

The pipeline stamps `last_scraped` on each `urls` document after annotation,
so stopping and restarting picks up exactly where it left off.

### 5d. Query results

```bash
mongosh memes --eval "
  db.annotations.countDocuments()
  db.annotations.findOne({page_template_type:'meme'}, {url:1, 'meme.title':1, 'meme.tags':1})
"
```

---

## Cost estimate (OpenAI `gpt-4o-mini`)

| Corpus size | Approx. tokens | Approx. cost |
|---|---|---|
| 50 pages (smoke test) | ~75k | < $0.05 |
| 1,000 pages | ~1.5M | ~$0.90 |
| 40,000 pages (full) | ~60M | ~$36 |

These are rough estimates. Real cost depends on page length and model. Run with
`--limit 50` first and check your provider dashboard before committing to the
full corpus.

To reduce cost: use `--provider ollama --model llama3` (free, local, slower),
or set `ANNOTATION_ONLY_CONFIRMED=true` (default) to skip unconfirmed entries
which cuts the corpus by roughly a third.

---

## Common issues

**IP banned by KYM (`fail=N` with "IP ban?" in the error)**
Set a proxy in `.env`:
```bash
ANNOTATION_PROXY_URL=http://host:port
# ANNOTATION_PROXY_USERNAME=user   # if the proxy requires auth
# ANNOTATION_PROXY_PASSWORD=pass
```
The pipeline passes these directly to Playwright via ScrapeGraph-AI's `loader_kwargs`. Without a proxy, a banned IP will cause every page to return the ban HTML and the annotator will raise `"Annotation extracted no title — page is likely blocked … (IP ban?)"`.

**`RuntimeError: scrapegraphai is not installed`**
Run `pip install -r requirements-annotation.txt` and `playwright install chromium`.

**`RuntimeError: Provider 'openai' requires OPENAI_API_KEY`**
Open `.env` and fill in the key, or pass `--mock` for an offline run.

**`RuntimeError: pymongo is not installed`** (only with `--mongo` flag)
Run `pip install pymongo`. Not needed for file-based runs.

**`Connection refused` / MongoDB errors**
Run `sudo systemctl start mongod` and verify with `mongosh --eval "db.runCommand({ping:1})"`.

**Annotation failures (`fail=N` in output)**
Failed URLs get an error document written to output so they are not retried
endlessly. Inspect with:
```bash
python -c "
import json, pathlib
for line in pathlib.Path('data/annotations.jsonl').read_text().splitlines():
    doc = json.loads(line)
    if not doc.get('_annotation', {}).get('ok'):
        print(doc['url'], doc['_annotation'].get('error'))
"
```
Then re-run with `--force` to retry them once the underlying issue is fixed.

**Rate limit errors from the LLM provider**
Set `ANNOTATION_RATE_LIMIT_PER_MIN=20` (or whatever your tier allows) in `.env`
and drop `ANNOTATION_CONCURRENCY` to `2`.

---

## Reference

| Document | What it covers |
|---|---|
| [DISCOVERY.md](DISCOVERY.md) | `kym_discover.py` internals, flags, what was fixed |
| [MEME_ANNOTATION.md](MEME_ANNOTATION.md) | Schema details, output format, teacher→student strategy |
| [../.env](../.env) | All tuneable settings with inline comments |
