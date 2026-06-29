# 🔎 URL Discovery (`kym_discover.py`)

`kym_discover.py` builds the master index of Know Your Meme entry URLs — the
input the annotation pipeline consumes. It can write a JSON file, MongoDB, or
both.

```
sitemaps + listing crawl ──> kym_discover.py ──> urls (JSON file and/or MongoDB)
                                                      │
                                                      ▼
                                              annotate_memes.py
```

---

## How it works

| Phase | What it does |
|---|---|
| **1. Sitemaps** | Reads `robots.txt`, walks every child sitemap (gzip aware). These hold the **confirmed** entries with `lastmod`. |
| **1b. Taxonomy** | Infers namespace prefixes + listing pages from the corpus so new KYM namespaces are picked up automatically. |
| **2. Listing crawl** | Paginates the status listings to catch what sitemaps omit. |

---

## What was fixed (the missing ~3% / ~10%)

The previous version under-collected for three reasons, all addressed here:

1. **Crash / broken pagination.** A stray `from sqlalchemy import null` made the
   script un-runnable, and `--max-category-pages` was compared as a string
   (`page <= "5"` → `TypeError`). The `/memes/all` page template was malformed
   (`/page/{n}?page=1`). → Removed the bad import, `--max-category-pages` is now
   `int`, and pagination **follows the page's own next-link** instead of guessing
   a URL format.

2. **Pagination stopped early (~3% of memes).** The advertised "stop after 3
   empty pages" was dead code, so the crawl relied entirely on a fragile
   next-link CSS selector and quit at the first miss. → `CONSECUTIVE_EMPTY_LIMIT`
   is now actually enforced as a backstop, and next-page detection tries several
   selectors plus `<link rel=next>`.

3. **Whole listings + sub-namespaces skipped (~10% of entries).** Only
   `/memes/all` was crawled, and `extract_entry_links` required a specific CSS
   class **and** an exact namespace match — dropping `/memes/events/…` style
   entries and never visiting deadpool/submission/researching listings. →
   Entry links are now detected by **URL shape** (`is_entry_url`, CSS-agnostic),
   sub-namespace entries are kept, and the crawl visits
   `/memes/all`, `/memes/confirmed`, `/memes/submissions`, `/memes/researching`,
   `/memes/deadpool` (plus any status listings discovered in the corpus).

`Confirmed` is treated as **monotonic** — the sitemap marks confirmed entries
`True`, and the crawl can add non-confirmed ones without ever downgrading them
(`src/db/store.merge_discovery`).

---

## Usage

```bash
pip install -r requirements-discovery.txt

python kym_discover.py                       # full run -> kym_urls.json
python kym_discover.py --sitemap-only        # sitemaps only
python kym_discover.py --max-category-pages 5  # cap pagination (testing)
python kym_discover.py --mongo               # also upsert into MongoDB
python kym_discover.py --mongo --no-file     # MongoDB only
python kym_discover.py --fresh               # ignore existing file
```

Record shape (unchanged, so the annotation pipeline ingests it directly):

```json
{
  "url": "https://knowyourmeme.com/memes/doge",
  "namespace": "memes",
  "Confirmed": true,
  "lastmod": "2026-06-23T12:16:57-04:00",
  "page_template_type": null,
  "last_scraped": null
}
```

---

## MongoDB hub

`--mongo` upserts every record into the **`urls`** collection
(`_id = sha1(url)`), via the shared store in [`src/db`](../src/db). Re-runs are
idempotent: discovery never overwrites a record's `last_scraped` and never
downgrades `Confirmed`.

Migrate the existing big JSON in one shot:

```bash
python kym_discover.py --resume --sitemap-only --mongo   # or load the file then --mongo
```

The annotation pipeline then ingests straight from Mongo:

```bash
python annotate_memes.py --source mongo
```

It reads pending records (`Confirmed: true`, `last_scraped: null`) from `urls`,
writes results to the **`annotations`** collection, and stamps `last_scraped`
back on the `urls` doc so the two stages stay in sync and both are resumable.

Connection is configured by `MONGODB_URI` / `MONGODB_DB` (see `.env.example`).
