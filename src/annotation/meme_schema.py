"""
Meme annotation schema for Know Your Meme pages.

This module is the single source of truth for *what* we extract from each page.
It is intentionally dependency-free (pure stdlib) so the orchestrator, the
URL store and the tests can import it without pulling in pydantic / scrapegraphai.

Two page families exist on knowyourmeme.com:

  * "entry"     — a meme/culture/subculture/event/person/site entry
                  (URLs under /memes/, /cultures/, /subcultures/, /people/, ...)
  * "editorial" — guides / news / editorials (URLs under /editorials/)

Each family has its own field spec. A spec is an ordered list of
(field_name, description) pairs. The descriptions are fed verbatim to the LLM
(so they double as extraction instructions) and are also used to build the
optional pydantic schema for structured output.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

# --------------------------------------------------------------------------
# Page template detection
# --------------------------------------------------------------------------

# Map of first-path-segment -> canonical template type.
# Know Your Meme groups every page under one of these top-level sections.
_SECTION_TO_TEMPLATE = {
    "memes": "meme",
    "cultures": "culture",
    "subcultures": "subculture",
    "people": "person",
    "sites": "site",
    "events": "event",
    "types": "type",
    "editorials": "editorial",
    "photos": "photo",
    "videos": "video",
}

# Which extraction spec each template type uses.
_EDITORIAL_TEMPLATES = {"editorial"}


def detect_template_type(url: str) -> str:
    """
    Infer the page template type from the URL path.

    Returns one of the values in ``_SECTION_TO_TEMPLATE`` (e.g. "meme",
    "editorial", "culture") or "unknown" when the section is not recognised.
    """
    path = urlparse(url).path.strip("/")
    if not path:
        return "unknown"
    first = path.split("/", 1)[0].lower()
    return _SECTION_TO_TEMPLATE.get(first, "unknown")


def spec_family(template_type: str) -> str:
    """Return "editorial" or "entry" — the field spec family for a template."""
    return "editorial" if template_type in _EDITORIAL_TEMPLATES else "entry"


# --------------------------------------------------------------------------
# Field specifications
# --------------------------------------------------------------------------
# A spec entry is (name, kind, description) where kind is "str" | "int" |
# "bool" | "list[str]" | "list[obj]". `kind` drives normalisation and the
# pydantic schema; `description` is the LLM instruction for that field.

ENTRY_FIELDS: List[Tuple[str, str, str]] = [
    ("title", "str",
     "The canonical name/title of the meme or entry, exactly as shown in the page heading."),
    ("aliases", "list[str]",
     "Other names, spellings or AKAs this meme is known by. Empty list if none."),
    ("entry_type", "list[str]",
     "The 'Type' classification(s) shown in the info box, e.g. 'Image Macro', "
     "'Catchphrase', 'Exploitable', 'Copypasta', 'Reaction', 'Snowclone', "
     "'Viral Video', 'Slang'. Empty list if not stated."),
    ("status", "str",
     "The moderation status from the info box: 'Confirmed', 'Submission', "
     "'Deadpool' or 'Researching'. Empty string if not shown."),
    ("origin", "str",
     "Where the meme originated — the platform/site/community named in the info "
     "box 'Origin' field (e.g. 'Reddit', '4chan /b/', 'TikTok', 'Twitter')."),
    ("year", "int",
     "The year the meme originated (info box 'Year'). Use 0 if unknown."),
    ("region", "list[str]",
     "Geographic region(s) of origin from the info box. Empty list if none."),
    ("tags", "list[str]",
     "All tags listed for the entry. Empty list if none."),
    ("about", "str",
     "A concise 1-3 sentence summary of what the meme is (from the 'About' section)."),
    ("origin_description", "str",
     "The full text of the 'Origin' section describing how the meme started."),
    ("spread_description", "str",
     "The full text of the 'Spread' / 'Development' section describing how it grew."),
    ("notable_examples", "list[obj]",
     "Notable examples/variations. Each item: {caption, source_url}. "
     "Capture captions/descriptions of example images or videos. Empty list if none."),
    ("related_memes", "list[str]",
     "Titles of related or derivative memes referenced on the page. Empty list if none."),
    ("parent_meme", "str",
     "Title of the parent meme this is a sub-entry of, if any. Empty string otherwise."),
    ("search_interest", "str",
     "A short summary of the 'Search Interest' / popularity over time, if present."),
    ("external_references", "list[obj]",
     "External references / sources. Each item: {title, url}. "
     "Include Wikipedia, news articles, etc. Empty list if none."),
    ("nsfw", "bool",
     "True if the page is flagged NSFW / adult content, otherwise false."),
]

EDITORIAL_FIELDS: List[Tuple[str, str, str]] = [
    ("title", "str", "The headline/title of the article."),
    ("subtitle", "str", "The subtitle or deck, if present. Empty string otherwise."),
    ("author", "str", "The author/byline of the article. Empty string if none."),
    ("published_date", "str",
     "Publication date in ISO 8601 (YYYY-MM-DD) if determinable, else as shown."),
    ("category", "str",
     "Editorial category, e.g. 'Guide', 'News', 'Editorial', 'Explainer'."),
    ("summary", "str",
     "A concise 2-4 sentence summary of the article's content."),
    ("body_excerpt", "str",
     "The first ~500 characters of the article body (lede / opening paragraphs)."),
    ("tags", "list[str]", "Article tags/topics. Empty list if none."),
    ("mentioned_memes", "list[str]",
     "Titles of memes/entries discussed or referenced in the article. Empty list if none."),
    ("related_articles", "list[obj]",
     "Related articles. Each item: {title, url}. Empty list if none."),
]


def get_field_spec(template_type: str) -> List[Tuple[str, str, str]]:
    """Return the field spec list for the given template type."""
    return EDITORIAL_FIELDS if spec_family(template_type) == "editorial" else ENTRY_FIELDS


# --------------------------------------------------------------------------
# Defaults / normalisation
# --------------------------------------------------------------------------

def _default_for(kind: str) -> Any:
    if kind == "int":
        return 0
    if kind == "bool":
        return False
    if kind in ("list[str]", "list[obj]"):
        return []
    return ""


def empty_record(template_type: str) -> Dict[str, Any]:
    """Return a dict with every field for the template set to its default."""
    return {name: _default_for(kind) for name, kind, _ in get_field_spec(template_type)}


def _coerce(value: Any, kind: str) -> Any:
    """Coerce a raw LLM value into the canonical type for ``kind``."""
    if value is None:
        return _default_for(kind)
    if kind == "int":
        try:
            return int(str(value).strip()[:4]) if str(value).strip() else 0
        except (ValueError, TypeError):
            return 0
    if kind == "bool":
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("true", "yes", "1", "nsfw")
    if kind == "str":
        return value if isinstance(value, str) else str(value)
    if kind == "list[str]":
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        # tolerate a comma-separated string
        return [s.strip() for s in str(value).split(",") if s.strip()]
    if kind == "list[obj]":
        if isinstance(value, list):
            return [v for v in value if isinstance(v, dict)]
        return []
    return value


def normalize(record: Dict[str, Any], template_type: str) -> Dict[str, Any]:
    """
    Coerce an arbitrary LLM output dict into the canonical schema for the
    template: every expected field present, correctly typed, extras dropped.
    """
    spec = get_field_spec(template_type)
    out = empty_record(template_type)
    if isinstance(record, dict):
        for name, kind, _ in spec:
            if name in record:
                out[name] = _coerce(record[name], kind)
    return out


# --------------------------------------------------------------------------
# Prompt + pydantic schema builders (used by the annotator)
# --------------------------------------------------------------------------

def build_prompt(template_type: str) -> str:
    """
    Build the natural-language extraction instruction handed to ScrapeGraph-AI.

    The field descriptions are embedded so the LLM knows exactly what each key
    should contain. We ask for strict JSON keyed by the field names.
    """
    family = "editorial article" if spec_family(template_type) == "editorial" else "Know Your Meme entry"
    spec = get_field_spec(template_type)
    lines = [
        f"You are extracting structured data from a {family} on knowyourmeme.com.",
        "Return a single JSON object with exactly these keys (no extra keys):",
    ]
    for name, kind, desc in spec:
        lines.append(f"  - {name} ({kind}): {desc}")
    lines += [
        "",
        "Rules:",
        "  - Use the empty value ('' for text, 0 for year, [] for lists, false for booleans) "
        "when information is missing. Never invent facts.",
        "  - Preserve original wording for description fields; do not summarise unless asked.",
        "  - Output ONLY the JSON object.",
    ]
    return "\n".join(lines)


def build_pydantic_schema(template_type: str):
    """
    Lazily build a pydantic model matching the field spec, for ScrapeGraph-AI's
    structured-output mode. Returns None if pydantic is unavailable.
    """
    try:
        from pydantic import BaseModel, Field, create_model
    except ImportError:
        return None

    type_map = {
        "str": (str, ""),
        "int": (int, 0),
        "bool": (bool, False),
        "list[str]": (List[str], None),
        "list[obj]": (List[Dict[str, Any]], None),
    }
    fields: Dict[str, Any] = {}
    for name, kind, desc in get_field_spec(template_type):
        py_type, default = type_map[kind]
        if default is None:  # mutable default -> use default_factory
            fields[name] = (py_type, Field(default_factory=list, description=desc))
        else:
            fields[name] = (py_type, Field(default=default, description=desc))
    model_name = "EditorialRecord" if spec_family(template_type) == "editorial" else "MemeEntryRecord"
    return create_model(model_name, __base__=BaseModel, **fields)
