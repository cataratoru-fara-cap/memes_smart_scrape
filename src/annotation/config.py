"""
Configuration for the meme annotation pipeline.

All values are read from environment variables (see .env.example) so nothing
sensitive is hard-coded. The LLM provider is intentionally swappable: set
ANNOTATION_LLM_PROVIDER / ANNOTATION_LLM_MODEL and the matching API key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # dotenv is optional for the pure-stdlib paths
    pass


def _get_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes")


# Env var that holds the API key, keyed by provider.
_PROVIDER_KEY_ENV: Dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "ollama": "",  # local, no key
}


@dataclass
class AnnotationConfig:
    """Resolved configuration for an annotation run."""

    # --- LLM ---
    provider: str = field(default_factory=lambda: os.getenv("ANNOTATION_LLM_PROVIDER", "openai").lower())
    model: str = field(default_factory=lambda: os.getenv("ANNOTATION_LLM_MODEL", "gpt-4o-mini"))
    temperature: float = field(default_factory=lambda: float(os.getenv("ANNOTATION_LLM_TEMPERATURE", "0.0")))
    api_key: Optional[str] = None  # resolved in __post_init__
    base_url: Optional[str] = field(default_factory=lambda: os.getenv("ANNOTATION_LLM_BASE_URL") or None)

    # --- IO ---
    # source: "file" reads URL records from input_path; "mongo" ingests the
    # discovery output from the MongoDB ``urls`` collection (see src/db).
    source: str = field(default_factory=lambda: os.getenv("ANNOTATION_SOURCE", "file").lower())
    input_path: str = field(default_factory=lambda: os.getenv("ANNOTATION_INPUT_PATH", "data/meme_urls.sample.json"))
    output_path: str = field(default_factory=lambda: os.getenv("ANNOTATION_OUTPUT_PATH", "data/annotations.jsonl"))

    # --- Proxy (required when your IP is banned by the target site) ---
    proxy_url: str | None = field(default_factory=lambda: os.getenv("ANNOTATION_PROXY_URL") or None)
    proxy_username: str | None = field(default_factory=lambda: os.getenv("ANNOTATION_PROXY_USERNAME") or None)
    proxy_password: str | None = field(default_factory=lambda: os.getenv("ANNOTATION_PROXY_PASSWORD") or None)

    # --- Run control ---
    concurrency: int = field(default_factory=lambda: int(os.getenv("ANNOTATION_CONCURRENCY", "4")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("ANNOTATION_MAX_RETRIES", "3")))
    request_timeout: int = field(default_factory=lambda: int(os.getenv("ANNOTATION_REQUEST_TIMEOUT", "120")))
    rate_limit_per_min: int = field(default_factory=lambda: int(os.getenv("ANNOTATION_RATE_LIMIT_PER_MIN", "0")))  # 0 = unlimited
    headless: bool = field(default_factory=lambda: _get_bool("ANNOTATION_HEADLESS", True))
    only_confirmed: bool = field(default_factory=lambda: _get_bool("ANNOTATION_ONLY_CONFIRMED", True))
    # Pass a JSON schema to the LLM (function-calling / json_schema). Hosted
    # OpenAI supports this; many self-hosted/Ollama models behind a gateway do
    # not — set false to rely on the prompt's "output JSON only" + normalize().
    structured_output: bool = field(default_factory=lambda: _get_bool("ANNOTATION_STRUCTURED_OUTPUT", True))

    def __post_init__(self) -> None:
        key_env = _PROVIDER_KEY_ENV.get(self.provider, "")
        if key_env:
            self.api_key = os.getenv(key_env)

    def require_api_key(self) -> None:
        """Raise a clear error if a key is required but missing."""
        key_env = _PROVIDER_KEY_ENV.get(self.provider, "")
        if key_env and not self.api_key:
            raise RuntimeError(
                f"Provider '{self.provider}' requires {key_env} to be set "
                f"(add it to your .env). For local/offline runs use --mock."
            )

    def graph_config(self, proxy_override: str | None = None) -> dict:
        """
        Build the ScrapeGraph-AI ``graph_config`` dict for this run.
        See https://github.com/ScrapeGraphAI/Scrapegraph-ai for the schema.

        ``proxy_override`` (e.g. from the rotating proxy pool) takes precedence
        over the static ANNOTATION_PROXY_URL for this call.
        """
        llm: dict = {"model": f"{self.provider}/{self.model}", "temperature": self.temperature}
        if self.api_key:
            llm["api_key"] = self.api_key
        if self.base_url:
            llm["base_url"] = self.base_url
        cfg: dict = {
            "llm": llm,
            "headless": self.headless,
            "verbose": False,
        }
        if proxy_override:
            cfg["loader_kwargs"] = {"proxy": {"server": proxy_override}}
        elif self.proxy_url:
            proxy: dict = {"server": self.proxy_url}
            if self.proxy_username:
                proxy["username"] = self.proxy_username
            if self.proxy_password:
                proxy["password"] = self.proxy_password
            cfg["loader_kwargs"] = {"proxy": proxy}
        return cfg
