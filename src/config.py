"""
Central configuration.

Everything the app needs from the outside world (API keys, tuning knobs) is
read HERE and only here, at startup. Agents never call os.getenv() themselves —
they receive a validated Settings object. That gives us one place to fail fast
with a helpful message instead of a cryptic crash deep inside an agent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Reads the `.env` file in the project root and injects its values into
# os.environ. Real environment variables (already set in your shell) win over
# .env values, which is the standard 12-factor behaviour.
load_dotenv()

# Project root = one level above src/. Reports are written next to the code.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)  # frozen=True -> immutable: no agent can mutate config at runtime
class Settings:
    groq_api_key: str
    groq_model: str
    groq_fallback_model: str | None  # auto-failover target when the primary is rate-limited
    tavily_api_key: str | None   # None => fall back to DuckDuckGo
    max_revisions: int           # cap on Fact-Checker -> Writer feedback loops
    max_search_results: int      # results fetched per sub-question
    reports_dir: Path


class ConfigError(RuntimeError):
    """Raised when the environment is missing something we cannot run without."""


def load_settings() -> Settings:
    """Validate the environment and return an immutable Settings object.

    Called by the web layer before a run starts, so a missing key becomes a
    clear error with instructions ("here is how to fix it") instead of an
    HTTP 401 stack trace three minutes into a research run.
    """
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if not groq_key:
        raise ConfigError(
            "GROQ_API_KEY is not set.\n"
            "  1. Get a free key at https://console.groq.com/keys\n"
            "  2. Copy .env.example to .env and paste the key into GROQ_API_KEY="
        )

    tavily_key = os.getenv("TAVILY_API_KEY", "").strip() or None

    def _positive_int(name: str, default: int) -> int:
        """Parse an optional integer env var, rejecting garbage values early."""
        raw = os.getenv(name, "").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            raise ConfigError(f"{name} must be an integer, got: {raw!r}")
        if value < 0:
            raise ConfigError(f"{name} must be >= 0, got: {value}")
        return value

    return Settings(
        groq_api_key=groq_key,
        groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip(),
        # Groq rate limits are PER MODEL, so a second model is a second daily
        # token budget. Set GROQ_FALLBACK_MODEL= (empty) to disable failover.
        groq_fallback_model=(os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b").strip() or None),
        tavily_api_key=tavily_key,
        max_revisions=_positive_int("MAX_REVISIONS", 2),
        max_search_results=_positive_int("MAX_SEARCH_RESULTS", 5),
        reports_dir=PROJECT_ROOT / "reports",
    )
