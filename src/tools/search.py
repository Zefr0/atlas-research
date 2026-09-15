"""
Web search tool with a two-tier fallback strategy.

Tier 1: Tavily  — built for LLM agents: returns the extracted TEXT of each
                  page (not just a link), so the Researcher can pull facts
                  without a separate scraping step. Needs an API key.
Tier 2: DuckDuckGo (ddgs) — keyless and unlimited, but its results only carry
                  ~150-char snippets, which usually contain no extractable
                  facts (a snippet like "List of foreign players..." names no
                  players). So after a DDG search we FETCH the top result
                  pages ourselves and extract their visible text, giving the
                  Researcher real content to work with. Used when Tavily is
                  unconfigured OR fails (quota exhausted, outage).

Both tiers are normalized to the same SearchResult shape, so the Researcher
never knows or cares which engine answered. That is the "tool abstraction"
pattern: agents depend on an interface, not a vendor.
"""

from __future__ import annotations

import re
import urllib.request
from html.parser import HTMLParser
from typing import TypedDict

from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import Settings
from ..console import say


class SearchResult(TypedDict):
    """Normalized shape every search backend must produce."""

    title: str
    url: str
    content: str  # extracted page text (Tavily) or snippet (DuckDuckGo)


class SearchError(RuntimeError):
    """Raised when every backend failed for a query."""


# `tenacity` decorator: up to 3 attempts with exponential backoff between them
# (2s, then 4s — each capped at 10s). Network calls are the flakiest part of
# any agent system; retrying here means one dropped packet doesn't kill a
# whole research run.
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10), reraise=True)
def _tavily_search(api_key: str, query: str, max_results: int) -> list[SearchResult]:
    from tavily import TavilyClient  # imported lazily: only needed if a key exists

    response = TavilyClient(api_key=api_key).search(
        query=query,
        max_results=max_results,
        search_depth="basic",   # "advanced" is better but costs 2 credits/call
    )
    return [
        SearchResult(
            title=r.get("title", ""),
            url=r.get("url", ""),
            content=r.get("content", ""),
        )
        for r in response.get("results", [])
        if r.get("url")  # a result without a URL can never be cited — drop it
    ]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10), reraise=True)
def _duckduckgo_search(query: str, max_results: int) -> list[SearchResult]:
    from ddgs import DDGS

    with DDGS() as ddgs:
        raw = ddgs.text(query, max_results=max_results)
    return [
        SearchResult(
            title=r.get("title", ""),
            url=r.get("href", ""),
            content=r.get("body", ""),
        )
        for r in raw
        if r.get("href")
    ]


# ---------------------------------------------------------------------------
# Page-content enrichment for the DuckDuckGo tier.
# ---------------------------------------------------------------------------

# Tags whose text is never useful content — we drop everything inside them.
_SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "head", "nav", "footer", "form"}
# Tags that carry readable content worth keeping (paragraphs, list items,
# headings, table cells — table cells matter a lot for "list of X" pages).
_TEXT_TAGS = {"p", "li", "h1", "h2", "h3", "h4", "td", "th", "caption", "blockquote", "dd", "dt"}


class _TextExtractor(HTMLParser):
    """Minimal HTML -> visible text, standard library only.

    Deliberately not a full readability implementation: collecting text from
    content-bearing tags while skipping script/style/nav gets ~90% of the
    value with zero extra dependencies.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.chunks: list[str] = []
        self._skip_depth = 0      # >0 while inside a skipped tag
        self._capture_depth = 0   # >0 while inside a content tag

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _TEXT_TAGS:
            self._capture_depth += 1

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif tag in _TEXT_TAGS and self._capture_depth:
            self._capture_depth -= 1
            self.chunks.append("\n")  # keep block boundaries readable

    def handle_data(self, data):
        if self._skip_depth == 0 and self._capture_depth > 0:
            self.chunks.append(data)


def _fetch_page_text(url: str, max_chars: int = 6000) -> str:
    """Download one page and return its visible text ('' on any failure).

    Guardrails: http(s) only, 10s timeout, HTML-only, read at most 500KB, and a
    browser-ish User-Agent (many sites 403 the default Python one). The URL
    comes from a third-party search engine, so the scheme check matters:
    urlopen would happily follow a `file:` URL and read the local disk.
    """
    if not url.lower().startswith(("http://", "https://")):
        return ""

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) research-assistant/1.0"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        if "html" not in response.headers.get("Content-Type", "").lower():
            return ""
        raw = response.read(500_000)

    parser = _TextExtractor()
    parser.feed(raw.decode("utf-8", errors="ignore"))
    # Collapse the whitespace soup HTML leaves behind into clean lines.
    text = re.sub(r"[ \t]+", " ", "".join(parser.chunks))
    text = re.sub(r"\n\s*\n+", "\n", text).strip()
    return text[:max_chars]


def _enrich_with_page_text(results: list[SearchResult], max_fetches: int = 4) -> list[SearchResult]:
    """Replace thin DDG snippets with real page text where we can get it.

    Only the top `max_fetches` results are fetched to bound latency; a page
    that fails to download (403, timeout, dead link) just keeps its snippet —
    enrichment is best-effort, never a new failure mode.
    """
    for result in results[:max_fetches]:
        try:
            page_text = _fetch_page_text(result["url"])
        except Exception:
            continue
        # Only swap if the page gave us more than the snippet already had.
        if len(page_text) > len(result["content"]):
            result["content"] = page_text
    return results


def search_web(settings: Settings, query: str) -> list[SearchResult]:
    """Search the web, preferring Tavily, falling back to DuckDuckGo.

    Returns an empty list (rather than raising) when a query genuinely has no
    results, so the Researcher can note the gap and keep going. Raises
    SearchError only when every backend errored — a real infrastructure
    problem the caller should surface.
    """
    if settings.tavily_api_key:
        try:
            return _tavily_search(settings.tavily_api_key, query, settings.max_search_results)
        except Exception as exc:  # quota exhausted, invalid key, outage, ...
            say("Search", f"Tavily failed ({exc}); falling back to DuckDuckGo")

    try:
        results = _duckduckgo_search(query, settings.max_search_results)
    except Exception as exc:
        raise SearchError(f"All search backends failed for query {query!r}: {exc}") from exc

    # DDG snippets alone are too thin to extract facts from — pull the real
    # page text for the top results (Tavily results already include content).
    return _enrich_with_page_text(results)
