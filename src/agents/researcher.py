"""
RESEARCHER agent — live search + fact extraction.

For EACH sub-question from the Manager's plan, this agent:
  1. Runs a real web search (Tavily, or DuckDuckGo fallback — see tools/search.py).
  2. Feeds the raw results to the LLM and asks it to extract atomic facts,
     each tagged with the exact source URL it came from (structured output:
     ExtractedFacts).
  3. Applies a *deterministic* anti-hallucination guard: any "fact" whose
     source_url was not actually in the search results is discarded. The LLM
     physically cannot smuggle an invented citation into the state, because
     plain Python — not another LLM — enforces this rule.

The result is the `findings` list: the ONLY evidence pool the Writer is
allowed to draw from, and the ground truth the Fact-Checker validates against.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import Settings
from ..console import say
from ..llm import invoke_with_failover, structured_chat
from ..schemas import ExtractedFacts
from ..state import Finding, ResearchState
from ..tools.search import SearchError, search_web

SYSTEM_PROMPT = """\
You are a meticulous Research Analyst. You will be given a sub-question and
raw web search results. Extract every fact that helps answer the sub-question.
Rules:
- Each fact must be ONE self-contained sentence.
- Each fact's source_url must be copied EXACTLY from the result it came from.
- Only extract what the results actually say. Never add outside knowledge.
- If the results contain nothing relevant, return an empty list."""


def _format_results_for_llm(results) -> str:
    """Render search results as a numbered plain-text block for the prompt.

    Content is truncated per result to keep the prompt inside the model's
    context comfortably even with 5 full Tavily page extracts.
    """
    blocks = []
    for i, r in enumerate(results, 1):
        blocks.append(
            f"[Result {i}]\nTitle: {r['title']}\nURL: {r['url']}\n"
            f"Content: {r['content'][:3000]}"
        )
    return "\n\n".join(blocks)


def build_researcher_node(settings: Settings):
    # Low temperature: extraction should be faithful, not creative.
    # effort="low": extraction is mechanical — no deep reasoning needed, and
    # reasoning tokens count against the daily quota (5 of these calls per
    # run make this the single biggest token saving in the pipeline).
    llm = structured_chat(settings, ExtractedFacts, temperature=0.0, effort="low")

    def researcher_node(state: ResearchState) -> dict:
        findings: list[Finding] = []
        last_error: Exception | None = None  # see the re-raise at the end

        for sub_q in state["sub_questions"]:
            say("Researcher", f"Searching: {sub_q}")

            # --- Step 1: live web search ------------------------------------
            try:
                results = search_web(settings, sub_q)
            except SearchError as exc:
                # One dead sub-question shouldn't kill the run; log and move on.
                say("Researcher", f"  search failed, skipping: {exc}")
                continue

            if not results:
                say("Researcher", "  no results found for this sub-question")
                continue

            # --- Step 2: LLM extracts facts from the raw results ------------
            # Catch broadly on purpose: invoke_with_failover re-raises whatever
            # the last model failed with — a RuntimeError for malformed output,
            # but equally a provider RateLimitError or a transport error. All of
            # them should cost us this ONE sub-question, not the whole run and
            # every finding gathered so far.
            try:
                extracted: ExtractedFacts = invoke_with_failover(
                    llm,
                    [
                        SystemMessage(content=SYSTEM_PROMPT),
                        HumanMessage(
                            content=(
                                f"Sub-question: {sub_q}\n\n"
                                f"Search results:\n\n{_format_results_for_llm(results)}"
                            )
                        ),
                    ],
                )
            except Exception as exc:
                last_error = exc
                say("Researcher", f"  extraction failed, skipping sub-question: {exc}")
                continue

            # --- Step 3: deterministic citation guard -----------------------
            # A set of URLs that REALLY came back from the search engine.
            valid_urls = {r["url"] for r in results}
            kept = 0
            for f in extracted.facts:
                if f.source_url in valid_urls:
                    findings.append(
                        Finding(sub_question=sub_q, fact=f.fact, source_url=f.source_url)
                    )
                    kept += 1
                else:
                    say("Researcher", f"  DISCARDED fact with invented URL: {f.source_url}")
            say("Researcher", f"  kept {kept} sourced fact(s)")

        # Surviving every sub-question is only a success if we actually got
        # something. Zero findings AND a swallowed error means the failure was
        # systemic (quota exhausted, provider down) — re-raise it so the user
        # sees the real cause instead of a misleading "no sources found".
        if not findings and last_error is not None:
            raise last_error

        say("Researcher", f"Research complete: {len(findings)} total findings")
        return {"findings": findings}

    return researcher_node
