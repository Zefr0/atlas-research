"""
FACT-CHECKER agent — hallucination detection and citation validation.

Runs two layers of checks, cheapest first:

Layer 1 (deterministic, pure Python): citation integrity.
  Regex-extract every citation in the draft — in any bracket style the model
  produces, ranges included — and verify each number exists in the evidence
  list. A citation like [17] when there are 12 findings is proof of
  hallucination: no LLM judgment needed, and no LLM can be sweet-talked out of
  it. If this layer fails we don't even spend tokens on layer 2.

Layer 2 (LLM-as-judge): claim support.
  A separate LLM pass — with temperature 0 and an adversarial "you are
  skeptical" persona — compares every claim in the draft against the evidence
  and returns a structured Verdict. Using a fresh call (not the Writer's
  conversation) matters: the checker has no investment in the draft being good.

On failure, the critique items flow back to the Writer through state, and the
graph (see graph.py) routes the draft back for revision — the validation loop.
"""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import Settings
from ..console import say
from ..llm import invoke_with_failover, structured_chat
from ..schemas import Verdict
from ..state import ResearchState
from .writer import _numbered_evidence

SYSTEM_PROMPT = """\
You are a skeptical Fact-Checker reviewing a research report before publication.
You are given the report and the ONLY evidence its author was allowed to use.
Check:
1. Every factual claim is supported by at least one cited piece of evidence.
2. Each citation [n] is attached to a claim that evidence item n actually supports.
3. Nothing important in the evidence is contradicted by the report.
Minor paraphrasing is fine; unsupported claims, wrong citations, or invented
specifics (numbers, dates, names not in evidence) are failures.
Return passed=true ONLY if the report survives all checks. Otherwise return
passed=false with a specific, actionable critique item for each problem.

Keep the output SHORT: at most 5 critique items, each ONE sentence under 200
characters. Name the section and the problem; never quote long passages back.
An over-long structured reply gets truncated and thrown away by the inference
server, so brevity here is a hard requirement, not a preference."""

# Appended when a verdict call failed to produce valid JSON — see the retry
# in fact_checker_node.
TERSE_NUDGE = (
    "Your previous reply could not be returned. Answer again with at most THREE "
    "critique items, each under 120 characters. Be terse."
)

# Matches an inline citation and captures its contents.
#
# This deliberately accepts EVERY bracket style the Writer has been observed to
# emit, not just the one we ask for. gpt-oss models freely produce CJK
# lenticular brackets (【3】) and ranges joined by a non-breaking hyphen
# (U+2011) or en dash. When this pattern missed those forms, the layer failed
# in BOTH directions: fully-cited reports were rejected as "no citations at
# all", and out-of-range citations hid inside the unmatched forms — the exact
# hallucination this check exists to catch.
_CITATION_RE = re.compile(
    r"[\[【]\s*(\d+(?:\s*[-‐-―]\s*\d+)?(?:\s*[,;]\s*\d+(?:\s*[-‐-―]\s*\d+)?)*)\s*[\]】]"
)
# One entry inside a citation: a number, optionally a range like "5-7".
_CITATION_ITEM_RE = re.compile(r"(\d+)(?:\s*[-‐-―]\s*(\d+))?")


def _cited_numbers(draft: str) -> set[int]:
    """Every evidence number the draft cites, ranges included.

    For a range only the endpoints are collected: evidence numbering is
    contiguous, so if both ends are in bounds every number between them is too.
    """
    cited: set[int] = set()
    for match in _CITATION_RE.finditer(draft):
        for item in re.split(r"[,;]", match.group(1)):
            parsed = _CITATION_ITEM_RE.fullmatch(item.strip())
            if parsed is None:
                continue
            cited.add(int(parsed.group(1)))
            if parsed.group(2):
                cited.add(int(parsed.group(2)))
    return cited


def _check_citation_integrity(draft: str, num_findings: int) -> list[str]:
    """Layer 1: return a critique item for every impossible citation."""
    problems = []
    cited = _cited_numbers(draft)

    if not cited:
        problems.append("The report contains no [n] citations at all; every factual claim must cite evidence.")
    for n in sorted(cited):
        if n < 1 or n > num_findings:
            problems.append(
                f"Citation [{n}] does not exist — the evidence list only has items 1-{num_findings}."
            )
    return problems


def build_fact_checker_node(settings: Settings):
    # Temperature 0: a judge should be maximally consistent, never creative.
    llm = structured_chat(settings, Verdict, temperature=0.0)
    # Used only when the main chain cannot emit valid JSON. Low reasoning effort
    # leaves more of the completion budget for the answer itself.
    llm_terse = structured_chat(settings, Verdict, temperature=0.0, effort="low")

    def fact_checker_node(state: ResearchState) -> dict:
        draft = state["draft_report"]
        say("Fact-Checker", f"Validating draft #{state['revision_count']}")

        # ---- Layer 1: deterministic citation check ------------------------
        problems = _check_citation_integrity(draft, len(state["findings"]))
        if problems:
            say("Fact-Checker", f"FAILED citation integrity ({len(problems)} problem(s))")
            return {"verified": False, "critique": problems}

        # ---- Layer 2: LLM claim-support check -----------------------------
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Evidence the author was allowed to use:\n"
                    f"{_numbered_evidence(state)}\n\n"
                    f"Report to check:\n{draft}"
                )
            ),
        ]
        try:
            verdict: Verdict = invoke_with_failover(llm, messages)
        except Exception as exc:
            # Structured output fails outright when the model spends its
            # completion budget before finishing the JSON: Groq reports that as
            # json_validate_failed with an empty generation, on every model in
            # the chain, so retrying the same way is pointless. Ask again
            # cheaply and tersely, and if that fails too, give up WITHOUT
            # destroying the run — a finished draft is far too valuable to
            # discard just because the reviewer could not phrase its answer.
            say("Fact-Checker", f"verdict call failed ({type(exc).__name__}); retrying tersely")
            try:
                verdict = invoke_with_failover(
                    llm_terse, messages + [SystemMessage(content=TERSE_NUDGE)]
                )
            except Exception as exc2:
                say("Fact-Checker", f"validation could not complete: {exc2}")
                say("Fact-Checker", "shipping the draft unverified rather than losing it")
                return {"verified": False, "critique": [], "check_failed": True}

        if verdict.passed:
            say("Fact-Checker", "PASSED — report is verified")
            return {"verified": True, "critique": []}

        # A rejection with no reasons would be useless to the Writer: it checks
        # `critique` to decide whether it is revising, so an empty list makes it
        # silently re-roll the same prompt until the revision cap. Substitute a
        # generic note so the revision loop always carries a signal.
        critique = verdict.critique or [
            "The fact-checker rejected this draft without specifics. Re-verify that every "
            "claim is supported by the cited evidence and remove anything that is not."
        ]

        say("Fact-Checker", f"FAILED claim support ({len(critique)} problem(s))")
        for c in critique:
            say("Fact-Checker", f"  - {c}")
        return {"verified": False, "critique": critique}

    return fact_checker_node
