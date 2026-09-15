"""
The shared graph state — the single "whiteboard" all agents read and write.

LangGraph works like this: the state is a dict that flows through the graph.
Each node (agent) receives the CURRENT state and returns a PARTIAL update
(only the keys it changed); LangGraph merges that update into the state before
passing it to the next node. This is why agents stay decoupled: the Writer
never calls the Researcher — it just reads `findings` off the whiteboard.
"""

from __future__ import annotations

from typing import TypedDict


class Finding(TypedDict):
    """One sourced fact gathered by the Researcher.

    Keeping `source_url` attached to every fact from the moment of capture is
    what makes fact-checking and citations possible later: the Writer may only
    cite URLs that exist in this list, and the Fact-Checker verifies that.
    """

    sub_question: str   # which part of the research plan this fact answers
    fact: str           # the atomic factual statement
    source_url: str     # where it came from


class ResearchState(TypedDict):
    """Everything the pipeline knows, at any point in time."""

    query: str                  # the user's original research question
    sub_questions: list[str]    # written by Manager
    report_outline: list[str]   # written by Manager
    findings: list[Finding]     # written by Researcher
    draft_report: str           # written by Writer (possibly multiple times)
    critique: list[str]         # written by Fact-Checker on failure; read by Writer
    revision_count: int         # how many drafts the Writer has produced
    verified: bool              # Fact-Checker's final verdict
    check_failed: bool          # the Fact-Checker could not run (not: found faults)
    final_report: str           # written by the finalize step
