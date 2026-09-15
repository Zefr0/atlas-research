"""
WRITER agent — report synthesis.

Takes the evidence pool (`findings`) and the Manager's outline and produces a
Markdown report with numbered citations. Two design points to notice:

1. Evidence numbering happens in PYTHON, not in the LLM. We assign [1], [2]...
   to each finding ourselves and hand the model a pre-numbered evidence list.
   The model only ever references numbers we defined, and the Sources section
   is generated from the same numbering — so citation numbers can't drift.

2. This node runs MORE THAN ONCE. When the Fact-Checker rejects a draft, the
   graph routes back here and `state["critique"]` is non-empty; we inject the
   critique (and the previous draft) into the prompt so the model fixes the
   specific problems instead of writing from scratch. `revision_count` tracks
   how many drafts have been produced — the graph uses it to cap the loop.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import Settings
from ..console import say
from ..llm import chat, invoke_with_failover
from ..state import ResearchState

SYSTEM_PROMPT = """\
You are a Senior Research Writer. Write a well-structured Markdown report.
Hard rules:
- Use ONLY the numbered evidence provided. Never add outside knowledge.
- Cite evidence inline with its number in PLAIN ASCII SQUARE BRACKETS, one
  number per bracket, e.g. "...grew 40% in 2025 [3]." or "...in both [3][7]."
  Never use other bracket characters and never write ranges like [3-7].
- Every factual claim needs at least one citation.
- Follow the given outline exactly, using ## for section headings.
- Start with a single # title line. Do NOT write a Sources section —
  it is appended automatically.
- Write clear, information-dense prose. No filler, no marketing tone."""


def _numbered_evidence(state: ResearchState) -> str:
    """Render findings as '[n] fact (source: url)' lines, grouped per sub-question."""
    lines = []
    current_q = None
    for i, f in enumerate(state["findings"], 1):
        if f["sub_question"] != current_q:
            current_q = f["sub_question"]
            lines.append(f"\n--- Evidence for: {current_q} ---")
        lines.append(f"[{i}] {f['fact']} (source: {f['source_url']})")
    return "\n".join(lines)


def build_writer_node(settings: Settings):
    # Slightly higher temperature than the Researcher: prose benefits from
    # some fluency, but stays low enough to keep claims tethered to evidence.
    llm = chat(settings, temperature=0.4)

    def writer_node(state: ResearchState) -> dict:
        is_revision = bool(state["critique"])
        draft_no = state["revision_count"] + 1
        say("Writer", f"{'Revising' if is_revision else 'Writing'} draft #{draft_no}")

        user_parts = [
            f"Research query: {state['query']}",
            f"Report outline:\n" + "\n".join(f"- {s}" for s in state["report_outline"]),
            f"Numbered evidence:\n{_numbered_evidence(state)}",
        ]

        # On a revision pass, show the model its rejected draft and the
        # Fact-Checker's specific complaints. Targeted repair beats rewriting.
        if is_revision:
            user_parts.append(
                "Your previous draft was REJECTED by the fact-checker.\n"
                "Previous draft:\n" + state["draft_report"] + "\n\n"
                "Fix ALL of these problems (change nothing that wasn't flagged):\n"
                + "\n".join(f"- {c}" for c in state["critique"])
            )

        response = invoke_with_failover(
            llm,
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content="\n\n".join(user_parts)),
            ],
        )

        say("Writer", f"Draft #{draft_no} complete ({len(response.content)} chars)")
        return {
            "draft_report": response.content,
            "revision_count": draft_no,
        }

    return writer_node
