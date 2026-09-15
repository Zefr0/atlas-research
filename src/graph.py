"""
The execution graph — where the multi-agent workflow is actually wired.

This file is the heart of the LangGraph design. Read it as a map:

    START ─▶ manager ─▶ researcher ─▶ writer ─▶ fact_checker ─▶ finalize ─▶ END
                            │            ▲            │
                            │            └── revise ──┘   (conditional edge,
                            └─▶ END (no evidence found)    capped at MAX_REVISIONS)

Key LangGraph concepts on display:

- StateGraph(ResearchState): declares the shared-state shape every node
  reads/writes (see state.py).
- add_node: registers each agent function under a name.
- add_edge: unconditional "after A, always run B".
- add_conditional_edges: after a node, a plain Python ROUTING FUNCTION looks
  at the state and returns the name of the next node. This is how the
  fact-check/revise loop exists: routing is deterministic code, so the loop
  can never run away — the LLM has no say in when it stops.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .agents.fact_checker import build_fact_checker_node
from .agents.manager import build_manager_node
from .agents.researcher import build_researcher_node
from .agents.writer import build_writer_node
from .config import Settings
from .console import say
from .state import ResearchState

UNVERIFIED_BANNER = (
    "> **⚠ Verification notice:** this report did not fully pass automated "
    "fact-checking within the revision limit. Unresolved reviewer notes are "
    "listed at the end. Treat specific figures with caution.\n\n"
)

# Distinct from the above: the reviewer never rendered a verdict at all, so this
# report is unreviewed rather than reviewed-and-found-wanting. Citation
# integrity (layer 1) did pass — nothing reaches the LLM judge until it does.
UNCHECKED_BANNER = (
    "> **⚠ Verification notice:** automated claim-checking could not be completed "
    "for this report — the reviewer model failed to return a verdict. Citations "
    "were still validated against the evidence list, but individual claims were "
    "not. Treat specific figures with caution.\n\n"
)


def _route_after_research(state: ResearchState) -> str:
    """If the Researcher found literally nothing, there is nothing to write.

    Ending early with zero findings beats letting the Writer hallucinate a
    report out of thin air — the empty `final_report` is detected by the web
    layer and explained to the user.
    """
    return "writer" if state["findings"] else END


def _build_route_after_check(max_revisions: int):
    """Build the routing function for the validation loop.

    The revision cap is THE guarantee this graph always terminates:
    revision_count only ever increases, so at worst we do
    1 initial draft + max_revisions rewrites, then finalize regardless.
    """

    def route_after_check(state: ResearchState) -> str:
        if state["verified"]:
            return "finalize"
        # The checker itself broke (not "found problems"). Another draft would
        # only be judged by the same broken checker, so ship what we have.
        if state.get("check_failed"):
            say("Graph", "Fact-check unavailable — finalizing the current draft")
            return "finalize"
        if state["revision_count"] > max_revisions:
            say("Graph", f"Revision limit ({max_revisions}) reached — finalizing with warnings")
            return "finalize"
        say("Graph", "Routing draft back to Writer for revision")
        return "writer"

    return route_after_check


def _finalize_node(state: ResearchState) -> dict:
    """Assemble the shippable report: banner (if needed) + draft + sources.

    The Sources section is generated here from `findings` — the same numbered
    list the Writer cited against — so citation numbers and source URLs can
    never disagree.
    """
    parts = []

    if not state["verified"]:
        parts.append(UNCHECKED_BANNER if state.get("check_failed") else UNVERIFIED_BANNER)

    parts.append(state["draft_report"].rstrip())

    # Numbered sources appendix, mirroring the evidence numbering exactly.
    parts.append("\n\n## Sources\n")
    parts.append(
        "\n".join(
            f"{i}. {f['fact']} — <{f['source_url']}>"
            for i, f in enumerate(state["findings"], 1)
        )
    )

    # If we shipped unverified, attach the outstanding critique transparently.
    if not state["verified"] and state["critique"]:
        parts.append("\n\n## Unresolved reviewer notes\n")
        parts.append("\n".join(f"- {c}" for c in state["critique"]))

    return {"final_report": "\n".join(parts)}


def build_graph(settings: Settings):
    """Wire nodes and edges, then compile into a runnable app."""
    graph = StateGraph(ResearchState)

    graph.add_node("manager", build_manager_node(settings))
    graph.add_node("researcher", build_researcher_node(settings))
    graph.add_node("writer", build_writer_node(settings))
    graph.add_node("fact_checker", build_fact_checker_node(settings))
    graph.add_node("finalize", _finalize_node)

    graph.add_edge(START, "manager")
    graph.add_edge("manager", "researcher")
    graph.add_conditional_edges("researcher", _route_after_research, ["writer", END])
    graph.add_edge("writer", "fact_checker")
    graph.add_conditional_edges(
        "fact_checker",
        _build_route_after_check(settings.max_revisions),
        ["writer", "finalize"],
    )
    graph.add_edge("finalize", END)

    # compile() turns the declaration into an executable with .invoke()/.stream().
    return graph.compile()


def initial_state(query: str) -> ResearchState:
    """A fresh whiteboard for one research run."""
    return ResearchState(
        query=query,
        sub_questions=[],
        report_outline=[],
        findings=[],
        draft_report="",
        critique=[],
        revision_count=0,
        verified=False,
        check_failed=False,
        final_report="",
    )
