"""
MANAGER agent — task decomposition.

Role in the pipeline: turn one fuzzy human question into a concrete work
plan. Two outputs, both structured (see schemas.ResearchPlan):

  - sub_questions: what the Researcher will search for, one query at a time.
    Decomposition matters because a single web search for a broad question
    ("impact of AI on healthcare") returns shallow, generic results; 4 sharp
    sub-questions return deep, specific ones.
  - report_outline: the section structure the Writer must follow, decided
    UP FRONT so the report is designed top-down rather than emerging from
    whatever facts happened to be found.

Note the factory pattern (`build_manager_node`): the graph needs node
functions with the signature `state -> partial_update`, but our nodes also
need settings and an LLM. A closure captures those once at build time.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import Settings
from ..console import say
from ..llm import invoke_with_failover, structured_chat
from ..schemas import ResearchPlan
from ..state import ResearchState

SYSTEM_PROMPT = """\
You are the Research Manager of an autonomous research team.
Given a user's research query, produce a research plan:
1. Break the query into 3-5 specific sub-questions. Each must be
   independently answerable via a web search, concrete, and non-overlapping.
2. Design a logical report outline (section headings only) that a reader
   would find clear and complete for this topic.
Do not answer the query yourself. Plan only.
Never ask for clarification and never refuse: if the query is vague, broad,
or oddly phrased, plan for the most reasonable interpretation of it."""


def build_manager_node(settings: Settings):
    # structured_chat() forces the model to reply as JSON matching the
    # ResearchPlan schema via Groq's constrained decoding, with automatic
    # failover to the fallback model on rate limits (see llm.py).
    llm = structured_chat(settings, ResearchPlan, temperature=0.3)

    def manager_node(state: ResearchState) -> dict:
        say("Manager", f'Decomposing query: "{state["query"]}"')

        # invoke_with_failover retries malformed generations, waits out short
        # rate limits, and switches to the fallback model when the primary's
        # daily quota is exhausted — see llm.py.
        plan: ResearchPlan = invoke_with_failover(
            llm,
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=f"Research query: {state['query']}"),
            ],
        )

        for i, sq in enumerate(plan.sub_questions, 1):
            say("Manager", f"  sub-question {i}: {sq}")
        say("Manager", f"Outline: {' > '.join(plan.report_outline)}")

        # Return ONLY the keys this agent owns; LangGraph merges them in.
        return {
            "sub_questions": plan.sub_questions,
            "report_outline": plan.report_outline,
        }

    return manager_node
