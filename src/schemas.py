"""
Pydantic schemas = the "contracts" between agents.

Why this matters: raw LLM output is free text, and free text is unreliable to
parse. Instead, every agent that produces data another agent depends on is
forced (via `llm.with_structured_output(Schema)`) to return JSON matching one
of these schemas. LangChain converts the schema into a tool definition, the
model fills it in, and Pydantic validates the result — if the model returns
malformed data, the call fails loudly instead of poisoning downstream agents.

The `description=` on each field is not just documentation: it is sent to the
LLM as part of the tool definition, so it doubles as a micro-prompt telling
the model exactly what belongs in that field.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------- Manager ---
class ResearchPlan(BaseModel):
    """Produced by the Manager: how to break the user's query into work."""

    # NOTE: no min_length/max_length constraints on any schema field — Groq's
    # STRICT structured-output grammar supports only a subset of JSON Schema,
    # and length constraints risk rejection. Emptiness is handled in code.
    sub_questions: list[str] = Field(
        description=(
            "3 to 5 specific, independently searchable sub-questions that "
            "together fully answer the user's research query."
        ),
    )
    report_outline: list[str] = Field(
        description=(
            "Ordered section headings for the final report, e.g. "
            "['Introduction', 'Current Landscape', ..., 'Conclusion']."
        ),
    )


# ------------------------------------------------------------- Researcher ---
class ExtractedFact(BaseModel):
    """One atomic, sourced fact pulled from search results."""

    fact: str = Field(
        description="A single self-contained factual statement, in one sentence."
    )
    source_url: str = Field(
        description=(
            "The URL of the search result this fact came from. Must be copied "
            "EXACTLY from the provided results — never invented."
        )
    )


class ExtractedFacts(BaseModel):
    """Produced by the Researcher: all facts found for one sub-question."""

    facts: list[ExtractedFact] = Field(
        description="Every relevant fact found in the search results (may be empty).",
    )


# ------------------------------------------------------------ Fact-Checker ---
class Verdict(BaseModel):
    """Produced by the Fact-Checker: does the draft survive scrutiny?"""

    passed: bool = Field(
        description=(
            "True only if every claim in the report is supported by the "
            "research findings and every citation is used correctly."
        )
    )
    # No default here on purpose: Groq strict mode requires every property,
    # so the model must always emit critique (an empty list when passed=True).
    critique: list[str] = Field(
        description=(
            "If passed is False: specific, actionable problems, one per item, "
            "e.g. 'The claim about X in section 2 is not supported by any finding.' "
            "MUST be an empty list when passed is True."
        ),
    )
