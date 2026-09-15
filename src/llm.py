"""
LLM factory + invocation with explicit model failover.

All four agents get their models from here. Centralizing this means:
(a) swapping providers later is a change in ONE file, (b) retry/failover
behaviour is consistent everywhere, (c) each agent can still ask for its own
temperature — creative for the Writer, near-deterministic for the Fact-Checker.

Failover design (learned the hard way on Groq's free tier):
- Groq rate limits are tracked PER MODEL, so a second model is a second
  daily token budget. We build a [primary, fallback] list and walk it.
- A 429 can mean two very different things: a per-MINUTE limit (relief in
  seconds — the right move is to wait and retry the SAME model) or a
  per-DAY limit (relief in hours — the right move is the NEXT model).
  Groq's error message includes "try again in 7.66s" / "in 19m29s", so we
  parse that and wait only when the wait is short.
- We used LangChain's .with_fallbacks() first, but when both models fail it
  re-raises the PRIMARY's error and silently swallows what happened to the
  fallback — undebuggable. Explicit loops beat clever wrappers.
"""

from __future__ import annotations

import re
import time

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_groq import ChatGroq

from .config import Settings
from .console import say

# (display-name, runnable) pairs — the name is for progress/error messages.
ModelChain = list[tuple[str, object]]


def _groq(settings: Settings, model: str, temperature: float, effort: str | None) -> ChatGroq:
    """One Groq chat model.

    max_retries=2 covers transient network blips; rate-limit handling is done
    by our failover loop below (the client's blind retries can't distinguish
    per-minute from per-day limits). timeout=60: never hang on a dead socket.

    reasoning_effort: the gpt-oss models are reasoning models, and their
    hidden reasoning tokens COUNT AGAINST the daily quota. Mechanical tasks
    (fact extraction) run fine at "low", which meaningfully stretches the
    free tier; None keeps the model's default (medium) for judgment tasks.
    """
    kwargs = {}
    if effort:
        kwargs["reasoning_effort"] = effort
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=model,
        temperature=temperature,
        max_retries=2,
        timeout=60,
        **kwargs,
    )


def _model_names(settings: Settings) -> list[str]:
    names = [settings.groq_model]
    fb = settings.groq_fallback_model
    if fb and fb != settings.groq_model:
        names.append(fb)
    return names


def chat(settings: Settings, temperature: float = 0.2, effort: str | None = None) -> ModelChain:
    """Plain chat models, primary first, fallback (if configured) second."""
    return [(name, _groq(settings, name, temperature, effort)) for name in _model_names(settings)]


def structured_chat(
    settings: Settings, schema, temperature: float = 0.2, effort: str | None = None
) -> ModelChain:
    """Schema-bound chat models with the same failover order.

    Output binding uses method="json_schema" with strict=True — Groq's
    server-side constrained decoding in STRICT mode. Two lessons baked in:
    - method="function_calling" was flaky with gpt-oss models (prose replies,
      misspelled tool names, empty arguments).
    - json_schema WITHOUT strict runs in Groq's "best-effort" mode, which
      their docs admit can return schema-violating JSON — we saw it as
      intermittent 400 json_validate_failed. strict=True makes the schema
      a hard grammar constraint (langchain converts it to Groq's strict
      dialect: every field required, additionalProperties: false).
    """
    return [
        (
            name,
            _groq(settings, name, temperature, effort).with_structured_output(
                schema, method="json_schema", strict=True
            ),
        )
        for name in _model_names(settings)
    ]


# Appended to the conversation on retry after a malformed structured reply.
_STRUCTURED_NUDGE = SystemMessage(
    content=(
        "IMPORTANT: your previous reply failed validation. Respond ONLY with "
        "data matching the required schema. Do not reply in plain text. Do not "
        "ask clarifying questions — make reasonable assumptions and produce "
        "your best answer."
    )
)

# "The model produced a malformed generation" — retry the same model with a nudge.
_MALFORMED_SIGNATURES = ("tool_use_failed", "json_validate_failed", "OutputParserException")

# Matches Groq's relief hint: "try again in 7.66s" / "19m29.424s" / "1h2m3s".
# Every component is optional so hints without a seconds part ("try again in
# 2m") still parse — previously those fell through as unparseable and the
# model was abandoned for the fallback even when relief was seconds away.
_RETRY_AFTER_RE = re.compile(r"try again in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?")

# Waits longer than this mean "come back much later" — switch models instead.
_MAX_WAIT_SECONDS = 90


def _retry_after_seconds(error_text: str) -> float | None:
    m = _RETRY_AFTER_RE.search(error_text)
    if not m or not any(m.groups()):  # all-optional pattern can match nothing
        return None
    hours, minutes, seconds = m.groups()
    return int(hours or 0) * 3600 + int(minutes or 0) * 60 + float(seconds or 0)


def _invoke_one(runnable, messages: list[BaseMessage], attempts: int) -> object:
    """Invoke ONE model, retrying only malformed generations (with a nudge)."""
    msgs = list(messages)
    last_exc: Exception | None = None
    for _ in range(attempts):
        try:
            return runnable.invoke(msgs)
        except Exception as exc:
            blob = str(exc) + type(exc).__name__
            if not any(sig in blob for sig in _MALFORMED_SIGNATURES):
                raise  # not a generation problem — let the failover loop decide
            last_exc = exc
            msgs = list(messages) + [_STRUCTURED_NUDGE]
    raise RuntimeError(
        f"Model produced invalid output after {attempts} attempts. Last error: {last_exc}"
    ) from last_exc


def invoke_with_failover(models: ModelChain, messages: list[BaseMessage], *, attempts: int = 3):
    """Invoke with per-model retries and explicit cross-model failover.

    Per model: try it (malformed generations get nudge-retried inside);
    if it is rate-limited with a SHORT relief time, wait that out once and
    try the same model again — per-minute limits clear in seconds and the
    next model may be weaker, so patience beats downgrading. Anything else
    (daily exhaustion, auth on that model, network death) moves on to the
    next model. Every switch is reported via say() so the server log and the
    web UI both show exactly which model did the work.
    """
    last_exc: Exception | None = None
    for index, (name, runnable) in enumerate(models):
        for pass_no in (1, 2):  # pass 2 only happens after a short rate-limit wait
            try:
                return _invoke_one(runnable, messages, attempts)
            except Exception as exc:
                last_exc = exc
                # Check the provider's status code, not a bare "429" substring:
                # that digit sequence turns up in token counts and request ids.
                is_rate_limit = (
                    "rate_limit_exceeded" in str(exc)
                    or getattr(exc, "status_code", None) == 429
                )
                wait = _retry_after_seconds(str(exc)) if is_rate_limit else None
                if pass_no == 1 and wait is not None and wait <= _MAX_WAIT_SECONDS:
                    say("System", f"{name} rate-limited — waiting {wait:.0f}s and retrying")
                    time.sleep(wait + 1)
                    continue
                break  # this model is out; fall through to the next one
        if index < len(models) - 1:
            say("System", f"{name} unavailable ({type(last_exc).__name__}) — switching to {models[index + 1][0]}")
    raise last_exc
