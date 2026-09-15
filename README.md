# Multi-Agent Autonomous Research Assistant

A web app where four cooperating agents (Manager → Researcher → Writer →
Fact-Checker) research a question and produce a **cited report**, viewable in the
browser and downloadable as Markdown or PDF. Built on **LangGraph** + **Groq**,
with live web search via **Tavily** (DuckDuckGo fallback) and a self-correcting
fact-checking loop.

## Setup (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# then edit .env and paste your GROQ_API_KEY (and optionally TAVILY_API_KEY)
```

API keys (both have free tiers):
- **Groq** (required): https://console.groq.com/keys
- **Tavily** (optional, much better search): https://app.tavily.com — without it,
  the app falls back to DuckDuckGo plus its own page-text fetching.

## Run

```powershell
.\.venv\Scripts\python.exe -m src.web.app
```

Then open **http://localhost:8010**. Type a question, watch the four agents work
live, read the report, and download it as PDF or Markdown. Past reports are
listed in the history panel.

`HOST` and `PORT` environment variables override the defaults. The default host
is `127.0.0.1` on purpose — **the app has no authentication**, so anyone who can
reach the port can spend your Groq quota. Put auth in front of it before
exposing it beyond your machine.

Deploy the same app behind any ASGI host:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.web.app:app --host 0.0.0.0 --port 8010
```

## Project layout

```
src/
  config.py                 .env loading + validation (fail fast)
  state.py                  Shared graph state (the agents' "whiteboard")
  schemas.py                Pydantic contracts for structured LLM outputs
  llm.py                    Groq model factory + retries and model failover
  console.py                Progress logging (server log + web UI stream)
  events.py                 Thread-local progress sink powering the live UI
  graph.py                  LangGraph wiring: nodes, edges, validation loop
  export.py                 Markdown report export
  pdf.py                    Markdown -> sanitized HTML -> styled PDF
  tools/search.py           Tavily search with DuckDuckGo + page-fetch fallback
  agents/manager.py         Task decomposition -> sub-questions + outline
  agents/researcher.py      Web search + sourced fact extraction
  agents/writer.py          Cited Markdown synthesis (handles revisions)
  agents/fact_checker.py    Citation integrity + LLM claim validation
  web/app.py                FastAPI backend + server entry point (`main()`)
  web/static/               Frontend: index.html, styles.css, app.js
reports/                    Generated reports (.md + .pdf), created on first run
```

See [HANDOVER.md](HANDOVER.md) for the full architecture walkthrough,
implementation notes, and troubleshooting history.

## Tuning

In `.env`:
- `GROQ_MODEL` — primary chat model (default `openai/gpt-oss-120b`)
- `GROQ_FALLBACK_MODEL` — auto-failover model with its own daily quota
  (default `openai/gpt-oss-20b`; leave empty to disable)
- `MAX_REVISIONS` — fact-check/rewrite loops before shipping with warnings (default 2)
- `MAX_SEARCH_RESULTS` — web results per sub-question (default 5)
