# HANDOVER — Multi-Agent Autonomous Research Assistant

> **Purpose of this document.** A complete handover for any engineer or AI agent taking over
> this project, and a learning companion for the project owner. It covers the project's
> context and history, the full architecture, how every module is implemented and *why*,
> every production incident hit during development and how it was diagnosed and fixed,
> and an operational runbook. Read top to bottom to learn the system; jump to
> [Troubleshooting History](#8-troubleshooting-history--the-most-instructive-part) when something breaks —
> odds are it broke here first.

---

## 1. What this project is

A production-style research assistant where **four LLM agents cooperate through a state
graph** to answer a research question with a fully cited report:

1. **Manager** — decomposes the user's question into 3–5 searchable sub-questions and designs the report outline.
2. **Researcher** — runs live web searches per sub-question and extracts atomic, source-attributed facts.
3. **Writer** — synthesizes the facts into a structured Markdown report with numbered citations.
4. **Fact-Checker** — validates every claim and citation; failed drafts loop back to the Writer (bounded).

Outputs: a Markdown report + a styled PDF in `reports/`, delivered through a **web app**
with live agent progress streaming. (The project had a CLI entry point until 2026-08-28;
it was deliberately removed so there is exactly one entry point to maintain.)

**Environment**: Windows 11, Python 3.12 (installed via winget during this project — the
machine originally had only the Microsoft Store stub), venv at `.venv/`. Not a git repository
(as of handover).

**Owner's context**: the owner is learning — explanations were requested alongside code.
When modifying this project, explain what you change and why. Code comments are deliberately
educational; keep that style.

---

## 2. Quick start (runbook)

```powershell
# One-time setup (already done on this machine)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env          # then paste GROQ_API_KEY into .env

# Start the app -> http://localhost:8010
.\.venv\Scripts\python.exe -m src.web.app

# Equivalent, if you want uvicorn flags directly
.\.venv\Scripts\python.exe -m uvicorn src.web.app:app --port 8010
```

`HOST` / `PORT` env vars override the defaults. HOST defaults to `127.0.0.1` deliberately:
**there is no authentication**, so anyone who can reach the port can spend the Groq quota.

**API keys** (`.env`, loaded by `src/config.py`):

| Variable | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | **yes** | LLM inference. Free key: https://console.groq.com/keys |
| `GROQ_MODEL` | no | Primary model. Default `openai/gpt-oss-120b` |
| `GROQ_FALLBACK_MODEL` | no | Auto-failover model. Default `openai/gpt-oss-20b`; empty disables |
| `TAVILY_API_KEY` | no | Better search. Without it: DuckDuckGo + page fetching (works, weaker) |
| `MAX_REVISIONS` | no | Fact-check → rewrite loops before shipping with warnings. Default 2 |
| `MAX_SEARCH_RESULTS` | no | Web results per sub-question. Default 5 |

**Important operational facts:**
- Groq free tier: **200,000 tokens/day PER MODEL** (that's why there's a fallback model — separate budget).
  A research run costs roughly 30–50k tokens. Reasoning tokens count toward the quota.
- The server must be **restarted after code changes** (uvicorn without `--reload` loads code once).
- `[winerror 10048]` on startup = something already on that port (usually a previous server instance).

---

## 3. Tech stack and why each piece was chosen

| Layer | Choice | Why (and what was rejected) |
|---|---|---|
| Agent framework | **LangGraph** | The Fact-Checker→Writer revision loop is a *cycle with a deterministic exit condition*. LangGraph models that natively (conditional edges + counter in state). CrewAI was rejected: faster to demo, but loop control is framework-internal and opaque. |
| LLM provider | **Groq** | Fast inference (matters: ~10-15 LLM calls per run, latency compounds), free tier. Models: `openai/gpt-oss-120b` primary, `openai/gpt-oss-20b` fallback. **History**: original pick was `llama-3.3-70b-versatile`; Groq retired all Llama models (see incident #1). |
| Search | **Tavily → DuckDuckGo fallback** | Tavily returns extracted page *content* (citable facts). DuckDuckGo (`ddgs`) is keyless but returns ~150-char snippets — useless for fact extraction until we added our own page fetching (incident #5). |
| Structured output | **Groq `json_schema` + `strict:true`** | Two incidents (#3, #8) taught us: function-calling is flaky with gpt-oss models, and non-strict json_schema is "best-effort" and intermittently invalid. Strict mode = server-side grammar constraint; malformed output becomes impossible. |
| Web backend | **FastAPI + SSE** | One-way progress streaming → SSE beats WebSockets (zero deps, auto-reconnect, proxy-friendly). Threads (not asyncio) run the pipeline because it's blocking HTTP calls. |
| Frontend | **Vanilla HTML/CSS/JS** | No build step; deploys as static files; full design control. Rejected Streamlit ("always looks like Streamlit") and React (toolchain overhead unjustified for one page). |
| PDF | **markdown + xhtml2pdf** | Pure Python. WeasyPrint needs GTK (painful on Windows); wkhtmltopdf needs a binary. Trade-off: xhtml2pdf supports only a CSS subset — keep `src/pdf.py` styles simple. |
| Validation/retry | **pydantic + tenacity** | Pydantic validates every structured LLM reply at the boundary. Tenacity gives the search tool exponential-backoff retries. |

---

## 4. Architecture

### 4.1 The execution graph (`src/graph.py`)

```
START ─▶ manager ─▶ researcher ─▶ writer ─▶ fact_checker ─▶ finalize ─▶ END
                        │            ▲            │
                        │            └─ revise ───┘   (conditional edge,
                        │                              max MAX_REVISIONS loops)
                        └─▶ END  (zero findings — never write from nothing)
```

The core mental model: **the state is a shared whiteboard** (`ResearchState`, a TypedDict in
`src/state.py`) that flows through the graph. Each node receives the current state and returns
a *partial update* (only the keys it owns); LangGraph merges it. Agents never call each other —
the Writer just reads `findings` that the Researcher left on the whiteboard. That's what keeps
them decoupled and independently testable.

```python
class ResearchState(TypedDict):
    query: str                  # user's question
    sub_questions: list[str]    # Manager
    report_outline: list[str]   # Manager
    findings: list[Finding]     # Researcher: {sub_question, fact, source_url}
    draft_report: str           # Writer (possibly several times)
    critique: list[str]         # Fact-Checker → read back by Writer
    revision_count: int         # drafts produced; bounds the loop
    verified: bool              # Fact-Checker verdict
    final_report: str           # finalize node
```

**Why the loop can never run away**: routing after the Fact-Checker is a plain Python function
(`_build_route_after_check`) — `verified → finalize`; `revision_count > MAX_REVISIONS → finalize
with warnings`; else `→ writer`. The LLM has no say in when the loop stops. `revision_count`
only ever increases. When a report ships unverified it gets a warning banner + the unresolved
critique appended — visible degradation, never silent failure and never an infinite loop.

### 4.2 File map

```
src/
  config.py                 .env loading; fail-fast validation; immutable Settings
  state.py                  ResearchState + Finding (the shared whiteboard)
  schemas.py                Pydantic contracts for structured LLM output (STRICT-mode compatible)
  llm.py                    Model factory + invoke_with_failover (THE reliability core)
  console.py                say()/banner() progress lines; Unicode-safe; feeds events.py
                            (prints to the server log AND streams to the browser)
  events.py                 Thread-local progress sink -> SSE bridge for the web UI
  graph.py                  LangGraph wiring; routing functions; finalize node
  export.py                 Markdown file export (timestamped slug filenames)
  pdf.py                    Markdown → HTML → PDF; Windows font workaround
  tools/search.py           Tavily → DDG fallback + page-content enrichment
  agents/manager.py         Decomposition (ResearchPlan)
  agents/researcher.py      Search + extraction + citation guard (ExtractedFacts)
  agents/writer.py          Synthesis + revision handling (plain text out)
  agents/fact_checker.py    2-layer validation (regex + LLM judge, Verdict)
  web/app.py                FastAPI: run registry, worker threads, SSE, downloads;
                            main() + __main__ = the application entry point
  web/static/               index.html / styles.css / app.js (no build step, dark only)
reports/                    Output: <UTCtimestamp>_<query-slug>.md + .pdf
.claude/launch.json         Dev-server config for Claude Code's browser preview
```

---

## 5. Module deep-dives (how everything is implemented)

### 5.1 `src/config.py` — configuration
- `load_dotenv()` reads `.env`; real environment variables win (12-factor convention).
- `Settings` is a **frozen dataclass** — no agent can mutate config at runtime.
- `load_settings()` raises `ConfigError` with *fix instructions* if `GROQ_API_KEY` is missing —
  the user sees "here's how to fix it", not an HTTP 401 stack trace three minutes in.
  Numeric envs are validated (`_positive_int`).
- **Lesson embodied**: validate the environment once, at the entry point, before any work.

### 5.2 `src/schemas.py` — the contracts between agents
Every agent whose output feeds another agent returns **schema-validated JSON**, never free text.
The schemas are the API between agents:
- `ResearchPlan { sub_questions, report_outline }` — Manager output.
- `ExtractedFacts { facts: [{fact, source_url}] }` — Researcher output.
- `Verdict { passed, critique }` — Fact-Checker output.

Three non-obvious rules (all learned from production failures — incident #8):
1. **Field `description=` is sent to the model** as part of the schema. It's a micro-prompt.
   Write them as instructions ("Must be copied EXACTLY", "MUST be an empty list when passed is True").
2. **Every field must be required** (no defaults) — Groq strict mode rejects optional fields.
3. **No `min_length`/`max_length` constraints** — strict mode supports only a JSON-Schema subset.
   Enforce emptiness/bounds in Python code instead.

### 5.3 `src/llm.py` — the reliability core (most battle-tested file)
Three public functions:
- `chat(settings, temperature, effort)` → list of `(name, model)` pairs: primary then fallback.
- `structured_chat(settings, schema, temperature, effort)` → same, schema-bound with
  `with_structured_output(schema, method="json_schema", strict=True)`.
- `invoke_with_failover(models, messages)` → the invocation brain. Per model:
  1. Try the call. **Malformed generations** (`tool_use_failed`, `json_validate_failed`,
     parser errors) are retried up to 3× with a corrective system message appended (the "nudge").
  2. On a **rate limit**, parse Groq's own relief hint from the error text
     (`"try again in 7.66s"` / `"19m29s"` — regex `_RETRY_AFTER_RE`). Short wait (≤90s =
     per-MINUTE limit) → sleep it out, retry the *same* model. Long wait (= per-DAY quota)
     → move to the *next* model.
  3. Every wait/switch is announced via `say()` so both CLI and web UI show which model worked.

**Why not LangChain's `.with_fallbacks()`** (incident #7): when the fallback *also* fails, it
re-raises the primary's error and swallows the fallback's — undebuggable in production. The
explicit loop gives control and observability.

**`reasoning_effort`**: gpt-oss are reasoning models; their hidden reasoning tokens count
against the daily quota. Mechanical calls (fact extraction) run at `effort="low"` — the single
biggest token saving (5 large calls per run). Judgment calls (fact-check) keep the default.

### 5.4 `src/tools/search.py` — search with two tiers + enrichment
- **Tier 1 Tavily** (if key present): returns extracted page content, ready for fact extraction.
  Tenacity retries (3×, exponential backoff, max 10s).
- **Tier 2 DuckDuckGo** (`ddgs`): keyless, but snippets are ~150-char page *descriptions* with
  no facts in them (incident #5 — "kept 0 sourced facts" across every sub-question). Fix:
  `_enrich_with_page_text()` downloads the **top 4 result pages** and extracts visible text with
  a ~60-line stdlib `HTMLParser` subclass (`_TextExtractor`): collects text from content tags
  (`p, li, h1-h4, td, th…`), skips `script/style/nav/footer/form`. Guardrails: 10s timeout,
  HTML-only, 500KB read cap, browser-ish User-Agent (default Python UA gets 403'd), and any
  fetch failure just keeps the snippet — enrichment can degrade but never break a search.
- Both tiers normalize to the same `SearchResult {title, url, content}` shape — the Researcher
  never knows which engine answered (tool abstraction: agents depend on an interface, not a vendor).

### 5.5 `src/agents/*` — the four agents
All follow the same **factory pattern**: `build_X_node(settings)` returns a closure
`state -> partial_update`. The graph needs that exact signature; the closure captures settings
and pre-built models once.

**Manager** (`manager.py`, temp 0.3): one structured call → `ResearchPlan`. The system prompt
ends with "Never ask for clarification and never refuse" — because a model that answers a vague
query with a question breaks structured output entirely (incident #3).

**Researcher** (`researcher.py`, temp 0.0, effort low): per sub-question —
1. `search_web()` (failure → log and skip that sub-question, never kill the run),
2. one structured call extracting facts from the numbered results (content truncated to 3000
   chars per result to bound the prompt),
3. **the deterministic citation guard** — the most important anti-hallucination device in the
   system: `valid_urls = {r["url"] for r in results}`; any extracted fact whose `source_url`
   is not in that set is discarded *by Python*, with a "DISCARDED fact with invented URL" log
   line. The LLM physically cannot smuggle an invented citation into the evidence pool,
   because code — not another LLM — enforces the rule.

**Writer** (`writer.py`, temp 0.4): evidence is numbered `[1]..[n]` **in Python** before the
model sees it; the model may only cite those numbers, and the Sources appendix is generated
later from the same list — numbering literally cannot drift. Runs multiple times: on revision,
the rejected draft + the critique are injected ("Fix ALL of these problems (change nothing that
wasn't flagged)") — targeted repair beats rewriting. Returns `revision_count = draft number`.

**Fact-Checker** (`fact_checker.py`, temp 0.0): two layers, cheapest first —
1. *Deterministic* (`_check_citation_integrity`): regex all `[n]` citations; any `n` outside
   `1..len(findings)` is proof of hallucination — zero tokens spent, and no LLM can be
   sweet-talked out of it. Also fails a report with no citations at all.
2. *LLM-as-judge*: a **fresh** temperature-0 call (not the Writer's conversation — a model
   reviewing its own output is biased toward approving it) with a skeptical persona, comparing
   every claim against the evidence → `Verdict {passed, critique}`.

### 5.6 `src/graph.py` — assembly
`build_graph(settings)` registers the four agent nodes + `finalize`, wires edges, compiles.
`_finalize_node` assembles the shippable artifact: warning banner (if unverified) + draft +
Sources appendix generated from `findings` (same numbering the Writer cited) + unresolved
critique section (if shipping unverified). `initial_state(query)` builds a fresh whiteboard.

### 5.7 Output: `export.py` and `pdf.py`
- `export.py`: `reports/<UTC timestamp>_<slugified query>.md`, UTF-8 explicitly (Windows'
  default codepage mangles non-ASCII).
- `pdf.py`: Markdown → HTML (`markdown` lib, `tables` + `sane_lists` extensions) → PDF
  (`xhtml2pdf`). Two Windows specifics: **(a)** embeds `C:\Windows\Fonts\arial.ttf` via
  `@font-face` for Unicode coverage (built-in Helvetica is Latin-1 only; falls back gracefully
  if the font is missing); **(b)** a **required monkeypatch**
  `xhtml2pdf.files.pisaFileObject.getNamedFile = lambda self: self.uri` — without it, xhtml2pdf
  copies the font to a temp file that Windows locks, and every export fails (incident #6).
  PDF export returns bool; a PDF failure warns but never fails a run that already succeeded.

### 5.8 The web layer: `events.py`, `web/app.py`, `web/static/`
**Progress plumbing** (`events.py` + `console.py`): every agent logs via `console.say()`, which
(a) prints — with `errors="replace"` stream reconfiguration so a Unicode char can't crash a run
on legacy Windows codepages (incident #4) — and (b) forwards to a **thread-local sink** if one
is registered. The CLI registers none (zero behavior change). The web server runs each research
job in a worker thread and registers a sink that pushes into that run's queue. Thread-local
(not global) means two browser tabs can run two jobs concurrently without crossing streams.

**Backend** (`web/app.py`):
- `POST /api/research` → validates config *first* (missing key = clean 503, not a doomed run),
  creates a run in the in-memory `_runs` registry, spawns a daemon worker thread, returns `run_id`.
- `GET /api/runs/{id}/events` → **SSE stream**: a generator reads the run's queue; 15s of
  silence emits an SSE comment (`: ping`) as heartbeat; the worker's `None` sentinel sends the
  final `done` event. Headers: `Cache-Control: no-cache`, `X-Accel-Buffering: no` (nginx-safe).
- `GET /api/runs/{id}` → status + server-rendered report HTML + verified flag.
- `GET /api/runs/{id}/download/{md|pdf}`, `GET /api/reports` (history),
  `GET /api/reports/{filename}/file` — the latter has a **path-traversal guard**
  (resolved path must be directly inside `reports/`, extension whitelisted).
- Static frontend mounted **last** so `/api/*` wins routing.
- Scaling note: `_runs` is in-memory — fine for single instance; swap for Redis if scaling out.

**Frontend** (`web/static/`, no build step):
- `styles.css`: token-based design system (`--bg, --surface, --accent…`), light theme on
  `:root`, dark via **both** `[data-theme="dark"]` (manual toggle, persisted in localStorage)
  and `prefers-color-scheme` guarded with `:root:not([data-theme="light"])` — so OS-follow and
  manual override coexist. Motion respects `prefers-reduced-motion`; focus-visible outlines;
  responsive to phone width; wide report tables scroll inside their own container.
- `app.js` (vanilla): submit → POST → `EventSource`; log lines appended with `textContent`
  (never `innerHTML` — log text is untrusted); stepper derives the active agent from log
  events (the Writer↔Fact-Checker loop naturally re-activates steps); `done` → fetch run,
  inject server-rendered report HTML, wire download links, refresh history.

### 5.9 Frontend design language (`web/static/styles.css`)

The UI was reworked on 2026-08-28 to a **dark-only** design. There is no light theme and no
theme toggle — committing to one mode removed a whole class of token duplication. The rules
that give it a recognisable identity, worth preserving if you extend it:

- **No glow, anywhere.** No coloured `box-shadow`, no blurred halo behind buttons. Depth comes
  from three cheap devices instead: hairline borders (`--line`), stacked flat surfaces
  (`--surface` → `--surface-3`), and a 1px top highlight on cards (`.panel::before`).
- **Primary actions are solid white on near-black.** Maximum contrast, zero decoration. The
  accent (`--accent`, a signal lime) is reserved for *live state only*: the running pill, the
  active pipeline step, the blinking eyebrow dot. If everything is accented, nothing is.
- **Editorial type.** Oversized display sizes with tight tracking (`-0.045em`) and a dimmed
  second line; monospace (JetBrains Mono) for anything machine-generated — the log, step
  indices, status pills, badges, history file links.
- **Texture, not bloom.** The fixed `.backdrop` layers a cold blue wash at the horizon with a
  68px architectural grid masked to fade out. It sits behind the whole page, never behind a
  single element.
- The pipeline steps are numbered `01`–`04` in mono, which reads as a machine process rather
  than a marketing stepper.

`app.js` wraps every rendered `<table>` in `.table-scroll` so wide tables scroll inside their
own box while still filling the column when they fit — the page itself never scrolls sideways.

---

## 6. The design patterns worth learning (teaching section)

1. **Deterministic guards beat LLM judgment wherever possible.** The two strongest
   anti-hallucination devices here are plain Python: the URL whitelist in the Researcher and
   the citation-range regex in the Fact-Checker. Use LLM judgment only for what code can't
   check (claim↔evidence support). Cheap deterministic layer first — it also saves tokens.

2. **Structured outputs are contracts, enforced at the boundary.** Free text between agents
   = parsing bugs waiting to happen. Schema + validation at the call site means a bad
   generation fails *loudly at the source* instead of poisoning downstream agents.

3. **Bound every loop with a counter the LLM can't touch.** `revision_count` + a plain-code
   routing function = guaranteed termination with graceful degradation (ship with warnings).

4. **Visible degradation beats silent failure.** Unverified reports ship with a banner and
   the unresolved critique attached. Failed PDF → warning, Markdown still delivered. A skipped
   sub-question → logged. Nothing pretends to have succeeded.

5. **Failover needs observability.** The `.with_fallbacks()` episode: a reliability wrapper
   that hides *which* thing failed makes production debugging impossible. Explicit loops with
   log lines ("switching to X") cost 30 lines and pay for themselves the first bad day.

6. **Rate limits are not one thing.** Per-minute (relief in seconds → wait it out, same model)
   vs per-day (relief in hours → switch models). Parse the provider's relief hint; don't
   blind-retry.

7. **Normalize at tool boundaries.** Tavily and DDG return different shapes; one
   `SearchResult` type means the Researcher never cares. Swapping search vendors is a
   one-file change.

8. **Thread-local context for cross-cutting concerns.** Progress streaming needed by the web
   UI but not the CLI: a thread-local sink adds it without touching a single agent signature.

9. **Closures as dependency injection.** `build_X_node(settings)` → the graph gets the exact
   `state -> update` signature it needs; the node gets its dependencies without globals.

10. **Fail fast, with instructions.** Config errors at startup tell the user *how to fix it*.

---

## 7. Costs, quotas, and performance

- **Per run**: ~10–15 LLM calls (1 manager + 5 extraction + 1–3 writer + 1–3 fact-checker),
  ~30–50k tokens total, 1–3 minutes wall-clock (search fetches + Groq inference).
- **Groq free tier**: 200k tokens/day *per model*; per-minute token caps also exist.
  The primary+fallback pair ≈ 400k/day effective. Reasoning tokens count (hence
  `effort="low"` on extraction). If both models exhaust: wait for the rolling window, or
  Groq Dev Tier.
- **Tavily free tier**: ~1,000 searches/month ≈ 200 runs. **The single biggest quality
  upgrade available is adding a `TAVILY_API_KEY`** — DDG+enrichment works but Tavily's
  extraction is better and faster (no page fetches).

---

## 8. Troubleshooting history — the most instructive part

Every incident actually hit during development, in order. Format: **symptom → root cause → fix → lesson**.

### #1 — 404 `model_not_found` on first run
- **Symptom**: `The model 'llama-3.3-70b-versatile' does not exist or you do not have access to it.`
- **Cause**: Groq retired all Llama models between design and first run.
- **Fix**: queried the account's live catalog (`Groq().models.list()`), picked
  `openai/gpt-oss-120b` (strongest chat model with tool-use/structured support).
- **Lesson**: never hardcode-and-assume a hosted model id; check the live catalog. Model ids
  are configuration, not constants (`GROQ_MODEL` env).

### #2 — 400 `tool_use_failed`: model replied in prose
- **Symptom**: crash when the query was vague placeholder text; Groq error contained the
  model's *prose* reply asking for clarification.
- **Cause**: structured output via tool-calling requires the model to call a tool; the model
  chose to "answer" with a clarifying question instead.
- **Fix**: Manager prompt: "Never ask for clarification and never refuse" + retry-with-nudge.
- **Lesson**: any agent whose output is parsed must be prompted so that *every* input yields
  parseable output. Refusal/clarification is a failure mode of structured pipelines.

### #3 — 400 `tool_use_failed`: misspelled tool name (`_extracteddfacts`)
- **Symptom**: intermittent crash in the Researcher; `failed_generation` showed a tool name
  that didn't exist, with empty arguments.
- **Cause**: gpt-oss models are flaky at function-calling: they garble tool names / emit empty args.
- **Fix**: switched structured output to `method="json_schema"` (Groq **constrained decoding** —
  the server constrains generation so output must match the schema; no tool names involved).
- **Lesson**: with Groq + gpt-oss, always use `json_schema`, never `function_calling`.

### #4 — `UnicodeEncodeError` killed a web run
- **Symptom**: worker thread died: `'charmap' codec can't encode character '\u2011'`.
- **Cause**: an agent message contained a non-breaking hyphen; `print()` under uvicorn on
  Windows used legacy cp1252, which can't encode it.
- **Fix**: `console.py` reconfigures stdout/stderr with `errors="replace"` at import.
- **Lesson**: on Windows, never let console encoding be able to throw in a worker. LLM/web text
  *will* contain characters your codepage doesn't have.

### #5 — "kept 0 sourced fact(s)" on every sub-question
- **Symptom**: runs completing with zero findings → "No report produced". Search returned
  results; the extractor honestly found nothing in them.
- **Cause**: DuckDuckGo snippets are ~150-char page *descriptions* ("List of foreign
  players — Players in bold…") containing no extractable facts. Earlier successes were luck.
- **Fix**: `_enrich_with_page_text()` — fetch top-4 result pages, extract real text (stdlib
  HTML parser), swap in when longer than the snippet. Went from 0 findings to 50 on the same query.
- **Lesson**: fact extraction is only as good as the text you feed it. Diagnose by *looking at
  the actual search payloads*, not the extraction prompt.

### #6 — PDF export failed on every attempt (Windows)
- **Symptom**: `xhtml2pdf` error chain ending in `Can't open file "...\Temp\tmpXXXX.ttf"`.
- **Cause**: known xhtml2pdf Windows bug — it copies each `@font-face` file to a
  `NamedTemporaryFile` it then can't reopen (Windows locks it).
- **Fix**: monkeypatch in `pdf.py`: `pisaFileObject.getNamedFile = lambda self: self.uri`
  (serve the original path; skip the temp copy).
- **Lesson**: keep the monkeypatch when upgrading xhtml2pdf; retest PDF export on Windows.

### #7 — Rate-limited run died *despite* failover existing
- **Symptom**: 429 TPD (daily quota) error killed a run; failover had worked for the Manager
  call but the Researcher call died one step later. Error shown was the *primary's* 429.
- **Cause**: two-part. (a) Daily quota exhausted on the primary (200k/day — all of the day's
  testing). (b) Failover used LangChain `.with_fallbacks()`, which — when the fallback also
  fails (here: the smaller model's per-*minute* cap right after a burst) — re-raises the
  **first** (primary's) error and hides the fallback's. Undebuggable.
- **Fix**: replaced with explicit `invoke_with_failover`: parse Groq's "try again in Xs" hint;
  short wait → sleep + retry same model; long wait → next model; log every switch.
- **Lesson**: distinguish per-minute from per-day 429s; never use a failover wrapper that
  swallows the fallback's error.

### #8 — 400 `json_validate_failed` on the fallback model
- **Symptom**: Fact-Checker call on `gpt-oss-20b` failed Groq JSON validation 3×, empty
  `failed_generation` → run died after nudge retries.
- **Cause** (confirmed via Groq docs + community forum): *without* `strict:true`, Groq
  structured outputs run in "best-effort" mode which intermittently returns schema-violating
  JSON (~10% reports on gpt-oss). Additionally our `Verdict.critique` had a default (optional
  field) and schemas used `min_length` — both incompatible with strict mode's requirements
  (all fields required, `additionalProperties:false`, JSON-Schema subset only).
- **Fix**: `strict=True` on every `with_structured_output`; made `critique` required ("MUST be
  an empty list when passed is True" in its description); removed length constraints from schemas.
- **Lesson**: Groq strict mode is the only reliable structured output for gpt-oss; design
  schemas for it from the start. Sources: console.groq.com/docs/structured-outputs,
  community.groq.com threads on `json_validate_failed`.

### #9 — Token quota burned faster than expected
- **Symptom**: 200k/day gone in ~5 runs.
- **Cause**: gpt-oss reasoning tokens (hidden chain-of-thought) count against the quota;
  extraction calls have the biggest prompts and ran at default (medium) effort.
- **Fix**: `reasoning_effort="low"` for extraction (mechanical work); default kept for
  judgment tasks (fact-check).
- **Lesson**: with reasoning models, effort is a cost dial. Match it to task difficulty.

### #10 — `[winerror 10048]` when starting the server
- **Symptom**: uvicorn fails to bind port 8010.
- **Cause**: a previous server instance already listening (only one process per port).
- **Fix/diagnosis**: `Get-NetTCPConnection -LocalPort 8010 -State Listen` → find/stop the PID,
  or use another port. Also remember: a *running* server keeps executing **old code** until
  restarted — two incidents were misdiagnosed at first because the fix was on disk but the
  process predated it. Compare process start time vs file mtime when "the fix didn't work".

### #11 — Fact-Checker rejected fully-cited reports ("no [n] citations at all")
- **Symptom**: reports with dozens of working citations were failed by Layer 1 with
  "The report contains no [n] citations at all", burning all 3 drafts and shipping a
  "Needs review" banner on sound work. Simultaneously, most citations were never validated.
- **Cause**: `_CITATION_RE` matched only ASCII `[n]` / `[n, m]`. The gpt-oss Writer also emits
  CJK lenticular brackets (`【7】`) and ranges joined by a non-breaking hyphen (`[5‑7]`, U+2011).
  64 such citations existed across the shipped reports. Both failure directions followed:
  false "no citations" verdicts, and out-of-range numbers hiding inside unmatched forms — in
  one report 54 of 63 citations were never range-checked, so a hallucinated `【99】` would pass
  the layer that exists precisely to be un-foolable.
- **Fix**: `_cited_numbers()` now accepts both bracket families, comma lists and ranges with
  ASCII/Unicode dashes, validating both endpoints of a range (evidence numbering is contiguous,
  so in-bounds endpoints imply in-bounds middles). The Writer prompt also asks for plain ASCII
  brackets — belt and braces, since the checker no longer depends on compliance.
- **Lesson**: this was only findable by testing the code against **real saved output**. Reading
  the regex alone, it looks correct. When a validator and a generator disagree about format,
  the artifacts on disk are the evidence.

### #12 — SSE stream hung forever on reconnect or late connect
- **Symptom**: the browser sat on "Running" with the submit button disabled until a reload,
  even though the run had finished.
- **Cause**: the progress queue is a *destructive* consumer with exactly one `None` sentinel.
  An EventSource reconnect created a second generator racing the zombie for events; whoever
  took the sentinel won, and the other waited on an empty queue emitting heartbeats forever.
  Connecting after a run finished had the same effect.
- **Fix**: the run's `status` is now the authority — emit `done` immediately if the run is not
  running, and re-check whenever the queue goes quiet. Also, `app.js` `onerror` now handles
  `readyState === CLOSED` (a permanent failure, e.g. the server restarted) instead of ignoring it.
- **Lesson**: a queue is a poor source of truth for "is this finished?" — durable state is.

### #13 — Hidden download button stayed visible
- **Symptom**: when PDF export failed, the "PDF" button still showed and downloading it saved
  the server's JSON 404 body as a file.
- **Cause**: `el.hidden = true` relies on the UA rule `[hidden] { display: none }`, which any
  author `display` declaration outranks — and `.primary-btn` sets `display: inline-flex`.
- **Fix**: `[hidden] { display: none !important; }` in `styles.css`.
- **Lesson**: if you toggle `.hidden` in JS, make the attribute authoritative in CSS.

### #14 — `json_validate_failed` again, this time killing a whole run
- **Symptom**: on draft #3 of a long report, the Fact-Checker failed structured output three
  times on the primary model, failed over to the fallback, failed three more times there, and
  the RuntimeError propagated out of the graph — destroying a finished 5,918-character draft.
  `failed_generation` was an **empty string** each time.
- **Cause** (proven, not guessed): a structured reply that does not finish within the
  completion budget is rejected wholesale by Groq with exactly this error. Reproduced on demand
  by capping `max_completion_tokens` — 64, 256 and 1024 all produce
  `json_validate_failed`. The Fact-Checker had been writing enormous critiques (multi-sentence
  items quoting whole passages of the draft back), and on gpt-oss models the hidden reasoning
  tokens come out of the same budget. Note that strict mode (incident #8) does not help here:
  the JSON is *valid so far*, just unfinished.
- **Fix**, three layers:
  1. Bound the output: the system prompt now caps the critique at 5 items of one sentence each
     and forbids quoting long passages.
  2. A terse retry: on failure, re-ask through a `reasoning_effort="low"` chain with a
     "three items, 120 characters" nudge, leaving far more budget for the answer.
  3. Graceful degradation: if that also fails, the node returns `check_failed=True` instead of
     raising. `route_after_check` finalizes immediately (another draft would only be judged by
     the same broken checker) and the report ships with its own honest banner —
     *claim-checking could not be completed*, distinct from *checked and found wanting*.
- **Verified**: re-running the failing query reproduced the failure on draft #2 and the terse
  retry recovered it; the report then passed verification.
- **Lesson**: the general principle is worth more than the fix — **a validation step must never
  be able to destroy the artifact it was asked to validate.** Ask of every step: if this fails,
  what work do we lose? The Manager and Writer must fail hard (no plan, no draft, no report);
  the Researcher and Fact-Checker must degrade.

---

## 9. Known limitations & sensible next steps

1. **No Tavily key configured** — the owner runs on DDG+enrichment. Adding the key is the
   biggest quality win available (better evidence → more findings → fewer 0-fact sub-questions).
2. **Fact-Checker pedantry**: it sometimes rejects drafts for uncited *topic-framing* sentences
   ("CSL clubs pursued a strategy of…") and burns the revision budget on them, shipping
   "Needs review" reports whose facts are actually solid. Tuning idea: allow uncited framing
   sentences while staying strict on figures/names/claims (prompt change in `fact_checker.py`).
3. **In-memory run registry** (`web/app.py _runs`): lost on restart, single-instance only, and
   **never evicted** — each finished run keeps its rendered HTML and event queue for the life of
   the process. Fine locally; use Redis/DB (and eviction) for real deployment. Runs also aren't
   rate-limited per client.
4. **No auth on the web app** — anyone who can reach the port can spend your Groq quota.
   Add auth before exposing beyond localhost.
5. **No tests** — the pipeline was verified end-to-end manually. Highest-value unit tests:
   `_check_citation_integrity`, `_retry_after_seconds`, the URL guard in the Researcher,
   `_TextExtractor`, and the path-traversal guard in `report_file`.
6. **Not a git repo** — `git init` recommended; `.gitignore` already present (excludes `.env`,
   `.venv/`, `reports/`, `__pycache__/`).
7. **Provider swap**: Gemini was the planned alternative. Implement inside `llm.py` only
   (`_groq` → a factory keyed on provider); nothing else changes.
8. **Deliberately not fixed** (reviewed, judged not worth the complexity for a single-user
   local tool — revisit before any multi-user deployment): SSE `Last-Event-ID` replay,
   graceful-shutdown teardown of idle streams, atomic report writes (a hard kill mid-write can
   leave a truncated file that the history panel still lists), gzip/charset handling in
   `_fetch_page_text`, and FastAPI 422 bodies rendering as `[object Object]` in the UI.
9. **Progress streaming assumes sequential nodes**: `events.py` registers a *thread-local* sink,
   which works only because LangGraph runs our single-task supersteps inline. Add a parallel
   branch and the browser log silently goes blank for those nodes (the server log still prints).
   Documented at the top of `src/events.py`.

---

## 10. Glossary (for the owner)

- **LangGraph node/edge**: a node is a function `state -> partial state update`; edges say what
  runs next; conditional edges pick the next node by inspecting state in plain Python.
- **Structured output / constrained decoding**: the inference server restricts which tokens the
  model may emit so the reply must parse as JSON matching your schema (vs hoping the model
  formats correctly).
- **SSE (Server-Sent Events)**: one-way HTTP stream (`text/event-stream`) the browser consumes
  with `EventSource`; auto-reconnects; simpler than WebSockets when you never need client→server.
- **Thread-local**: a variable with a separate value per thread — how each web run gets its own
  progress channel without global state collisions.
- **TPD / TPM**: Groq's tokens-per-day / tokens-per-minute limits, tracked per model.
- **Nudge retry**: re-sending a failed structured request with an appended corrective system
  message ("respond ONLY with data matching the schema…").
- **Monkeypatch**: replacing a library's function at runtime to fix behavior you can't change
  upstream (used for xhtml2pdf's Windows temp-file bug).
