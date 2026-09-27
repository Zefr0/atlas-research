# Atlas — Multi-Agent Research Assistant

Atlas transforms open-ended research questions into structured, cited reports through a coordinated multi-agent workflow.

The system decomposes a question into focused research tasks, gathers evidence from the web, synthesizes the findings into a report, and validates the result through an automated fact-checking loop.

## Architecture

Atlas is built around a stateful LangGraph workflow. Each agent has a focused responsibility and communicates through a shared typed research state.

```text
User Question
      │
      ▼
Manager Agent
Decomposes the question into focused sub-questions
and defines the report outline
      │
      ▼
Researcher Agent
Searches the web and extracts source-backed findings
for each sub-question
      │
      ▼
Writer Agent
Synthesizes the findings into a structured,
citation-aware report
      │
      ▼
Fact-Checker Agent
Reviews claims, citations, and factual consistency
      │
      ├── Issues found ──► Writer Agent
      │
      └── Approved ──────► Final Report
```

The workflow is iterative rather than one-shot. When the fact-checker identifies unsupported claims, weak citations, or inconsistencies, the draft is returned to the writer with structured feedback. The process continues until the report passes validation or reaches its configured revision limit.

## Agent Responsibilities

### Manager Agent

Converts a broad research question into focused sub-questions and defines the structure of the final report.

### Researcher Agent

Searches for relevant sources, retrieves page content, and produces findings linked to their originating sources.

### Writer Agent

Combines the research findings into a coherent report with headings, explanations, and inline citations. It can revise the report using fact-checking feedback.

### Fact-Checker Agent

Reviews the report for citation integrity, unsupported claims, factual concerns, and inconsistencies. It decides whether the report is ready or requires another revision cycle.

## Shared Research State

Agents communicate through a typed shared state object containing:

- The original research question
- Generated sub-questions and report outline
- Search results and extracted findings
- The current report draft
- Fact-checking feedback
- Revision count and workflow metadata

Keeping this information in an explicit state object makes the workflow observable, testable, and easier to extend with additional agents or validation stages.

## Key Technical Features

### Stateful Graph Orchestration

LangGraph manages agent execution, state transitions, conditional routing, and the writer–fact-checker revision loop.

### Structured Model Outputs

Pydantic schemas define the expected shape of intermediate agent responses. This provides validation between workflow stages and reduces ambiguity when agents exchange information.

### Evidence-Grounded Research

Atlas uses Tavily for research-oriented search and supports DuckDuckGo as a fallback. Search results are supplemented with page-content retrieval so that reports can be grounded in source material rather than search snippets alone.

### Model Reliability

Model calls include retry handling and configurable model fallback. The workflow can continue with a secondary model when the primary model is unavailable or reaches a usage limit.

### Real-Time Progress Events

The backend emits progress events while a research task is running. The browser interface displays the active workflow stage, agent activity, and completed report as execution proceeds.

### Report Export

Completed reports can be viewed in the browser and exported as Markdown or styled PDF documents.

## Project Structure

```text
src/
├── agents/
│   ├── manager.py
│   ├── researcher.py
│   ├── writer.py
│   └── fact_checker.py
├── tools/
│   └── search.py
├── web/
│   ├── app.py
│   └── static/
├── config.py
├── events.py
├── graph.py
├── llm.py
├── schemas.py
├── state.py
├── export.py
└── pdf.py
```

## End-to-End Workflow

1. A user submits a research question through the web interface.
2. The manager creates focused sub-questions and a report outline.
3. The researcher gathers source-backed findings for each sub-question.
4. The writer creates a cited report from the collected evidence.
5. The fact-checker evaluates the draft and returns structured feedback.
6. The writer revises the report when validation issues are found.
7. The validated report is displayed and made available for export.

## Technology

- Python
- LangGraph
- LangChain Core
- Groq
- FastAPI
- Pydantic
- Tavily
- DuckDuckGo Search
- Markdown
- xhtml2pdf
- HTML, CSS, and JavaScript
