"""
Web frontend backend: a FastAPI app wrapping the research pipeline.

Architecture (deliberately deployment-friendly):

  browser ── POST /api/research ──▶ spawn worker thread, return run_id
  browser ── GET  /api/runs/{id}/events ──▶ Server-Sent Events stream of the
             live agent progress (the same messages the server logs)
  browser ── GET  /api/runs/{id} ──▶ final status + rendered report HTML
  browser ── GET  /api/runs/{id}/download/{md|pdf} ──▶ file downloads
  browser ── GET  /api/reports ──▶ past reports on disk (run history)

Why SSE and not WebSockets: progress is strictly one-way server->browser,
SSE needs zero extra dependencies, auto-reconnects natively in the browser,
and passes through proxies/load balancers without special configuration.

Why threads and not asyncio for the pipeline: the LangGraph run is a chain
of blocking HTTP calls (LLM + search). One worker thread per run keeps the
event loop free; the thread-local sink in events.py keeps concurrent runs'
progress streams separate.

Run locally (either form works):
    .venv/Scripts/python.exe -m src.web.app
    .venv/Scripts/python.exe -m uvicorn src.web.app:app --port 8010
Deploy behind any ASGI host; HOST/PORT env vars are honoured by the __main__
block at the bottom of this file.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import events
from ..config import ConfigError, load_settings
from ..export import export_report
from ..graph import build_graph, initial_state
from ..pdf import export_pdf, render_report_html

_STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Multi-Agent Research Assistant", docs_url="/api/docs")

# In-memory run registry: {run_id: {...}}. Fine for a single-instance deploy;
# swap for Redis/DB if this ever needs horizontal scaling.
_runs: dict[str, dict] = {}
_runs_lock = threading.Lock()


class ResearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=500)


# ---------------------------------------------------------------- worker ---
def _research_worker(run_id: str, query: str) -> None:
    """Runs one research job in its own thread, streaming progress events."""
    run = _runs[run_id]
    q: queue.Queue = run["queue"]

    # Every console.say() from THIS thread now lands in this run's queue.
    events.set_sink(lambda agent, message: q.put({"type": "log", "agent": agent, "message": message}))
    try:
        settings = load_settings()
        graph = build_graph(settings)
        final = graph.invoke(initial_state(query), config={"recursion_limit": 50})

        if not final["final_report"]:
            run["status"] = "failed"
            run["error"] = (
                "No usable sources were found for this query. "
                "Try rephrasing it, or check your network / API quota."
            )
        else:
            md_path = export_report(settings.reports_dir, query, final["final_report"])
            pdf_path = md_path.with_suffix(".pdf")
            run["md_path"] = md_path
            if export_pdf(final["final_report"], pdf_path):
                run["pdf_path"] = pdf_path
            run["report_html"] = render_report_html(final["final_report"])
            run["verified"] = final["verified"]
            run["status"] = "completed"
    except ConfigError as exc:
        run["status"] = "failed"
        run["error"] = str(exc)
    except Exception as exc:
        run["status"] = "failed"
        run["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        events.set_sink(None)
        q.put(None)  # sentinel: closes the SSE stream


# -------------------------------------------------------------- endpoints ---
@app.post("/api/research")
def start_research(body: ResearchRequest) -> dict:
    """Kick off a research run; returns immediately with its id."""
    # Fail fast on configuration BEFORE spawning a thread, so a missing API
    # key is a clear 503 instead of a run that instantly dies.
    try:
        load_settings()
    except ConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    run_id = uuid.uuid4().hex[:12]
    with _runs_lock:
        _runs[run_id] = {
            "query": body.query.strip(),
            "status": "running",
            "queue": queue.Queue(),
            "error": None,
            "md_path": None,
            "pdf_path": None,
            "report_html": None,
            "verified": False,
        }
    threading.Thread(
        target=_research_worker, args=(run_id, body.query.strip()), daemon=True
    ).start()
    return {"run_id": run_id}


@app.get("/api/runs/{run_id}/events")
def run_events(run_id: str) -> StreamingResponse:
    """SSE stream of live progress for one run."""
    run = _runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Unknown run id")

    def done_event() -> str:
        return "data: %s\n\n" % json.dumps(
            {"type": "done", "status": run["status"], "error": run["error"]}
        )

    def stream():
        q: queue.Queue = run["queue"]
        # The queue is a DESTRUCTIVE consumer with exactly one None sentinel, so
        # a client that connects late — after the run ended, or after an
        # EventSource reconnect where a zombie generator already took the
        # sentinel — would wait on it forever, leaving the UI stuck on
        # "Running". The run's status is the authority, so check it up front
        # and again whenever the queue goes quiet.
        if run["status"] != "running":
            yield done_event()
            return
        while True:
            try:
                item = q.get(timeout=15)
            except queue.Empty:
                if run["status"] != "running":
                    yield done_event()
                    return
                yield ": ping\n\n"  # SSE comment = heartbeat, keeps proxies from timing out
                continue
            if item is None:  # worker finished (either way)
                yield done_event()
                return
            yield "data: %s\n\n" % json.dumps(item)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/runs/{run_id}")
def run_status(run_id: str) -> dict:
    """Final state of a run: status, verification badge, rendered report."""
    run = _runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Unknown run id")
    return {
        "query": run["query"],
        "status": run["status"],
        "error": run["error"],
        "verified": run["verified"],
        "report_html": run["report_html"],
        "has_pdf": run["pdf_path"] is not None,
    }


@app.get("/api/runs/{run_id}/download/{fmt}")
def run_download(run_id: str, fmt: str) -> FileResponse:
    run = _runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Unknown run id")
    path = {"md": run["md_path"], "pdf": run["pdf_path"]}.get(fmt)
    if path is None:
        raise HTTPException(status_code=404, detail=f"No {fmt} file for this run")
    return FileResponse(path, filename=path.name)


@app.get("/api/reports")
def list_reports() -> list[dict]:
    """Past reports on disk, newest first — powers the history panel."""
    try:
        reports_dir = load_settings().reports_dir
    except ConfigError:
        return []
    if not reports_dir.exists():
        return []
    out = []
    for md in sorted(reports_dir.glob("*.md"), reverse=True):
        out.append(
            {
                "name": md.stem,
                "md": md.name,
                "pdf": md.with_suffix(".pdf").name if md.with_suffix(".pdf").exists() else None,
            }
        )
    return out


@app.get("/api/reports/{filename}/file")
def report_file(filename: str) -> FileResponse:
    """Serve one past report file by name (history downloads)."""
    try:
        reports_dir = load_settings().reports_dir
    except ConfigError:
        raise HTTPException(status_code=503, detail="Not configured")
    # Path-traversal guard: the name must resolve to a file directly inside
    # reports/ — "..%2F..%2F.env" style names get rejected here.
    path = (reports_dir / filename).resolve()
    if path.parent != reports_dir.resolve() or path.suffix not in {".md", ".pdf"} or not path.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path, filename=path.name)


# Static frontend LAST so /api/* wins routing; html=True serves index.html at /.
app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")


def main() -> None:
    """Start the server. This is the application's only entry point.

    HOST defaults to 127.0.0.1 deliberately: the app has no authentication, so
    anyone who can reach the port can spend your Groq quota. Set HOST=0.0.0.0
    only once you have put authentication (or a trusted network) in front of it.
    """
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8010")),
    )


if __name__ == "__main__":
    main()
