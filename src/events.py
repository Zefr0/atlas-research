"""
Per-run progress event sink.

The agents report progress through console.say(), which prints to the server
log. The web UI ALSO needs those messages, streamed live to the browser.
Rather than threading a callback through every agent signature, we use a
thread-local sink: the web server runs each research job in its own worker
thread and registers a sink for that thread; console.say() forwards every
message to it. Code running without a sink registered just prints.

Thread-local (not global) matters: two browser tabs can run two research
jobs concurrently without their progress streams crossing.

CAVEAT — this works because LangGraph runs our nodes inline on the calling
thread. It does that whenever a superstep holds a single task, which is always
true for our strictly sequential graph. Add a parallel branch (two nodes in one
superstep) and LangGraph hands those to a thread pool, where this sink is
absent: emit() silently no-ops and the browser's live log goes blank for those
nodes while the server log keeps printing. If the graph ever gains parallel
branches, pass the sink through LangGraph's config instead.
"""

from __future__ import annotations

import threading
from typing import Callable

_local = threading.local()

Sink = Callable[[str, str], None]  # (agent, message) -> None


def set_sink(sink: Sink | None) -> None:
    """Register (or clear, with None) the progress sink for THIS thread."""
    _local.sink = sink


def emit(agent: str, message: str) -> None:
    """Forward one progress message to this thread's sink, if any.

    A crashing sink (e.g. a disconnected browser) must never take down the
    research run itself, hence the broad swallow.
    """
    sink: Sink | None = getattr(_local, "sink", None)
    if sink is not None:
        try:
            sink(agent, message)
        except Exception:
            pass
