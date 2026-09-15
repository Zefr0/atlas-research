"""
Tiny console reporter so you can watch each agent work live.

Deliberately minimal (no logging framework, no colors library) — the point is
progress visibility, not observability infrastructure. Swap for `rich` or
structured logging later if the project grows.
"""

from __future__ import annotations

import sys

from . import events

# Windows consoles often use a legacy codepage (cp1252) that can't encode
# characters LLMs and web content routinely produce (curly quotes, special
# hyphens...). errors="replace" swaps the offending character for '?' instead
# of letting a UnicodeEncodeError kill a whole research run mid-flight.
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(errors="replace")
        except Exception:
            pass


def say(agent: str, message: str) -> None:
    """Print one progress line, tagged with the agent that produced it.

    Also forwards the message to the current thread's event sink (if one is
    registered) so the web UI can stream progress live — see events.py.
    """
    events.emit(agent, message)
    print(f"  [{agent:>12}] {message}", flush=True)


def banner(text: str) -> None:
    """Print a section divider for major pipeline phases."""
    print(f"\n=== {text} " + "=" * max(0, 60 - len(text)), flush=True)
