"""
Markdown export — turning the final state into a file on disk.

Filenames are `<UTC timestamp>_<slugified query>.md`, e.g.
`20260827T143012Z_impact-of-ai-on-healthcare.md`, so repeated runs never
overwrite each other and the reports/ folder sorts chronologically.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path


def _slugify(text: str, max_len: int = 60) -> str:
    """Make a query safe for a filename: lowercase, hyphens, ASCII-ish."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].rstrip("-") or "report"


def export_report(reports_dir: Path, query: str, report_markdown: str) -> Path:
    """Write the report and return its path. Creates reports/ on first run."""
    reports_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = reports_dir / f"{timestamp}_{_slugify(query)}.md"

    # utf-8 explicitly: Windows' default codepage would mangle non-ASCII
    # characters (names, currency symbols) that often appear in web content.
    path.write_text(report_markdown, encoding="utf-8")
    return path
