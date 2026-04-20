"""Transcript I/O. Serializes a `CouncilResult` as JSON."""

from __future__ import annotations

from pathlib import Path

from .models import CouncilResult


def write_transcript(result: CouncilResult, path: Path | str) -> Path:
    """Write the transcript as UTF-8 JSON. Returns the path written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return p
