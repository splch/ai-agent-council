"""Retrospective persistence: append-only JSONL log, one file per council name.

Each line is a `Retrospective` record — the lessons extracted from a single run. On the
next run of a council with the same name, the most recent N entries are loaded and injected
into every agent's system prompt as "Recent lessons from prior council runs", closing the
feedback loop the design brief calls out: *the highest-performing human teams all have this
loop; AI councils should too.*

Design intent — the loop only helps if:
    1. it writes concrete, actionable lessons (the prompt enforces this),
    2. it reads them back on the next run (Council does this in `__init__`),
    3. it's cheap and local (JSONL — no schema migrations, no DB dependency).

Storage root defaults to `$XDG_DATA_HOME/ai_agent_council/retrospectives/` (falling back
to `~/.local/share/...` on systems without XDG); override per-run by passing `dir_=`.
"""

import os
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class Retrospective(BaseModel):
    """Lessons learned from a single council run."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    council_name: str
    task: str
    config_digest: str
    lessons: list[str]
    cost_usd: float = 0.0


def default_dir() -> Path:
    """XDG-compliant default location. Expands at call time, not import time, so tests
    that monkeypatch `$HOME` work as expected."""
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return base / "ai_agent_council" / "retrospectives"


def _path_for(council_name: str, dir_: Path | None) -> Path:
    return (dir_ or default_dir()).expanduser() / f"{council_name}.jsonl"


def append(record: Retrospective, *, dir_: Path | None = None) -> Path:
    """Append a Retrospective as one JSON line. Creates the parent dir if missing."""
    path = _path_for(record.council_name, dir_)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")
    return path


def load_recent(
    council_name: str, limit: int = 5, *, dir_: Path | None = None
) -> list[Retrospective]:
    """Return the most-recent `limit` retrospectives for this council, oldest-first.
    Malformed lines are skipped silently rather than aborting the whole load — the log is
    meant to be append-only, but partial writes can happen."""
    path = _path_for(council_name, dir_)
    if not path.exists() or limit <= 0:
        return []
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out: list[Retrospective] = []
    for line in lines[-limit:]:
        try:
            out.append(Retrospective.model_validate_json(line))
        except Exception:
            continue
    return out
