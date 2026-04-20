"""Council: the public entry point. Orchestrates the Braintrust loop."""

import contextlib
import json
import logging
import re
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from . import phases as phases_mod
from . import retrospectives
from .agent import Agent
from .config import CouncilConfig, hash_config, load_council_config
from .models import CouncilResult, PhaseOutput, Role
from .phases import TokenStream
from .prompts import render_lessons_block

PhaseStream = Callable[[PhaseOutput], None]


class Council:
    """A stable roster of agents running the Braintrust workflow on a task."""

    def __init__(self, config: CouncilConfig) -> None:
        self.config = config
        lessons_block = ""
        if config.retrospective:
            recent = retrospectives.load_recent(
                config.name,
                limit=config.retrospective_recall,
                dir_=config.retrospective_dir,
            )
            flat_lessons = [lesson for r in recent for lesson in r.lessons]
            lessons_block = render_lessons_block(flat_lessons)

        self.agents: dict[str, Agent] = {}
        self.by_role: defaultdict[Role, list[Agent]] = defaultdict(list)
        for cfg in config.agents:
            agent = Agent.from_config(cfg, lessons=lessons_block)
            self.agents[cfg.name] = agent
            self.by_role[cfg.role].append(agent)

    @classmethod
    def from_yaml(cls, path: Path | str) -> Self:
        return cls(load_council_config(path))

    async def run(
        self,
        task: str,
        *,
        stream: PhaseStream | None = None,
        tokens: TokenStream | None = None,
    ) -> CouncilResult:
        """Run the council end-to-end on `task`.

        Phase order: [restate] → divergent → critique → synthesis → [finishing] →
        orchestrate → [retrospective]. Optional phases run only when their config flag
        is set. Each completed phase is passed to `stream` if given, before the next
        begins. Each token chunk is passed to `tokens(agent_name, chunk)` as it
        arrives, for callers that want to render live token streams. Retrospective
        lessons are appended to `<retrospective_dir>/<council_name>.jsonl` when enabled.
        """
        started = datetime.now(UTC)
        phases_run: list[PhaseOutput] = []

        if self.config.restate:
            restate = await phases_mod.run_restate(self, task, tokens=tokens)
            _emit(stream, restate)
            phases_run.append(restate)

        divergent = await phases_mod.run_divergent(self, task, tokens=tokens)
        _emit(stream, divergent)
        phases_run.append(divergent)

        critique = await phases_mod.run_critique(self, task, divergent, tokens=tokens)
        _emit(stream, critique)
        phases_run.append(critique)

        synthesis = await phases_mod.run_synthesis(self, task, divergent, critique, tokens=tokens)
        _emit(stream, synthesis)
        phases_run.append(synthesis)

        if Role.FINISHER in self.by_role:
            finishing = await phases_mod.run_finishing(self, task, synthesis, tokens=tokens)
            _emit(stream, finishing)
            phases_run.append(finishing)

        orchestration = await phases_mod.run_orchestrate(self, task, phases_run, tokens=tokens)
        _emit(stream, orchestration)
        phases_run.append(orchestration)
        final_answer = orchestration.messages[0].content if orchestration.messages else ""

        if self.config.retrospective:
            retro_phase = await phases_mod.run_retrospective(self, task, phases_run, tokens=tokens)
            _emit(stream, retro_phase)
            phases_run.append(retro_phase)
            _persist_retrospective(self.config, task, retro_phase, phases_run)

        return CouncilResult(
            task=task,
            final_answer=final_answer,
            phases=phases_run,
            config_digest=hash_config(self.config),
            started_at=started,
            finished_at=datetime.now(UTC),
        )


async def run_council(task: str, config: CouncilConfig) -> CouncilResult:
    """Library-level convenience: construct a Council and run `task`."""
    return await Council(config).run(task)


def write_transcript(result: CouncilResult, path: Path | str) -> Path:
    """Write a CouncilResult to `path` as UTF-8 JSON. Returns the path written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return p


_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+\.)\s+(.+)$")


def _extract_lessons(retro_content: str) -> list[str]:
    """Pull lessons out of the critic's retrospective message.

    Preferred: JSON with `{"lessons": [...]}`. Fallback: bullet / numbered lines.
    Caps at 3 non-empty entries so a runaway Critic can't bloat the log.
    """
    if not retro_content.strip():
        return []
    try:
        data = json.loads(retro_content)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict):
        raw = data.get("lessons")
        if isinstance(raw, list):
            return [str(x).strip() for x in raw if str(x).strip()][:3]
    out = [
        m.group(1).strip() for line in retro_content.splitlines() if (m := _BULLET_RE.match(line))
    ]
    return [x for x in out if x][:3]


def _persist_retrospective(
    config: CouncilConfig, task: str, retro_phase: PhaseOutput, all_phases: list[PhaseOutput]
) -> None:
    """Extract lessons from the retrospective phase and append a JSONL record. Failures
    are logged but never raised — persistence is side-channel; a storage hiccup must not
    fail the run."""
    if not retro_phase.messages or retro_phase.messages[0].error:
        return
    lessons = _extract_lessons(retro_phase.messages[0].content)
    if not lessons:
        return
    total_cost = sum((m.cost_usd or 0.0) for ph in all_phases for m in ph.messages)
    record = retrospectives.Retrospective(
        timestamp=datetime.now(UTC),
        council_name=config.name,
        task=task,
        config_digest=hash_config(config),
        lessons=lessons,
        cost_usd=total_cost,
    )
    try:
        retrospectives.append(record, dir_=config.retrospective_dir)
    except OSError as e:
        logging.getLogger(__name__).warning("failed to persist retrospective: %s", e)


def _emit(stream: PhaseStream | None, phase: PhaseOutput) -> None:
    # A broken stream sink should not abort the run.
    if stream is not None:
        with contextlib.suppress(Exception):
            stream(phase)
