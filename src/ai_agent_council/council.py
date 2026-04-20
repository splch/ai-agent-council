"""Council: the public entry point. Orchestrates the Braintrust loop."""

import contextlib
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from . import phases as phases_mod
from .agent import Agent
from .config import CouncilConfig, hash_config, load_council_config
from .models import CouncilResult, PhaseOutput, Role
from .phases import TokenStream

PhaseStream = Callable[[PhaseOutput], None]


class Council:
    """A stable roster of agents running the Braintrust workflow on a task."""

    def __init__(self, config: CouncilConfig) -> None:
        self.config = config
        self.agents: dict[str, Agent] = {}
        self.by_role: defaultdict[Role, list[Agent]] = defaultdict(list)
        for cfg in config.agents:
            agent = Agent.from_config(cfg)
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

        Phase order: divergent → critique → synthesis → [finishing] → orchestrate.
        Each completed phase is passed to `stream` if given, before the next begins.
        Each token chunk is passed to `tokens(agent_name, chunk)` as it arrives, for
        callers that want to render live token streams.
        """
        started = datetime.now(UTC)
        phases_run: list[PhaseOutput] = []

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


def _emit(stream: PhaseStream | None, phase: PhaseOutput) -> None:
    if stream is None:
        return
    # A broken stream sink should not abort the run.
    with contextlib.suppress(Exception):
        stream(phase)
