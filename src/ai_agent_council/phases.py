"""The Braintrust loop, phase by phase.

Anti-anchoring invariant
------------------------
The divergent phase prompt is built via `render_divergent_prompt(task)` — a pure function of
the task. Peer output is structurally inaccessible here. `test_divergent_anti_anchoring`
asserts no diverger's prompt contains any other diverger's output.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from .models import Message, Phase, PhaseOutput, Role
from .prompts import (
    render_critique_prompt,
    render_divergent_prompt,
    render_finishing_prompt,
    render_orchestrate_prompt,
    render_synthesis_prompt,
)

if TYPE_CHECKING:
    from .agent import Agent
    from .council import Council


def _elapsed_ms(t0: float) -> int:
    return int((time.monotonic() - t0) * 1000)


def _select_divergers(council: Council) -> list[Agent]:
    """Pick the agents that will produce independent first-pass answers.

    Primary: Ideator(s) + Specialist(s).
    Fallback: if fewer than two divergers would be rostered (common in 4-agent configs
    without a Specialist), enlist one Reasoner so the "multiple independent drafts"
    invariant is preserved.
    """
    divergers: list[Agent] = []
    divergers.extend(council.by_role.get(Role.IDEATOR, []))
    divergers.extend(council.by_role.get(Role.SPECIALIST, []))
    if len(divergers) < 2:
        for reasoner in council.by_role.get(Role.REASONER, []):
            divergers.append(reasoner)
            if len(divergers) >= 2:
                break
    return divergers


async def run_divergent(council: Council, task: str) -> PhaseOutput:
    """Divergent phase. Each agent receives the same isolated prompt — no peer output."""
    divergers = _select_divergers(council)
    # The prompt builder signature takes only `task`. Peer drafts are structurally
    # unreachable from here. This is the enforcement point of the anti-anchoring rule.
    user_prompt = render_divergent_prompt(task)

    t0 = time.monotonic()
    messages = await asyncio.gather(
        *(a.respond(user_prompt, phase=Phase.DIVERGENT) for a in divergers)
    )
    return PhaseOutput(phase=Phase.DIVERGENT, messages=list(messages), elapsed_ms=_elapsed_ms(t0))


async def run_critique(council: Council, task: str, diverge: PhaseOutput) -> PhaseOutput:
    """Critique phase. Critic + Reasoner review all divergent drafts. Pixar rule applies."""
    reviewers: list[Agent] = []
    reviewers.extend(council.by_role.get(Role.CRITIC, []))
    reviewers.extend(council.by_role.get(Role.REASONER, []))
    if not reviewers:
        return PhaseOutput(phase=Phase.CRITIQUE, messages=[], elapsed_ms=0)
    prompt = render_critique_prompt(task, diverge.messages)
    t0 = time.monotonic()
    messages = await asyncio.gather(*(a.respond(prompt, phase=Phase.CRITIQUE) for a in reviewers))
    return PhaseOutput(phase=Phase.CRITIQUE, messages=list(messages), elapsed_ms=_elapsed_ms(t0))


async def run_synthesis(
    council: Council, task: str, diverge: PhaseOutput, critique: PhaseOutput
) -> PhaseOutput:
    """Synthesis phase. Each original drafter sees critiques, revises.

    Drafters are permitted to disagree with any critique point (system prompt says so).
    """
    drafters: list[tuple[Agent, Message]] = []
    for msg in diverge.messages:
        agent = council.agents.get(msg.agent_name)
        if agent is not None:
            drafters.append((agent, msg))
    t0 = time.monotonic()
    tasks = [
        drafter.respond(
            render_synthesis_prompt(task, original, critique.messages),
            phase=Phase.SYNTHESIS,
        )
        for drafter, original in drafters
    ]
    messages = await asyncio.gather(*tasks) if tasks else []
    return PhaseOutput(phase=Phase.SYNTHESIS, messages=list(messages), elapsed_ms=_elapsed_ms(t0))


async def run_finishing(council: Council, task: str, synthesis: PhaseOutput) -> PhaseOutput:
    """Finishing phase (skipped if no Finisher rostered)."""
    finishers = council.by_role.get(Role.FINISHER, [])
    if not finishers:
        return PhaseOutput(phase=Phase.FINISHING, messages=[], elapsed_ms=0)
    finisher = finishers[0]
    prompt = render_finishing_prompt(task, synthesis.messages)
    t0 = time.monotonic()
    msg = await finisher.respond(prompt, phase=Phase.FINISHING)
    return PhaseOutput(phase=Phase.FINISHING, messages=[msg], elapsed_ms=_elapsed_ms(t0))


async def run_orchestrate(
    council: Council, task: str, phases_so_far: list[PhaseOutput]
) -> PhaseOutput:
    """Orchestration. Thin — integrates transcript into final answer, no new content."""
    orchestrators = council.by_role.get(Role.ORCHESTRATOR, [])
    if not orchestrators:  # should be unreachable — config validator enforces presence
        return PhaseOutput(phase=Phase.ORCHESTRATE, messages=[], elapsed_ms=0)
    orchestrator = orchestrators[0]
    prompt = render_orchestrate_prompt(task, list(phases_so_far))
    t0 = time.monotonic()
    msg = await orchestrator.respond(prompt, phase=Phase.ORCHESTRATE)
    return PhaseOutput(phase=Phase.ORCHESTRATE, messages=[msg], elapsed_ms=_elapsed_ms(t0))
