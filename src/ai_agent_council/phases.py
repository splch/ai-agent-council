"""The Braintrust loop, phase by phase.

Anti-anchoring invariant
------------------------
The divergent phase prompt is built via `render_divergent_prompt(task)` — a pure function of
the task. Peer output is structurally inaccessible here. `test_divergent_anti_anchoring`
asserts no diverger's prompt contains any other diverger's output.
"""

import asyncio
import time
from collections.abc import Callable, Coroutine, Sequence
from typing import TYPE_CHECKING, Any

from .llm import StreamHandler
from .models import Message, Phase, PhaseOutput, Role
from .prompts import (
    render_critique_prompt,
    render_divergent_prompt,
    render_finishing_prompt,
    render_orchestrate_prompt,
    render_restate_prompt,
    render_retrospective_prompt,
    render_steelman_prompt,
    render_synthesis_prompt,
)

if TYPE_CHECKING:
    from .agent import Agent
    from .council import Council


# Callback signature: (agent_name, content_chunk) → None. Invoked once per streamed delta.
TokenStream = Callable[[str, str], None]


def _handler_for(ts: TokenStream | None, agent_name: str) -> StreamHandler | None:
    """Wrap a TokenStream so it looks like a per-agent StreamHandler."""
    if ts is None:
        return None
    return lambda chunk: ts(agent_name, chunk)


def _select_divergers(council: Council) -> list[Agent]:
    """Ideator(s) + Specialist(s); enlist a Reasoner as fallback to keep the
    'multiple independent drafts' invariant satisfied in small rosters."""
    primary = council.by_role.get(Role.IDEATOR, []) + council.by_role.get(Role.SPECIALIST, [])
    if len(primary) >= 2:
        return primary
    return primary + council.by_role.get(Role.REASONER, [])[: 2 - len(primary)]


async def _run_concurrently(
    coros: Sequence[Coroutine[Any, Any, Message]],
) -> list[Message]:
    """TaskGroup runner. Results come back in submission order."""
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(c) for c in coros]
    return [t.result() for t in tasks]


async def run_restate(
    council: Council, task: str, *, tokens: TokenStream | None = None
) -> PhaseOutput:
    """Restate phase. Every council member independently restates the task plus one
    alternative framing. Per the anti-anchoring rule, each agent receives the same
    isolated prompt — they don't see each other's restatements. The transcript makes
    divergent interpretations visible to a human reviewer.
    """
    members = list(council.agents.values())
    # Orchestrator skipped — its job is synthesis, not interpretation.
    members = [a for a in members if a.config.role is not Role.ORCHESTRATOR]
    if not members:
        return PhaseOutput(phase=Phase.RESTATE, messages=[], elapsed_ms=0)
    user_prompt = render_restate_prompt(task)
    t0 = time.monotonic()
    messages = await _run_concurrently(
        [
            a.respond(
                user_prompt,
                phase=Phase.RESTATE,
                stream_handler=_handler_for(tokens, a.config.name),
            )
            for a in members
        ]
    )
    return PhaseOutput(
        phase=Phase.RESTATE,
        messages=messages,
        elapsed_ms=int((time.monotonic() - t0) * 1000),
    )


async def run_divergent(
    council: Council, task: str, *, tokens: TokenStream | None = None
) -> PhaseOutput:
    """Divergent phase. Each agent receives the same isolated prompt — no peer output."""
    divergers = _select_divergers(council)
    # The prompt builder signature takes only `task`. Peer drafts are structurally
    # unreachable from here. This is the enforcement point of the anti-anchoring rule.
    user_prompt = render_divergent_prompt(task)

    t0 = time.monotonic()
    messages = await _run_concurrently(
        [
            a.respond(
                user_prompt,
                phase=Phase.DIVERGENT,
                stream_handler=_handler_for(tokens, a.config.name),
            )
            for a in divergers
        ]
    )
    return PhaseOutput(
        phase=Phase.DIVERGENT,
        messages=messages,
        elapsed_ms=int((time.monotonic() - t0) * 1000),
    )


async def run_critique(
    council: Council,
    task: str,
    diverge: PhaseOutput,
    *,
    tokens: TokenStream | None = None,
) -> PhaseOutput:
    """Critique phase. Critic + Reasoner review all divergent drafts. Pixar rule applies."""
    reviewers = council.by_role.get(Role.CRITIC, []) + council.by_role.get(Role.REASONER, [])
    if not reviewers:
        return PhaseOutput(phase=Phase.CRITIQUE, messages=[], elapsed_ms=0)
    prompt = render_critique_prompt(task, diverge.messages)
    t0 = time.monotonic()
    messages = await _run_concurrently(
        [
            a.respond(
                prompt,
                phase=Phase.CRITIQUE,
                stream_handler=_handler_for(tokens, a.config.name),
            )
            for a in reviewers
        ]
    )
    return PhaseOutput(
        phase=Phase.CRITIQUE,
        messages=messages,
        elapsed_ms=int((time.monotonic() - t0) * 1000),
    )


async def run_steelman(
    council: Council,
    task: str,
    diverge: PhaseOutput,
    critique: PhaseOutput,
    *,
    tokens: TokenStream | None = None,
) -> PhaseOutput:
    """Steelman round: each critic is asked to produce the strongest possible
    objection to each proposal, triggered when the first critique pass returned
    too few substantive issues."""
    reviewers = council.by_role.get(Role.CRITIC, []) + council.by_role.get(Role.REASONER, [])
    if not reviewers:
        return PhaseOutput(phase=Phase.STEELMAN, messages=[], elapsed_ms=0)
    prompt = render_steelman_prompt(task, diverge.messages, critique.messages)
    t0 = time.monotonic()
    messages = await _run_concurrently(
        [
            a.respond(
                prompt,
                phase=Phase.STEELMAN,
                stream_handler=_handler_for(tokens, a.config.name),
            )
            for a in reviewers
        ]
    )
    return PhaseOutput(
        phase=Phase.STEELMAN,
        messages=messages,
        elapsed_ms=int((time.monotonic() - t0) * 1000),
    )


async def run_synthesis(
    council: Council,
    task: str,
    diverge: PhaseOutput,
    critique: PhaseOutput,
    *,
    tokens: TokenStream | None = None,
) -> PhaseOutput:
    """Synthesis phase. Each original drafter sees critiques, revises. Drafters are
    explicitly permitted to disagree with any critique point (system prompt says so).

    If any council member has a sycophancy_prior set, those priors are surfaced next
    to the critic's name — drafters use the prior to weigh how independently each
    critique was produced.
    """
    priors = {
        name: agent.config.sycophancy_prior
        for name, agent in council.agents.items()
        if agent.config.sycophancy_prior is not None
    }
    t0 = time.monotonic()
    coros = [
        agent.respond(
            render_synthesis_prompt(
                task, original, critique.messages, sycophancy_priors=priors or None
            ),
            phase=Phase.SYNTHESIS,
            stream_handler=_handler_for(tokens, agent.config.name),
        )
        for original in diverge.messages
        if (agent := council.agents.get(original.agent_name)) is not None
    ]
    messages = await _run_concurrently(coros)
    return PhaseOutput(
        phase=Phase.SYNTHESIS,
        messages=messages,
        elapsed_ms=int((time.monotonic() - t0) * 1000),
    )


async def run_finishing(
    council: Council,
    task: str,
    synthesis: PhaseOutput,
    *,
    tokens: TokenStream | None = None,
) -> PhaseOutput:
    """Finishing phase (skipped if no Finisher rostered)."""
    finishers = council.by_role.get(Role.FINISHER, [])
    if not finishers:
        return PhaseOutput(phase=Phase.FINISHING, messages=[], elapsed_ms=0)
    prompt = render_finishing_prompt(task, synthesis.messages)
    t0 = time.monotonic()
    msg = await finishers[0].respond(
        prompt,
        phase=Phase.FINISHING,
        stream_handler=_handler_for(tokens, finishers[0].config.name),
    )
    return PhaseOutput(
        phase=Phase.FINISHING, messages=[msg], elapsed_ms=int((time.monotonic() - t0) * 1000)
    )


async def run_orchestrate(
    council: Council,
    task: str,
    phases_so_far: list[PhaseOutput],
    *,
    tokens: TokenStream | None = None,
) -> PhaseOutput:
    """Orchestration. Thin — integrates transcript into final answer, no new content."""
    # Validator guarantees exactly one orchestrator is present.
    orchestrator = council.by_role[Role.ORCHESTRATOR][0]
    prompt = render_orchestrate_prompt(task, list(phases_so_far))
    t0 = time.monotonic()
    msg = await orchestrator.respond(
        prompt,
        phase=Phase.ORCHESTRATE,
        stream_handler=_handler_for(tokens, orchestrator.config.name),
    )
    return PhaseOutput(
        phase=Phase.ORCHESTRATE, messages=[msg], elapsed_ms=int((time.monotonic() - t0) * 1000)
    )


async def run_retrospective(
    council: Council,
    task: str,
    phases_so_far: list[PhaseOutput],
    *,
    tokens: TokenStream | None = None,
) -> PhaseOutput:
    """Retrospective phase. The Critic reads the transcript and extracts 1-3 lessons for
    future runs. No-op (empty output) if no Critic is rostered — the config validator
    requires one, so this fallback is defensive, not expected."""
    critics = council.by_role.get(Role.CRITIC, [])
    if not critics:
        return PhaseOutput(phase=Phase.RETROSPECTIVE, messages=[], elapsed_ms=0)
    critic = critics[0]
    prompt = render_retrospective_prompt(task, phases_so_far)
    t0 = time.monotonic()
    msg = await critic.respond(
        prompt,
        phase=Phase.RETROSPECTIVE,
        stream_handler=_handler_for(tokens, critic.config.name),
    )
    return PhaseOutput(
        phase=Phase.RETROSPECTIVE,
        messages=[msg],
        elapsed_ms=int((time.monotonic() - t0) * 1000),
    )
