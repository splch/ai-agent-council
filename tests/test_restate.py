"""Restate phase tests."""

import inspect

from ai_agent_council.config import AgentConfig, CouncilConfig
from ai_agent_council.council import Council
from ai_agent_council.models import Phase, Role
from ai_agent_council.phases import run_restate
from ai_agent_council.prompts import render_restate_prompt

from .fakes import FakeCall, FakeLLM, agent_from_system


def _config(*, restate: bool = False) -> CouncilConfig:
    return CouncilConfig(
        name="restate-test",
        restate=restate,
        agents=[
            AgentConfig(name="Muse", role=Role.IDEATOR, model="ollama/gemma3:2b", temperature=0.9),
            AgentConfig(
                name="Judge",
                role=Role.REASONER,
                model="ollama/deepseek-r1:1.5b",
                temperature=0.4,
            ),
            AgentConfig(
                name="Hawk", role=Role.CRITIC, model="ollama/phi4-mini:3.8b", temperature=0.2
            ),
            AgentConfig(
                name="Scribe",
                role=Role.ORCHESTRATOR,
                model="ollama/llama3.2:3b",
                temperature=0.3,
            ),
        ],
    )


def test_restate_prompt_signature_is_pure_in_task() -> None:
    """Same structural invariant as divergent — restate is task-only. Peer output is
    unavailable by signature."""
    sig = inspect.signature(render_restate_prompt)
    assert list(sig.parameters) == ["task"]
    assert sig.parameters["task"].annotation is str


async def test_restate_disabled_by_default(fake_llm: FakeLLM) -> None:
    fake_llm.default = "ok"
    result = await Council(_config()).run("task")
    assert Phase.RESTATE not in [p.phase for p in result.phases]


async def test_restate_runs_when_enabled(fake_llm: FakeLLM) -> None:
    fake_llm.responder = lambda call: f"RESTATE: {agent_from_system(call.system)} view\nALT: x"
    result = await Council(_config(restate=True)).run("Find recursion examples.")

    phases = [p.phase for p in result.phases]
    assert phases[0] is Phase.RESTATE  # first phase
    restate_phase = result.phases[0]
    # Orchestrator skipped, so 3 non-orchestrator members participate.
    assert {m.agent_name for m in restate_phase.messages} == {"Muse", "Judge", "Hawk"}


async def test_restate_isolation(fake_llm: FakeLLM) -> None:
    """Each restate prompt contains only the task — no peer restatements. Same
    anti-anchoring invariant as divergent."""
    fake_llm.responder = lambda call: f"RESTATE: {agent_from_system(call.system)} view"
    council = Council(_config(restate=True))
    phase = await run_restate(council, "Explain quantum tunneling.")

    restate_calls = [
        c
        for c in fake_llm.calls
        if any(f"You are {m.agent_name}," in c.system for m in phase.messages)
    ]
    # No agent's user prompt contains another's name/response.
    for call in restate_calls:
        for m in phase.messages:
            if f"You are {m.agent_name}," not in call.system:
                assert f"You are {m.agent_name}" not in call.user
                assert m.content not in call.user


async def test_restate_phase_type_is_restate(fake_llm: FakeLLM) -> None:
    fake_llm.default = "RESTATE: x\nALT: y"
    phase = await run_restate(Council(_config(restate=True)), "anything")
    assert all(m.phase is Phase.RESTATE for m in phase.messages)


async def test_restate_excludes_orchestrator(fake_llm: FakeLLM) -> None:
    """Orchestrator's job is synthesis, not interpretation — it should not restate."""

    def responder(call: FakeCall) -> str:
        return f"RESTATE: {agent_from_system(call.system)} view\nALT: x"

    fake_llm.responder = responder
    phase = await run_restate(Council(_config(restate=True)), "task")
    names = {m.agent_name for m in phase.messages}
    assert "Scribe" not in names  # Scribe is the orchestrator
