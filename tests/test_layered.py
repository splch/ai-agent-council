"""Cross-synthesis / layered MoA tests."""

from ai_agent_council.config import AgentConfig, CouncilConfig
from ai_agent_council.council import Council
from ai_agent_council.models import Message, Phase, Role
from ai_agent_council.phases import run_cross_synthesis
from ai_agent_council.prompts import render_cross_synthesis_prompt

from .fakes import FakeCall, FakeLLM, agent_from_system


def _cfg(*, layered: bool = False) -> CouncilConfig:
    return CouncilConfig(
        name="layered-test",
        layered=layered,
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


def _msg(name: str, content: str) -> Message:
    return Message(
        role=Role.IDEATOR,
        agent_name=name,
        phase=Phase.SYNTHESIS,
        content=content,
        model="ollama/x",
        temperature=0.5,
    )


def test_cross_synthesis_prompt_excludes_own_draft_from_peers_block() -> None:
    """The own-synthesis section shouldn't be duplicated as a 'peer'."""
    own = _msg("Muse", "MY OWN UNIQUE DRAFT")
    peers = [own, _msg("Judge", "PEER JUDGE DRAFT")]
    prompt = render_cross_synthesis_prompt("t", own, peers)
    # Own draft appears exactly once (in the "your own" section).
    assert prompt.count("MY OWN UNIQUE DRAFT") == 1
    # Peer's content appears exactly once.
    assert prompt.count("PEER JUDGE DRAFT") == 1


def test_cross_synthesis_prompt_includes_all_peers() -> None:
    own = _msg("Muse", "own")
    peers = [own, _msg("Judge", "JUDGE_ZZZ"), _msg("Hawk", "HAWK_ZZZ")]
    prompt = render_cross_synthesis_prompt("t", own, peers)
    assert "JUDGE_ZZZ" in prompt
    assert "HAWK_ZZZ" in prompt


def test_cross_synthesis_prompt_handles_failed_peer() -> None:
    own = _msg("Muse", "own")
    failed = Message(
        role=Role.IDEATOR,
        agent_name="Judge",
        phase=Phase.SYNTHESIS,
        content="",
        model="ollama/x",
        temperature=0.5,
        error="synthetic",
    )
    prompt = render_cross_synthesis_prompt("t", own, [own, failed])
    assert "FAILED" in prompt
    assert "synthetic" in prompt


async def test_layered_disabled_by_default(fake_llm: FakeLLM) -> None:
    fake_llm.default = "ok"
    result = await Council(_cfg()).run("task")
    assert Phase.CROSS_SYNTHESIS not in [p.phase for p in result.phases]


async def test_layered_runs_cross_synthesis(fake_llm: FakeLLM) -> None:
    fake_llm.responder = lambda call: f"draft from {agent_from_system(call.system)}"
    result = await Council(_cfg(layered=True)).run("task")
    phases = [p.phase for p in result.phases]
    assert Phase.CROSS_SYNTHESIS in phases
    # CROSS_SYNTHESIS comes immediately after SYNTHESIS.
    assert phases.index(Phase.CROSS_SYNTHESIS) == phases.index(Phase.SYNTHESIS) + 1


async def test_layered_cross_synthesis_feeds_downstream(fake_llm: FakeLLM) -> None:
    """With layered=true, the Finisher / Orchestrator see cross-synthesis output."""

    def responder(call: FakeCall) -> str:
        # The cross-synthesis prompt has a specific opening line.
        is_cross = "Your own revised draft from the synthesis phase" in call.user
        if is_cross:
            return f"CROSS-{agent_from_system(call.system)}"
        return f"PLAIN-{agent_from_system(call.system)}"

    fake_llm.responder = responder
    await Council(_cfg(layered=True)).run("task")

    # Orchestrator's prompt should include CROSS- content (the downstream pointer).
    scribe_calls = [c for c in fake_llm.calls if "You are Scribe," in c.system]
    assert scribe_calls
    orch_prompt = scribe_calls[-1].user
    assert "CROSS-" in orch_prompt


async def test_cross_synthesis_phase_correct_label(fake_llm: FakeLLM) -> None:
    fake_llm.default = "x"
    council = Council(_cfg(layered=True))
    # Build a synthesis phase manually to feed run_cross_synthesis.
    from ai_agent_council.models import PhaseOutput

    synthesis = PhaseOutput(
        phase=Phase.SYNTHESIS,
        messages=[_msg("Muse", "a"), _msg("Judge", "b")],
        elapsed_ms=0,
    )
    phase = await run_cross_synthesis(council, "task", synthesis)
    assert phase.phase is Phase.CROSS_SYNTHESIS
    assert all(m.phase is Phase.CROSS_SYNTHESIS for m in phase.messages)
