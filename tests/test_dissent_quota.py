"""Dissent quota / steelman round tests."""

from ai_agent_council.config import AgentConfig, CouncilConfig
from ai_agent_council.council import (
    Council,
    _count_substantive_critiques,
    _should_steelman,
)
from ai_agent_council.models import Message, Phase, PhaseOutput, Role

from .fakes import FakeCall, FakeLLM


def _config(*, min_dissent: float = 0.0) -> CouncilConfig:
    return CouncilConfig(
        name="steelman-test",
        min_dissent=min_dissent,
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


def _fake_critique(name: str, content: str) -> Message:
    return Message(
        role=Role.CRITIC,
        agent_name=name,
        phase=Phase.CRITIQUE,
        content=content,
        model="ollama/x",
        temperature=0.2,
    )


def _critique_phase(*contents: str) -> PhaseOutput:
    return PhaseOutput(
        phase=Phase.CRITIQUE,
        messages=[_fake_critique(f"C{i}", c) for i, c in enumerate(contents)],
        elapsed_ms=10,
    )


# -----------------------------------------------------------------------------
# _count_substantive_critiques
# -----------------------------------------------------------------------------


def test_counts_substantive_json_critique() -> None:
    phase = _critique_phase(
        '{"critiques": [{"target": "Proposal A", "issues": [{"quote": "x", "problem": "y"}]}]}'
    )
    assert _count_substantive_critiques(phase) == (1, 1)


def test_counts_empty_json_as_non_substantive() -> None:
    phase = _critique_phase('{"critiques": []}')
    assert _count_substantive_critiques(phase) == (0, 1)


def test_counts_issues_empty_array_as_non_substantive() -> None:
    phase = _critique_phase('{"critiques": [{"target": "A", "issues": []}]}')
    assert _count_substantive_critiques(phase) == (0, 1)


def test_counts_non_json_long_text_as_substantive() -> None:
    phase = _critique_phase("This proposal has several specific flaws worth discussing in depth.")
    assert _count_substantive_critiques(phase) == (1, 1)


def test_counts_short_non_json_as_non_substantive() -> None:
    phase = _critique_phase("looks ok")
    assert _count_substantive_critiques(phase) == (0, 1)


def test_mixed_critiques() -> None:
    phase = _critique_phase(
        '{"critiques": [{"target": "A", "issues": [{"quote": "x", "problem": "y"}]}]}',
        '{"critiques": []}',
        "all good",
    )
    assert _count_substantive_critiques(phase) == (1, 3)


# -----------------------------------------------------------------------------
# _should_steelman
# -----------------------------------------------------------------------------


def test_should_steelman_disabled_when_min_dissent_zero() -> None:
    phase = _critique_phase('{"critiques": []}', '{"critiques": []}')
    assert _should_steelman(phase, 0.0) is False


def test_should_steelman_triggers_below_threshold() -> None:
    # 0 of 2 critics found issues; threshold is 0.5 → trigger.
    phase = _critique_phase('{"critiques": []}', '{"critiques": []}')
    assert _should_steelman(phase, 0.5) is True


def test_should_steelman_does_not_trigger_above_threshold() -> None:
    phase = _critique_phase(
        '{"critiques": [{"target": "A", "issues": [{"quote": "x", "problem": "y"}]}]}',
        '{"critiques": [{"target": "B", "issues": [{"quote": "x", "problem": "y"}]}]}',
    )
    assert _should_steelman(phase, 0.5) is False  # 2/2 = 100% > 50%


def test_should_steelman_with_all_agents_failed_does_not_trigger() -> None:
    failed = Message(
        role=Role.CRITIC,
        agent_name="C0",
        phase=Phase.CRITIQUE,
        content="",
        model="ollama/x",
        temperature=0.2,
        error="synthetic",
    )
    phase = PhaseOutput(phase=Phase.CRITIQUE, messages=[failed], elapsed_ms=0)
    assert _should_steelman(phase, 0.5) is False


# -----------------------------------------------------------------------------
# End-to-end: does the council actually invoke steelman?
# -----------------------------------------------------------------------------


async def test_steelman_phase_runs_when_quota_triggers(fake_llm: FakeLLM) -> None:
    """Critics return empty critiques in the first round, triggering the dissent
    quota and a steelman pass. The round-2 prompt is distinguishable by its
    opening line ('too few substantive objections')."""

    def responder(call: FakeCall) -> str:
        is_steelman_round = "too few substantive objections" in call.user
        # Hawk + Judge are the critique-phase reviewers.
        if "You are Hawk," in call.system or "You are Judge," in call.system:
            if is_steelman_round:
                return (
                    '{"steelman": true, "critiques": [{"target": "Proposal A", '
                    '"issues": [{"quote": "x", "problem": "y", "endorsed": false}]}]}'
                )
            # First critique pass: empty, triggers the quota.
            return '{"critiques": []}'
        return "content"

    fake_llm.responder = responder
    result = await Council(_config(min_dissent=0.5)).run("task")
    assert Phase.STEELMAN in [p.phase for p in result.phases]


async def test_steelman_phase_does_not_run_when_disabled(fake_llm: FakeLLM) -> None:
    """Default config has min_dissent=0; steelman never runs."""
    fake_llm.default = '{"critiques": []}'
    result = await Council(_config()).run("task")
    assert Phase.STEELMAN not in [p.phase for p in result.phases]


async def test_steelman_phase_does_not_run_when_quota_satisfied(fake_llm: FakeLLM) -> None:
    """Critics produce substantive critiques; steelman is skipped."""
    fake_llm.default = (
        '{"critiques": [{"target": "Proposal A", "issues": '
        '[{"quote": "x", "problem": "y"}]}]}'
    )
    result = await Council(_config(min_dissent=0.5)).run("task")
    assert Phase.STEELMAN not in [p.phase for p in result.phases]
