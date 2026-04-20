"""Retrospective loop: persist lessons after a run, inject them on the next run."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from ai_agent_council import retrospectives
from ai_agent_council.config import AgentConfig, CouncilConfig
from ai_agent_council.council import Council, _extract_lessons
from ai_agent_council.models import Phase, Role
from ai_agent_council.retrospectives import Retrospective

from .fakes import FakeCall, FakeLLM


def _roster(name: str = "test-council", *, retrospective: bool = False, **extra) -> CouncilConfig:
    return CouncilConfig(
        name=name,
        retrospective=retrospective,
        agents=[
            AgentConfig(
                name="Muse", role=Role.IDEATOR, model="ollama/gemma3:2b", temperature=0.9
            ),
            AgentConfig(
                name="Judge",
                role=Role.REASONER,
                model="ollama/deepseek-r1:1.5b",
                temperature=0.4,
            ),
            AgentConfig(
                name="Hawk",
                role=Role.CRITIC,
                model="ollama/phi4-mini:3.8b",
                temperature=0.2,
                json_mode=True,
            ),
            AgentConfig(
                name="Scribe",
                role=Role.ORCHESTRATOR,
                model="anthropic/claude-haiku-4-5-20251001",
                temperature=0.3,
            ),
        ],
        **extra,
    )


# -----------------------------------------------------------------------------
# _extract_lessons
# -----------------------------------------------------------------------------


def test_extract_lessons_parses_json() -> None:
    out = _extract_lessons('{"lessons": ["one", "two"]}')
    assert out == ["one", "two"]


def test_extract_lessons_parses_bullets() -> None:
    out = _extract_lessons("- first lesson\n- second lesson\n* third lesson")
    assert out == ["first lesson", "second lesson", "third lesson"]


def test_extract_lessons_parses_numbered() -> None:
    out = _extract_lessons("1. one thing\n2. two things\n3. three")
    assert out == ["one thing", "two things", "three"]


def test_extract_lessons_caps_at_three() -> None:
    out = _extract_lessons('{"lessons": ["a", "b", "c", "d", "e"]}')
    assert out == ["a", "b", "c"]


def test_extract_lessons_ignores_garbage() -> None:
    assert _extract_lessons("") == []
    assert _extract_lessons("this is prose with no bullets") == []


def test_extract_lessons_strips_empties() -> None:
    out = _extract_lessons('{"lessons": ["valid", "", "  ", "also valid"]}')
    assert out == ["valid", "also valid"]


# -----------------------------------------------------------------------------
# retrospectives persistence round-trip
# -----------------------------------------------------------------------------


def _make_record(name: str = "test-council", *lessons: str) -> Retrospective:
    return Retrospective(
        timestamp=datetime.now(UTC),
        council_name=name,
        task="t",
        config_digest="d",
        lessons=list(lessons) or ["default lesson"],
        cost_usd=0.0,
    )


def test_append_and_load_roundtrip(tmp_path: Path) -> None:
    record = _make_record("rt", "lesson A")
    retrospectives.append(record, dir_=tmp_path)
    loaded = retrospectives.load_recent("rt", dir_=tmp_path)
    assert len(loaded) == 1
    assert loaded[0].lessons == ["lesson A"]


def test_load_recent_limits_and_orders(tmp_path: Path) -> None:
    for i in range(10):
        retrospectives.append(_make_record("rt", f"l{i}"), dir_=tmp_path)
    loaded = retrospectives.load_recent("rt", limit=3, dir_=tmp_path)
    assert [r.lessons[0] for r in loaded] == ["l7", "l8", "l9"]


def test_load_recent_missing_file_returns_empty(tmp_path: Path) -> None:
    assert retrospectives.load_recent("never-run", dir_=tmp_path) == []


def test_load_recent_skips_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / "test.jsonl"
    good = _make_record("test", "good").model_dump_json()
    path.write_text(f"{good}\n{{ not json\n{good}\n", encoding="utf-8")
    loaded = retrospectives.load_recent("test", dir_=tmp_path)
    assert len(loaded) == 2


# -----------------------------------------------------------------------------
# End-to-end: disabled by default, writes when enabled, injects on next run
# -----------------------------------------------------------------------------


async def test_retrospective_disabled_by_default_no_extra_phase(
    fake_llm: FakeLLM, tmp_path: Path
) -> None:
    fake_llm.default = "ok"
    result = await Council(_roster()).run("task")
    assert Phase.RETROSPECTIVE not in [p.phase for p in result.phases]
    # and no files written
    assert list(tmp_path.iterdir()) == []


async def test_retrospective_enabled_runs_phase_and_persists(
    fake_llm: FakeLLM, tmp_path: Path
) -> None:
    """With retrospective on, the Critic produces lessons which are parsed and persisted."""

    def responder(call: FakeCall) -> str:
        if "You are Hawk," in call.system and "Identify 1" in call.user:
            return '{"lessons": ["avoid overclaiming", "cite sources"]}'
        return "content"

    fake_llm.responder = responder
    cfg = _roster(name="retro-run", retrospective=True, retrospective_dir=tmp_path)
    result = await Council(cfg).run("task")

    assert any(p.phase is Phase.RETROSPECTIVE for p in result.phases)

    loaded = retrospectives.load_recent("retro-run", dir_=tmp_path)
    assert len(loaded) == 1
    assert loaded[0].lessons == ["avoid overclaiming", "cite sources"]
    assert loaded[0].task == "task"
    assert loaded[0].cost_usd > 0  # FakeLLM reports per-call cost


async def test_lessons_injected_into_next_runs_prompts(
    fake_llm: FakeLLM, tmp_path: Path
) -> None:
    """Append a retrospective manually, construct a fresh Council with retrospective=true,
    assert each agent's system prompt now contains the prior lesson."""
    retrospectives.append(
        _make_record("inject-test", "prior lesson X"), dir_=tmp_path
    )
    cfg = _roster(
        name="inject-test", retrospective=True, retrospective_dir=tmp_path
    )
    council = Council(cfg)
    for agent in council.agents.values():
        assert "Recent lessons from prior runs" in agent.system_prompt
        assert "prior lesson X" in agent.system_prompt


async def test_unparseable_retrospective_is_not_persisted(
    fake_llm: FakeLLM, tmp_path: Path
) -> None:
    """If the Critic emits garbage, we skip the persist — no empty entries."""
    fake_llm.default = "totally unstructured prose"
    cfg = _roster(name="garbage", retrospective=True, retrospective_dir=tmp_path)
    await Council(cfg).run("task")
    assert retrospectives.load_recent("garbage", dir_=tmp_path) == []


async def test_retrospective_phase_errors_do_not_leak(
    fake_llm: FakeLLM, tmp_path: Path
) -> None:
    """If the Critic LLM call fails during retrospective, the run still completes and
    nothing is persisted for that run."""
    fake_llm.raise_for = {"ollama/phi4-mini:3.8b"}  # Hawk fails → no lessons extracted
    fake_llm.default = "ok"
    cfg = _roster(name="critic-fail", retrospective=True, retrospective_dir=tmp_path)
    result = await Council(cfg).run("task")
    assert result.final_answer
    assert retrospectives.load_recent("critic-fail", dir_=tmp_path) == []


# -----------------------------------------------------------------------------
# default_dir respects XDG_DATA_HOME
# -----------------------------------------------------------------------------


def test_default_dir_uses_xdg_when_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert retrospectives.default_dir() == tmp_path / "ai_agent_council" / "retrospectives"


def test_default_dir_falls_back_to_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    # Path.home() uses $HOME on POSIX.
    assert retrospectives.default_dir() == tmp_path / ".local" / "share" / "ai_agent_council" / "retrospectives"
