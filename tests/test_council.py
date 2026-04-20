"""End-to-end council runs with FakeLLM."""

import pytest

from ai_agent_council.config import CouncilConfig
from ai_agent_council.council import Council, run_council
from ai_agent_council.models import Phase, PhaseOutput

from .fakes import FakeCall, FakeLLM, agent_from_system


def _responder_from_role(call: FakeCall) -> str:
    return f"Response from {agent_from_system(call.system)}"


async def test_end_to_end_minimal_roster(
    fake_llm: FakeLLM, minimal_council_config: CouncilConfig
) -> None:
    fake_llm.responder = _responder_from_role
    council = Council(minimal_council_config)
    result = await council.run("Write a haiku about recursion.")

    # No finisher in minimal roster → 4 phases expected
    phases = [p.phase for p in result.phases]
    assert phases == [Phase.DIVERGENT, Phase.CRITIQUE, Phase.SYNTHESIS, Phase.ORCHESTRATE]
    assert result.final_answer == "Response from Scribe"
    assert result.config_digest  # non-empty hash
    assert result.started_at <= result.finished_at


async def test_end_to_end_full_roster_includes_finishing(
    fake_llm: FakeLLM, full_council_config: CouncilConfig
) -> None:
    fake_llm.responder = _responder_from_role
    council = Council(full_council_config)
    result = await council.run("task")

    phases = [p.phase for p in result.phases]
    assert phases == [
        Phase.DIVERGENT,
        Phase.CRITIQUE,
        Phase.SYNTHESIS,
        Phase.FINISHING,
        Phase.ORCHESTRATE,
    ]
    # Divergent: Ideator + Specialist (no fallback needed)
    divergent = result.phases[0]
    diverger_names = {m.agent_name for m in divergent.messages}
    assert diverger_names == {"Muse", "Adept"}


async def test_streaming_callback_invoked_per_phase(
    fake_llm: FakeLLM, minimal_council_config: CouncilConfig
) -> None:
    fake_llm.default = "content"
    council = Council(minimal_council_config)
    streamed: list[PhaseOutput] = []
    await council.run("task", stream=streamed.append)
    assert [p.phase for p in streamed] == [
        Phase.DIVERGENT,
        Phase.CRITIQUE,
        Phase.SYNTHESIS,
        Phase.ORCHESTRATE,
    ]


async def test_broken_stream_does_not_abort(
    fake_llm: FakeLLM, minimal_council_config: CouncilConfig
) -> None:
    fake_llm.default = "content"
    council = Council(minimal_council_config)

    def broken(_: PhaseOutput) -> None:
        raise RuntimeError("observer explode")

    result = await council.run("task", stream=broken)
    assert result.final_answer is not None


async def test_transcript_is_json_serializable(
    fake_llm: FakeLLM, minimal_council_config: CouncilConfig
) -> None:
    fake_llm.default = "content"
    council = Council(minimal_council_config)
    result = await council.run("task")
    blob = result.model_dump_json()
    assert '"phases"' in blob
    assert '"final_answer"' in blob


async def test_library_run_council_helper(
    fake_llm: FakeLLM, minimal_council_config: CouncilConfig
) -> None:
    fake_llm.default = "x"
    result = await run_council("task", minimal_council_config)
    assert result.final_answer == "x"


async def test_agent_error_captured_in_transcript(
    fake_llm: FakeLLM, minimal_council_config: CouncilConfig
) -> None:
    """If one agent's LLM call fails, the run still completes and the failure is in the
    transcript as an explicit Message.error."""
    fake_llm.raise_for = {"ollama/gemma3:2b"}  # Muse's model → Muse fails
    fake_llm.default = "ok"
    result = await Council(minimal_council_config).run("task")

    muse_messages = [m for ph in result.phases for m in ph.messages if m.agent_name == "Muse"]
    assert any(m.error for m in muse_messages)
    assert result.final_answer  # orchestrator still produced an answer


async def test_total_cost_sums_per_message_cost(
    fake_llm: FakeLLM, minimal_council_config: CouncilConfig
) -> None:
    """total_cost_usd aggregates cost_usd across all phase messages."""
    fake_llm.default = "x"
    fake_llm.cost_per_call = 0.01
    result = await Council(minimal_council_config).run("task")
    call_count = sum(len(p.messages) for p in result.phases)
    assert call_count > 0
    assert result.total_cost_usd == pytest.approx(0.01 * call_count)


async def test_failed_messages_contribute_zero_cost(
    fake_llm: FakeLLM, minimal_council_config: CouncilConfig
) -> None:
    fake_llm.raise_for = {"ollama/gemma3:2b"}  # Muse fails → cost_usd is None
    fake_llm.default = "ok"
    fake_llm.cost_per_call = 0.005
    result = await Council(minimal_council_config).run("task")
    muse_msgs = [m for ph in result.phases for m in ph.messages if m.agent_name == "Muse"]
    assert all(m.cost_usd is None for m in muse_msgs)
    # total is finite and non-negative
    assert result.total_cost_usd > 0


async def test_total_tokens_aggregated(
    fake_llm: FakeLLM, minimal_council_config: CouncilConfig
) -> None:
    fake_llm.default = "x"
    result = await Council(minimal_council_config).run("task")
    tin, tout = result.total_tokens
    count = sum(len(p.messages) for p in result.phases if p.messages)
    assert tin == count  # FakeLLM reports 1 in per call
    assert tout == count


async def test_tokens_callback_is_driven_live(
    fake_llm: FakeLLM, minimal_council_config: CouncilConfig
) -> None:
    """With a `tokens` callback, each agent's content is streamed as it's produced."""
    fake_llm.responder = lambda call: f"RESPONSE_FROM_{agent_from_system(call.system)}"
    received: list[tuple[str, str]] = []

    def on_token(agent: str, chunk: str) -> None:
        received.append((agent, chunk))

    await Council(minimal_council_config).run("task", tokens=on_token)

    # At least one chunk per agent per phase they participate in. Concatenating a
    # given agent's chunks reproduces the full response.
    by_agent: dict[str, str] = {}
    for agent, chunk in received:
        by_agent[agent] = by_agent.get(agent, "") + chunk
    assert "Muse" in by_agent
    assert by_agent["Muse"].startswith("RESPONSE_FROM_Muse")
    assert "Scribe" in by_agent  # orchestrator also streams
