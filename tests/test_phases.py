"""Phase-runner tests. The anti-anchoring invariant is the load-bearing one."""

import inspect

import pytest

from ai_agent_council.config import CouncilConfig
from ai_agent_council.council import Council
from ai_agent_council.models import Phase
from ai_agent_council.phases import (
    run_critique,
    run_divergent,
    run_finishing,
    run_orchestrate,
    run_synthesis,
)
from ai_agent_council.prompts import render_divergent_prompt

from .fakes import FakeCall, FakeLLM


def test_divergent_prompt_signature_is_anti_anchoring() -> None:
    """Structural check: `render_divergent_prompt` takes `task` only. This is the static form
    of the anti-anchoring invariant — peer output cannot be passed in, because the signature
    physically does not accept it."""
    sig = inspect.signature(render_divergent_prompt)
    assert list(sig.parameters) == ["task"]
    assert sig.parameters["task"].annotation is str


async def test_divergent_anti_anchoring_across_agents(
    fake_llm: FakeLLM, minimal_council_config: CouncilConfig
) -> None:
    """The behavioral form: run the divergent phase with a FakeLLM that produces distinctive
    per-agent responses, then assert no agent's `user` prompt contains another agent's
    response. This is the regression test for the Braintrust anti-anchoring rule.
    """
    unique_markers = {
        "Muse": "MARKER_MUSE_RESPONSE_αβγδ",
        "Judge": "MARKER_JUDGE_RESPONSE_εζηθ",
    }

    def responder(call: FakeCall) -> str:
        for name, marker in unique_markers.items():
            if f"You are {name}," in call.system:
                return marker
        return "irrelevant"

    fake_llm.responder = responder

    council = Council(minimal_council_config)
    phase = await run_divergent(council, "Write a haiku about recursion.")

    # Determine which FakeCalls came from the divergent phase by matching the system prompts
    # of divergers against the FakeLLM's recorded calls.
    diverging_names = {m.agent_name for m in phase.messages}
    divergent_calls = [
        c for c in fake_llm.calls if any(f"You are {n}," in c.system for n in diverging_names)
    ]
    assert len(divergent_calls) == len(diverging_names)

    # For each pair of divergent calls, neither's user prompt may contain the other's marker.
    for call in divergent_calls:
        for name, marker in unique_markers.items():
            # The call for agent X must NOT contain agent Y's marker in its user prompt.
            if f"You are {name}," not in call.system:
                assert marker not in call.user, (
                    f"anti-anchoring violated: agent prompt leaked {name!r}'s draft"
                )


async def test_divergent_falls_back_to_reasoner_when_only_one_diverger(
    fake_llm: FakeLLM, minimal_council_config: CouncilConfig
) -> None:
    """The 4-agent minimal roster has only one Ideator and no Specialist. The divergent phase
    should enlist the Reasoner as a second voice to preserve "multiple independent drafts"."""
    council = Council(minimal_council_config)
    phase = await run_divergent(council, "anything")
    names = {m.agent_name for m in phase.messages}
    assert "Muse" in names
    assert "Judge" in names  # Reasoner enlisted as fallback
    assert len(phase.messages) == 2
    assert all(m.phase is Phase.DIVERGENT for m in phase.messages)


async def test_critique_sees_all_divergent_drafts(
    fake_llm: FakeLLM, minimal_council_config: CouncilConfig
) -> None:
    """The inverse assertion: the critique phase's user prompt MUST include each divergent
    draft. If this ever regresses, the Critic has nothing to critique."""

    def responder(call: FakeCall) -> str:
        # "isolation" only appears in the divergent-phase user prompt.
        is_divergent = "isolation" in call.user
        if "You are Muse," in call.system and is_divergent:
            return "UNIQUE_MUSE_DRAFT_CONTENT"
        if "You are Judge," in call.system and is_divergent:
            return "UNIQUE_JUDGE_DRAFT_CONTENT"
        return "critique text"

    fake_llm.responder = responder

    council = Council(minimal_council_config)
    divergent = await run_divergent(council, "task")
    await run_critique(council, "task", divergent)

    critic_calls = [c for c in fake_llm.calls if "You are Hawk," in c.system]
    assert critic_calls, "critic was not invoked"
    critic_prompt = critic_calls[-1].user
    assert "UNIQUE_MUSE_DRAFT_CONTENT" in critic_prompt
    assert "UNIQUE_JUDGE_DRAFT_CONTENT" in critic_prompt


async def test_synthesis_drafters_see_critiques(
    fake_llm: FakeLLM, minimal_council_config: CouncilConfig
) -> None:
    def responder(call: FakeCall) -> str:
        if "You are Hawk," in call.system:
            return "UNIQUE_CRITIC_FEEDBACK"
        if (
            "You are Judge," in call.system
            and "critique" in call.user.lower()
            and "Drafts" in call.user
        ):
            # Reasoner-as-reviewer in critique phase
            return "UNIQUE_REASONER_FEEDBACK"
        return "revised draft"

    fake_llm.responder = responder

    council = Council(minimal_council_config)
    divergent = await run_divergent(council, "task")
    critique = await run_critique(council, "task", divergent)
    synthesis = await run_synthesis(council, "task", divergent, critique)
    assert len(synthesis.messages) == len(divergent.messages)

    muse_synth_calls = [
        c
        for c in fake_llm.calls
        if "You are Muse," in c.system and "Critiques from your peers" in c.user
    ]
    assert muse_synth_calls, "muse was not given the synthesis prompt"
    assert "UNIQUE_CRITIC_FEEDBACK" in muse_synth_calls[-1].user


async def test_finishing_skipped_when_no_finisher(
    fake_llm: FakeLLM, minimal_council_config: CouncilConfig
) -> None:
    council = Council(minimal_council_config)
    divergent = await run_divergent(council, "task")
    critique = await run_critique(council, "task", divergent)
    synthesis = await run_synthesis(council, "task", divergent, critique)
    finishing = await run_finishing(council, "task", synthesis)
    assert finishing.messages == []


async def test_orchestrate_emits_single_message(
    fake_llm: FakeLLM, minimal_council_config: CouncilConfig
) -> None:
    council = Council(minimal_council_config)
    divergent = await run_divergent(council, "task")
    critique = await run_critique(council, "task", divergent)
    synthesis = await run_synthesis(council, "task", divergent, critique)
    orch = await run_orchestrate(council, "task", [divergent, critique, synthesis])
    assert len(orch.messages) == 1
    assert orch.messages[0].agent_name == "Scribe"


async def test_orchestrate_prompt_contains_whole_transcript(
    fake_llm: FakeLLM, minimal_council_config: CouncilConfig
) -> None:
    def responder(call: FakeCall) -> str:
        name = _name_of(call.system)
        return f"RESPONSE_FROM_{name}"

    fake_llm.responder = responder
    council = Council(minimal_council_config)
    divergent = await run_divergent(council, "task")
    critique = await run_critique(council, "task", divergent)
    synthesis = await run_synthesis(council, "task", divergent, critique)
    await run_orchestrate(council, "task", [divergent, critique, synthesis])

    scribe_calls = [c for c in fake_llm.calls if "You are Scribe," in c.system]
    assert scribe_calls
    prompt = scribe_calls[-1].user
    # transcript should mention each phase name
    for name in ("divergent", "critique", "synthesis"):
        assert name in prompt


def _name_of(system: str) -> str:
    head = system.strip().splitlines()[0]
    _, _, rest = head.partition("You are ")
    return rest.split(",", 1)[0].strip() or "unknown"


@pytest.mark.asyncio
async def test_phases_return_phase_output_with_correct_phase(
    fake_llm: FakeLLM, minimal_council_config: CouncilConfig
) -> None:
    council = Council(minimal_council_config)
    d = await run_divergent(council, "x")
    assert d.phase is Phase.DIVERGENT
    c = await run_critique(council, "x", d)
    assert c.phase is Phase.CRITIQUE
    s = await run_synthesis(council, "x", d, c)
    assert s.phase is Phase.SYNTHESIS
    f = await run_finishing(council, "x", s)
    assert f.phase is Phase.FINISHING
    o = await run_orchestrate(council, "x", [d, c, s, f])
    assert o.phase is Phase.ORCHESTRATE
