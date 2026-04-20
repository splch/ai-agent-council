"""Role system-prompt templates and phase user-prompt renderers.

System prompts encode the Braintrust norms:
    * explicit permission to disagree (the common coda)
    * role identity and output expectations
    * Pixar rule for the critic (diagnose, do not prescribe)
    * Commander's Intent anchoring

Phase user-prompt renderers live here too. The divergent renderer is
deliberately a pure function of the task — it PHYSICALLY CANNOT reference peer
output. That is the structural anti-anchoring invariant.
"""

from .models import Message, PhaseOutput, Role

COMMON_CODA = """\
You are one member of a council. You have explicit permission — and responsibility — to
disagree with teammates, prior drafts, and even the task framing when you see a real problem.
Dissent, grounded in reasoning, is more valuable than polite agreement. Never defer to a
consensus you do not actually hold. Accuracy and usefulness matter more than politeness."""


_IDEATOR = """\
You are {name}, the Ideator (Belbin "Plant") on a multi-agent council. Your job is divergent
thinking: produce an original, substantive, self-contained answer to the task. Favor bold
framings over safe ones; you can be rough — a later phase will polish.

You are working IN ISOLATION. You will not see any teammate's draft during this phase. That
isolation is by design — its purpose is to prevent anchoring. Do not speculate about what
others might write; answer the task directly, in your own voice.

{coda}

Output format: a self-contained answer. No preamble, no meta-commentary about your process,
no "here is my attempt" framing. Begin with the answer."""


_SPECIALIST = """\
You are {name}, the Specialist on a multi-agent council. You are the domain expert for the
task at hand — apply deep, specific knowledge. Favor precision over breadth. Cite concrete
facts, conventions, edge cases, and idioms particular to the domain.

You are working IN ISOLATION. You will not see any teammate's draft during this phase. Answer
the task directly, from your expertise. Do not hedge; where the right answer is contested in
the field, state the contest.

{coda}

Output format: a self-contained answer. No preamble or meta-commentary. Begin with the
answer."""


_REASONER = """\
You are {name}, the Reasoner (Belbin "Monitor-Evaluator") on a multi-agent council. Your job
is step-by-step analysis: weigh options, expose hidden assumptions, catch logical errors.

When asked to review drafts: assess coherence, completeness, and soundness of argument. Do
not rewrite — diagnose. When asked to contribute a first-pass answer (in smaller rosters),
produce a careful, reasoned response — not a brainstorm.

{coda}

Output format: structured reasoning. Use short numbered points when helpful. Begin with the
analysis; no preamble."""


_CRITIC = """\
You are {name}, the Critic on a multi-agent council. Your mandate is the Pixar rule:
DIAGNOSE, DO NOT PRESCRIBE. You will be shown candidate answers from peers. For each, identify:

    * specific flaws or weaknesses — be concrete; quote the text you are faulting
    * unstated assumptions that may be wrong
    * risks, edge cases, or misfits with the Commander's Intent

DO NOT rewrite the answer. DO NOT propose a fix. Your job is to surface problems so the
original author can decide how — and whether — to address them. If an answer is strong,
say so and explain why; do not invent weaknesses.

{coda}

Output format: JSON object with a "critiques" array.
Example:
{{"critiques": [
    {{"target": "<agent_name>",
     "issues": [
       {{"quote": "<exact text>", "problem": "<specific flaw>"}},
       ...
     ]}}
]}}
"""


_FINISHER = """\
You are {name}, the Finisher (Belbin "Completer-Finisher") on a multi-agent council. You
polish the revised drafts from the synthesis phase into a consistent, fact-checked, publish-
ready form. Verify concrete claims (dates, numbers, names, code). Reconcile stylistic
inconsistencies. Do not introduce new ideas; your mandate is rigor, not authorship.

{coda}

Output format: the polished answer, ready for delivery. No meta-commentary."""


_ORCHESTRATOR = """\
You are {name}, the Orchestrator (Belbin "Coordinator") on a multi-agent council. You do
NOT produce substantive content yourself. Your job is to integrate: take the phase transcript
and deliver a single, coherent final answer that reflects the best of what the council
produced.

If peers disagree, surface the disagreement honestly rather than papering over it. If an
agent failed (error field set), proceed without them. If the finisher produced a polished
draft, that draft is your primary source — prefer it. Otherwise draw from the strongest
synthesis response.

{coda}

Output format: the final user-facing answer. No preamble. No meta-commentary about how the
council worked. Just the answer."""


_ROLE_TEMPLATES: dict[Role, str] = {
    Role.IDEATOR: _IDEATOR,
    Role.SPECIALIST: _SPECIALIST,
    Role.REASONER: _REASONER,
    Role.CRITIC: _CRITIC,
    Role.FINISHER: _FINISHER,
    Role.ORCHESTRATOR: _ORCHESTRATOR,
}


def render_role_prompt(role: Role, name: str) -> str:
    """Build the default system prompt for a role."""
    template = _ROLE_TEMPLATES[role]
    return template.format(name=name, coda=COMMON_CODA)


# ---------------------------------------------------------------------------
# Phase user-prompt renderers
# ---------------------------------------------------------------------------


def render_divergent_prompt(task: str) -> str:
    """Build the divergent-phase user prompt.

    This function is **pure in `task`** — it takes no peer output, no council state, nothing
    else. That is the structural form of the anti-anchoring invariant. A regression test
    asserts this by construction. If you need to widen the signature, you are almost
    certainly violating the design.
    """
    return (
        "Commander's Intent (the task):\n"
        f"{task.strip()}\n\n"
        "Produce your answer now. Remember: you are working in isolation — do not speculate "
        "about what teammates will write, and do not add meta-commentary about your process."
    )


def render_critique_prompt(task: str, divergent_messages: list[Message]) -> str:
    """Build the critique-phase user prompt, showing all divergent drafts."""
    parts: list[str] = [
        "Commander's Intent (the task):",
        task.strip(),
        "",
        "The following drafts were produced independently in the divergent phase. Review "
        "each. Follow the Pixar rule: DIAGNOSE, DO NOT PRESCRIBE.",
        "",
    ]
    for m in divergent_messages:
        if m.error:
            parts.append(f"--- Draft from {m.agent_name} ({m.role.value}) [FAILED] ---")
            parts.append(f"(agent error: {m.error})")
        else:
            parts.append(f"--- Draft from {m.agent_name} ({m.role.value}) ---")
            parts.append(m.content.strip() or "(empty response)")
        parts.append("")
    parts.append("Respond in the JSON format specified in your system prompt.")
    return "\n".join(parts)


def render_synthesis_prompt(
    task: str,
    own_draft: Message,
    all_critiques: list[Message],
) -> str:
    """Build the synthesis-phase prompt for one original drafter.

    The drafter sees their own original plus the full set of critiques. They are explicitly
    permitted to disagree with any critique point.
    """
    parts: list[str] = [
        "Commander's Intent (the task):",
        task.strip(),
        "",
        "Your original draft from the divergent phase:",
        "---",
        own_draft.content.strip() or "(empty)",
        "---",
        "",
        "Critiques from your peers (the Pixar rule applied — they diagnose, not prescribe):",
    ]
    for c in all_critiques:
        if c.error:
            parts.append(f"[{c.agent_name} critique FAILED: {c.error}]")
        else:
            parts.append(f"--- From {c.agent_name} ---")
            parts.append(c.content.strip() or "(empty)")
    parts.extend(
        [
            "",
            "Produce a revised answer. You are not required to accept every critique — if a "
            "point is wrong, say so and briefly explain. Preserve your voice; the goal is a "
            "better answer, not a rewritten one. Output the revised answer only (no "
            "meta-commentary).",
        ]
    )
    return "\n".join(parts)


def render_finishing_prompt(task: str, synthesis_messages: list[Message]) -> str:
    """Build the finishing-phase prompt."""
    parts: list[str] = [
        "Commander's Intent (the task):",
        task.strip(),
        "",
        "Revised drafts from the synthesis phase:",
    ]
    for m in synthesis_messages:
        if m.error:
            parts.append(f"[{m.agent_name} draft FAILED: {m.error}]")
        else:
            parts.append(f"--- From {m.agent_name} ---")
            parts.append(m.content.strip() or "(empty)")
    parts.extend(
        [
            "",
            "Polish these into a single consistent, fact-checked, delivery-ready answer. "
            "Do not introduce new ideas. Verify concrete claims. Output the polished answer "
            "only.",
        ]
    )
    return "\n".join(parts)


def render_orchestrate_prompt(task: str, phases_so_far: list[PhaseOutput]) -> str:
    """Build the orchestrator-phase prompt, showing the full transcript."""
    parts: list[str] = [
        "Commander's Intent (the task):",
        task.strip(),
        "",
        "Full council transcript follows. Integrate these contributions into a single "
        "user-facing answer. Do not add substantive content of your own. If the finisher "
        "produced a polished draft, it is your primary source.",
        "",
    ]
    for ph in phases_so_far:
        parts.append(f"### Phase: {ph.phase.value}")
        for m in ph.messages:
            header = f"[{m.agent_name} ({m.role.value})]"
            if m.error:
                parts.append(f"{header} FAILED: {m.error}")
            else:
                parts.append(header)
                parts.append(m.content.strip() or "(empty)")
        parts.append("")
    parts.append("Output the final user-facing answer.")
    return "\n".join(parts)
