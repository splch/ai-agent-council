"""Runtime data models. Pydantic-first — serialization is `model_dump_json`."""

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Role(StrEnum):
    """Belbin-adapted roles. One agent may hold only one role."""

    IDEATOR = "ideator"
    REASONER = "reasoner"
    SPECIALIST = "specialist"
    CRITIC = "critic"
    FINISHER = "finisher"
    ORCHESTRATOR = "orchestrator"


class Phase(StrEnum):
    """Braintrust workflow phases, in canonical order."""

    RESTATE = "restate"
    DIVERGENT = "divergent"
    CRITIQUE = "critique"
    STEELMAN = "steelman"
    SYNTHESIS = "synthesis"
    FINISHING = "finishing"
    ORCHESTRATE = "orchestrate"
    RETROSPECTIVE = "retrospective"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ToolCall(BaseModel):
    """A single tool invocation made during an agent's turn."""

    model_config = ConfigDict(frozen=True)

    name: str
    arguments: dict[str, object]
    result: str
    error: str | None = None


class Message(BaseModel):
    """A single agent utterance produced during a phase."""

    model_config = ConfigDict(frozen=True)

    role: Role
    agent_name: str
    phase: Phase
    content: str
    model: str
    temperature: float
    tokens_in: int | None = None
    tokens_out: int | None = None
    latency_ms: int | None = None
    cost_usd: float | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    error: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)


class PhaseOutput(BaseModel):
    """All messages produced during a single phase."""

    model_config = ConfigDict(frozen=True)

    phase: Phase
    messages: list[Message]
    elapsed_ms: int


class CouncilResult(BaseModel):
    """The final deliverable plus full transcript."""

    model_config = ConfigDict(frozen=True)

    task: str
    final_answer: str
    phases: list[PhaseOutput]
    config_digest: str
    started_at: datetime
    finished_at: datetime

    @property
    def total_cost_usd(self) -> float:
        """Sum of per-message cost across the whole run (USD). Zero for models LiteLLM
        doesn't price (e.g. local Ollama)."""
        return sum((m.cost_usd or 0.0) for ph in self.phases for m in ph.messages)

    @property
    def total_tokens(self) -> tuple[int, int]:
        """(tokens_in, tokens_out) summed across the run. Unknowns count as zero."""
        tin = sum((m.tokens_in or 0) for ph in self.phases for m in ph.messages)
        tout = sum((m.tokens_out or 0) for ph in self.phases for m in ph.messages)
        return tin, tout
