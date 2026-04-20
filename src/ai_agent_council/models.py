"""Runtime data models. Pydantic-first — serialization is `model_dump_json`."""

from __future__ import annotations

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

    DIVERGENT = "divergent"
    CRITIQUE = "critique"
    SYNTHESIS = "synthesis"
    FINISHING = "finishing"
    ORCHESTRATE = "orchestrate"


def _utcnow() -> datetime:
    return datetime.now(UTC)


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
