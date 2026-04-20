"""Agent: a frozen dataclass pairing an AgentConfig with its resolved system prompt."""

from dataclasses import dataclass
from typing import Self

from . import llm
from .config import AgentConfig
from .exceptions import LLMError
from .llm import StreamHandler
from .models import Message, Phase
from .prompts import render_role_prompt


@dataclass(slots=True, frozen=True)
class Agent:
    """A single council member.

    Errors during `respond` are CAPTURED into the returned Message (with `error` set and
    `content` empty) rather than raised. Rationale: one dead agent should not abort a run;
    the orchestrator is prompted to cope with missing contributions, and the transcript
    always surfaces errors explicitly.
    """

    config: AgentConfig
    system_prompt: str

    @classmethod
    def from_config(cls, cfg: AgentConfig) -> Self:
        prompt = cfg.system_prompt if cfg.system_prompt else render_role_prompt(cfg.role, cfg.name)
        return cls(config=cfg, system_prompt=prompt)

    async def respond(
        self,
        user_prompt: str,
        *,
        phase: Phase,
        stream_handler: StreamHandler | None = None,
    ) -> Message:
        cfg = self.config
        try:
            content, meta = await llm.complete(
                model=cfg.model,
                system=self.system_prompt,
                user=user_prompt,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                timeout_s=cfg.timeout_s,
                json_mode=cfg.json_mode,
                stream_handler=stream_handler,
            )
            return Message(
                role=cfg.role,
                agent_name=cfg.name,
                phase=phase,
                content=content,
                model=cfg.model,
                temperature=cfg.temperature,
                tokens_in=meta.get("tokens_in"),
                tokens_out=meta.get("tokens_out"),
                latency_ms=meta.get("latency_ms"),
                cost_usd=meta.get("cost_usd"),
            )
        except LLMError as e:
            return Message(
                role=cfg.role,
                agent_name=cfg.name,
                phase=phase,
                content="",
                model=cfg.model,
                temperature=cfg.temperature,
                error=f"{type(e).__name__}: {e}",
            )
