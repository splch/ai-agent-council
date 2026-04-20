"""ai-agent-council: a Belbin-roles multi-agent council with cognitive diversity."""

from __future__ import annotations

from .config import AgentConfig, CouncilConfig, load_council_config
from .council import Council, run_council
from .exceptions import (
    AgentError,
    CouncilConfigError,
    CouncilError,
    LLMError,
    PhaseError,
)
from .models import CouncilResult, Message, Phase, PhaseOutput, Role

__version__ = "0.1.0"

__all__ = [
    "AgentConfig",
    "AgentError",
    "Council",
    "CouncilConfig",
    "CouncilConfigError",
    "CouncilError",
    "CouncilResult",
    "LLMError",
    "Message",
    "Phase",
    "PhaseError",
    "PhaseOutput",
    "Role",
    "__version__",
    "load_council_config",
    "run_council",
]
