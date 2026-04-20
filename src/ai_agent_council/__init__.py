"""ai-agent-council: a Belbin-roles multi-agent council with cognitive diversity."""

from .config import AgentConfig, CouncilConfig, load_council_config
from .council import Council, run_council
from .exceptions import CouncilConfigError, CouncilError, LLMError
from .models import CouncilResult, Message, Phase, PhaseOutput, Role

__version__ = "0.1.0"

__all__ = [
    "AgentConfig",
    "Council",
    "CouncilConfig",
    "CouncilConfigError",
    "CouncilError",
    "CouncilResult",
    "LLMError",
    "Message",
    "Phase",
    "PhaseOutput",
    "Role",
    "__version__",
    "load_council_config",
    "run_council",
]
