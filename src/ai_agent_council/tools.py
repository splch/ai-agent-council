"""Built-in tools the Finisher (or any agent) can call during a turn.

Design
------
A tool is a named, schema-documented function that returns a string. The set available to
a given agent is controlled by `AgentConfig.tools` — a list of registered tool names.

Safety stance
-------------
The shipped tools are deliberately narrow and sandboxed at the source:

    * `current_time`        — stdlib only, no I/O
    * `calculate`           — AST-validated arithmetic, no builtins, no attribute access
    * `read_file`           — refuses paths outside the current working directory tree,
                              caps output at 8 KB

Arbitrary code execution, unrestricted web access, and shell calls are **not** included.
Users who want them can register their own tools at runtime via `register`, at which point
the responsibility for sandboxing is theirs.
"""

import ast
import operator as op
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Tool:
    """A callable exposed to an LLM via the OpenAI tool-calling protocol."""

    name: str
    description: str
    parameters_schema: dict[str, Any]  # JSONSchema under `function.parameters`
    fn: Callable[..., str]

    def to_openai_schema(self) -> dict[str, Any]:
        """Serialize to the OpenAI/LiteLLM `tools=[...]` entry shape."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }


_REGISTRY: dict[str, Tool] = {}


def register(tool: Tool) -> Tool:
    """Add a tool to the process-wide registry. Returns the tool for chaining."""
    _REGISTRY[tool.name] = tool
    return tool


def get(name: str) -> Tool:
    """Look up a tool by name. Raises KeyError with a list of known names if missing."""
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"unknown tool {name!r}; registered: {known}") from None


def resolve(names: list[str]) -> list[Tool]:
    return [get(n) for n in names]


def available() -> list[str]:
    """List currently registered tool names."""
    return sorted(_REGISTRY)


# -----------------------------------------------------------------------------
# Built-in tools
# -----------------------------------------------------------------------------


def _current_time() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


register(
    Tool(
        name="current_time",
        description="Return the current UTC date and time as an ISO 8601 string.",
        parameters_schema={"type": "object", "properties": {}, "required": []},
        fn=_current_time,
    )
)


_ARITH_OPS: dict[type[ast.operator] | type[ast.unaryop], Callable[..., float]] = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.FloorDiv: op.floordiv,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def _eval_arith(node: ast.AST) -> float:
    match node:
        case ast.Expression(body=b):
            return _eval_arith(b)
        case ast.Constant(value=v) if isinstance(v, int | float):
            return v
        case ast.BinOp(op=operator, left=left, right=right) if type(operator) in _ARITH_OPS:
            return _ARITH_OPS[type(operator)](_eval_arith(left), _eval_arith(right))
        case ast.UnaryOp(op=operator, operand=operand) if type(operator) in _ARITH_OPS:
            return _ARITH_OPS[type(operator)](_eval_arith(operand))
        case _:
            raise ValueError(f"unsupported expression: {ast.dump(node)}")


def _calculate(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval_arith(tree))
    except (ValueError, SyntaxError, ZeroDivisionError) as e:
        return f"error: {e}"


register(
    Tool(
        name="calculate",
        description=(
            "Evaluate an arithmetic expression. "
            "Supports numbers, + - * / % ** // and parentheses. No variables or functions."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "An arithmetic expression, e.g. '2 + 2 * (3 - 1)'.",
                }
            },
            "required": ["expression"],
        },
        fn=_calculate,
    )
)


_READ_FILE_CAP = 8192


def _read_file(path: str) -> str:
    p = Path(path).expanduser().resolve()
    cwd = Path.cwd().resolve()
    if cwd != p and cwd not in p.parents:
        return f"error: refusing to read {path!r}: outside the working directory"
    if not p.is_file():
        return f"error: {path!r} is not a regular file"
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"error: {e}"
    return text[:_READ_FILE_CAP]


register(
    Tool(
        name="read_file",
        description=(
            f"Read a text file from the current working directory tree. "
            f"Output capped at {_READ_FILE_CAP} bytes."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to a file inside the current working directory.",
                }
            },
            "required": ["path"],
        },
        fn=_read_file,
    )
)
