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
    * `fetch_url`           — HTTP GET only, http/https schemes only, follows at most 3
                              redirects, 10s timeout, 64 KB response cap, rejects
                              private / link-local / loopback hosts (SSRF protection)

Explicitly NOT shipped:

    * Arbitrary Python execution — cannot be safely sandboxed without platform-
      specific tooling (firejail/nsjail/bubblewrap). Users who need it can add a
      tool that shells out to their own sandbox.
    * Web search — requires an API key choice and varies in quality across
      backends. Users should register their own wrapper around whatever
      search service they trust.

Any additional tools a user registers are their own responsibility to sandbox.
"""

import ast
import ipaddress
import operator as op
import socket
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


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


def register(tool: Tool, *, overwrite: bool = False) -> Tool:
    """Add a tool to the process-wide registry. Returns the tool for chaining.

    Refuses by default to overwrite an existing name — this catches the "user registers
    a custom tool whose name happens to match a built-in" typo class. Pass
    `overwrite=True` to replace intentionally (tests, hot-reload workflows)."""
    if not overwrite and tool.name in _REGISTRY:
        raise ValueError(f"tool {tool.name!r} already registered; pass overwrite=True to replace")
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


def _safe_pow(base: float, exp: float) -> float:
    """`9 ** 9 ** 9` (and similar nested exponents) can hang CPython for minutes while
    allocating a single gigantic bignum — during which the whole asyncio loop stalls,
    since tool calls run inline on `await tool.fn(**args)`. Bound both operands so an
    adversarial model can't freeze the council with one expression."""
    if abs(exp) > 1000 or abs(base) > 10**100:
        raise ValueError("calculator refuses exponent/base that large")
    return base**exp


_ARITH_OPS: dict[type[ast.operator | ast.unaryop], Callable[..., float]] = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Mod: op.mod,
    ast.Pow: _safe_pow,
    ast.FloorDiv: op.floordiv,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def _eval_arith(node: ast.AST) -> float:
    match node:
        case ast.Expression(body=b):
            return _eval_arith(b)
        case ast.Constant(value=v) if type(v) in (int, float):
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


# -----------------------------------------------------------------------------
# fetch_url — HTTP GET with SSRF protection
# -----------------------------------------------------------------------------
#
# The research's "verification gates beat critic opinions" argument depends on
# the Finisher being able to check claims against an external source of truth.
# fetch_url is the minimum viable verifier: the Finisher sees a claim, pulls
# the referenced page, and can then reason against the retrieved text.
#
# Safety constraints are non-negotiable because the URL comes from an LLM:
#   * scheme allowlist (http/https only)
#   * DNS resolution + private/loopback/link-local IP rejection (SSRF)
#   * maximum 3 redirects, each re-validated
#   * 10 second total timeout
#   * 64 KB hard cap on response size; 8 KB of text returned to the model

_FETCH_TIMEOUT_S = 10.0
_FETCH_MAX_REDIRECTS = 3
_FETCH_MAX_BYTES = 64 * 1024
_FETCH_MAX_RETURN = 8 * 1024
_ALLOWED_SCHEMES = frozenset({"http", "https"})


def _is_public_address(host: str) -> bool:
    """Resolve `host` and verify every returned address is routable-public.
    Returns False on any private/loopback/link-local/reserved address.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for family, _type, _proto, _canon, sockaddr in infos:
        if family not in (socket.AF_INET, socket.AF_INET6):
            continue
        ip = ipaddress.ip_address(sockaddr[0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return False
    return True


def _fetch_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        return f"error: only http/https URLs are allowed (got {parsed.scheme!r})"
    if not parsed.hostname:
        return f"error: no hostname in {url!r}"
    if not _is_public_address(parsed.hostname):
        return f"error: refusing to fetch private/loopback address {parsed.hostname!r}"

    current_url = url
    for _ in range(_FETCH_MAX_REDIRECTS + 1):
        try:
            req = urllib.request.Request(
                current_url,
                headers={"User-Agent": "ai-agent-council/fetch_url"},
            )
            # manual redirect handling so every hop re-validates the SSRF check
            opener = urllib.request.build_opener(_NoRedirectHandler())
            with opener.open(req, timeout=_FETCH_TIMEOUT_S) as resp:
                status = resp.status
                if status in (301, 302, 303, 307, 308):
                    loc = resp.headers.get("Location")
                    if not loc:
                        return f"error: redirect with no Location header from {current_url!r}"
                    current_url = urllib.parse.urljoin(current_url, loc)
                    re_parsed = urlparse(current_url)
                    if re_parsed.scheme not in _ALLOWED_SCHEMES:
                        return f"error: redirect to disallowed scheme {re_parsed.scheme!r}"
                    if not re_parsed.hostname or not _is_public_address(re_parsed.hostname):
                        return f"error: redirect to non-public host {re_parsed.hostname!r}"
                    continue
                body = resp.read(_FETCH_MAX_BYTES + 1)
                break
        except urllib.error.HTTPError as e:
            return f"error: HTTP {e.code} from {current_url!r}"
        except (urllib.error.URLError, OSError, ValueError) as e:
            return f"error: {type(e).__name__}: {e}"
    else:
        return f"error: too many redirects fetching {url!r}"

    if len(body) > _FETCH_MAX_BYTES:
        body = body[:_FETCH_MAX_BYTES]
    try:
        text = body.decode("utf-8", errors="replace")
    except Exception as e:
        return f"error: decode failed: {e}"
    return text[:_FETCH_MAX_RETURN]


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """urllib follows redirects by default. We handle them ourselves so every hop
    re-runs the SSRF / scheme checks."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        return None


register(
    Tool(
        name="fetch_url",
        description=(
            "HTTP GET a public http/https URL and return up to 8 KB of text. "
            "Refuses private/loopback/link-local hosts, non-http schemes, and more than "
            f"{_FETCH_MAX_REDIRECTS} redirects. Times out after {_FETCH_TIMEOUT_S:.0f}s."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "A fully-qualified http:// or https:// URL.",
                }
            },
            "required": ["url"],
        },
        fn=_fetch_url,
    )
)
