"""Tests for the built-in tool registry and safe-tool implementations."""

from pathlib import Path

import pytest

from ai_agent_council import tools


def test_builtins_registered() -> None:
    """The three shipped tools must be available by default."""
    available = tools.available()
    assert "current_time" in available
    assert "calculate" in available
    assert "read_file" in available


def test_openai_schema_shape() -> None:
    tool = tools.get("calculate")
    schema = tool.to_openai_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "calculate"
    assert "expression" in schema["function"]["parameters"]["properties"]


def test_resolve_returns_tools_in_order() -> None:
    resolved = tools.resolve(["calculate", "current_time"])
    assert [t.name for t in resolved] == ["calculate", "current_time"]


def test_resolve_unknown_raises() -> None:
    with pytest.raises(KeyError, match="unknown tool"):
        tools.resolve(["nonexistent"])


def test_current_time_returns_iso_string() -> None:
    out = tools.get("current_time").fn()
    assert "T" in out  # ISO 8601 has a 'T' between date and time
    assert out.endswith(("+00:00", "Z"))


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 2", "4"),
        ("2 * (3 + 4)", "14"),
        ("10 / 4", "2.5"),
        ("10 % 3", "1"),
        ("2 ** 10", "1024"),
        ("-5 + 3", "-2"),
        ("17 // 5", "3"),
    ],
)
def test_calculate_arithmetic(expression: str, expected: str) -> None:
    assert tools.get("calculate").fn(expression=expression) == expected


def test_calculate_rejects_names() -> None:
    """The whole point of the AST walker is that bare names / function calls are refused."""
    result = tools.get("calculate").fn(expression="__import__('os').system('ls')")
    assert result.startswith("error:")


def test_calculate_rejects_attribute_access() -> None:
    assert tools.get("calculate").fn(expression="object.__subclasses__()").startswith("error:")


def test_calculate_handles_divide_by_zero() -> None:
    out = tools.get("calculate").fn(expression="1 / 0")
    assert out.startswith("error:")


@pytest.mark.parametrize(
    "expression",
    [
        "9 ** 9 ** 9",  # nested pow — would hang without the cap
        "10 ** 5000",  # huge exponent alone
        "(10**200) ** 10",  # huge base
    ],
)
def test_calculate_refuses_huge_powers(expression: str) -> None:
    """Without a bound, an adversarial model could freeze the whole event loop with one
    expression. The cap must fire before CPython starts allocating a gigantic bignum."""
    out = tools.get("calculate").fn(expression=expression)
    assert out.startswith("error:"), f"expected refusal, got {out[:40]!r}"


def test_read_file_refuses_paths_outside_cwd(tmp_path: Path) -> None:
    """The sandbox is intentional — no `/etc/passwd` no matter what the LLM asks."""
    out = tools.get("read_file").fn(path="/etc/passwd")
    # Either the path is outside cwd or the file doesn't exist — either is a refusal.
    assert out.startswith("error:")


def test_read_file_reads_cwd_subtree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "note.txt"
    f.write_text("hello\n", encoding="utf-8")
    out = tools.get("read_file").fn(path=str(f))
    assert out == "hello\n"


def test_read_file_caps_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "big.txt"
    f.write_text("x" * 20_000, encoding="utf-8")
    out = tools.get("read_file").fn(path=str(f))
    assert len(out) <= 8192


def test_read_file_rejects_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    out = tools.get("read_file").fn(path=str(tmp_path))
    assert out.startswith("error:")


def test_register_is_idempotent_for_same_name() -> None:
    """Registering under an existing name replaces the prior entry (useful for tests)."""
    custom = tools.Tool(
        name="__test_override__",
        description="test",
        parameters_schema={"type": "object", "properties": {}, "required": []},
        fn=lambda: "first",
    )
    tools.register(custom)
    assert tools.get("__test_override__").fn() == "first"
    replacement = tools.Tool(
        name="__test_override__",
        description="test",
        parameters_schema={"type": "object", "properties": {}, "required": []},
        fn=lambda: "second",
    )
    tools.register(replacement)
    assert tools.get("__test_override__").fn() == "second"
