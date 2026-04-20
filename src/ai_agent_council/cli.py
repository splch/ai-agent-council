"""Typer CLI: `council run | init | validate`."""

from __future__ import annotations

import asyncio
import shutil
import sys
from importlib import resources
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from .config import load_council_config
from .council import Council
from .exceptions import CouncilConfigError
from .models import PhaseOutput
from .transcript import write_transcript

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="A Belbin-roles multi-agent council running the Braintrust workflow.",
)

_console = Console(stderr=True)
_stdout = Console()


_TEMPLATE_PACKAGE = "ai_agent_council.templates"

AVAILABLE_TEMPLATES = (
    "laptop-4agent",
    "workstation-4agent",
    "server-6agent",
)


class RichPhasePrinter:
    """Stream phase output to stdout as each phase completes."""

    def __init__(self, *, quiet: bool) -> None:
        self.quiet = quiet

    def __call__(self, phase: PhaseOutput) -> None:
        if self.quiet:
            return
        _stdout.print(Rule(f"[bold]{phase.phase.value.upper()}[/bold] ({phase.elapsed_ms} ms)"))
        if not phase.messages:
            _stdout.print("[dim](no participants for this phase)[/dim]")
            return
        for msg in phase.messages:
            title = f"{msg.agent_name} [{msg.role.value}] — {msg.model}"
            if msg.error:
                body = f"[red]ERROR:[/red] {msg.error}"
            else:
                body = msg.content or "[dim](empty)[/dim]"
            _stdout.print(Panel.fit(body, title=title, border_style="blue"))

    def final(self, final_answer: str) -> None:
        if self.quiet:
            _stdout.print(final_answer)
            return
        _stdout.print(Rule("[bold green]FINAL ANSWER[/bold green]"))
        _stdout.print(Panel.fit(final_answer, border_style="green"))


def _resolve_template(name: str) -> Path:
    if name not in AVAILABLE_TEMPLATES:
        raise typer.BadParameter(
            f"unknown template {name!r}; available: {', '.join(AVAILABLE_TEMPLATES)}"
        )
    resource = resources.files(_TEMPLATE_PACKAGE).joinpath(f"{name}.yaml")
    with resources.as_file(resource) as path:
        return Path(path)


@app.command()
def run(
    task: Annotated[str, typer.Argument(help="Commander's Intent — what you want.")],
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Council YAML.", exists=True, readable=True),
    ],
    transcript: Annotated[
        Path | None,
        typer.Option("--transcript", "-t", help="Write JSON transcript to this path."),
    ] = None,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Suppress streaming phase output."),
    ] = False,
) -> None:
    """Run the council on a task."""
    try:
        cfg = load_council_config(config)
    except CouncilConfigError as e:
        _console.print(f"[red]config error:[/red] {e}")
        raise typer.Exit(code=2) from e

    council = Council(cfg)
    printer = RichPhasePrinter(quiet=quiet)
    result = asyncio.run(council.run(task, stream=printer))
    printer.final(result.final_answer)
    if transcript is not None:
        written = write_transcript(result, transcript)
        _console.print(f"[dim]transcript: {written}[/dim]")


@app.command()
def init(
    path: Annotated[Path, typer.Argument(help="Where to write the new council YAML.")] = Path(
        "council.yaml"
    ),
    template: Annotated[
        str,
        typer.Option(
            "--template",
            "-t",
            help=f"One of: {', '.join(AVAILABLE_TEMPLATES)}",
        ),
    ] = "workstation-4agent",
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Overwrite an existing file at the target path."),
    ] = False,
) -> None:
    """Scaffold a starter council config from a shipped template."""
    src = _resolve_template(template)
    if path.exists() and not force:
        _console.print(f"[red]{path} already exists; use --force to overwrite[/red]")
        raise typer.Exit(code=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, path)
    _console.print(f"[green]wrote {path} from template {template!r}[/green]")


@app.command()
def validate(
    config: Annotated[
        Path, typer.Argument(help="Council YAML to validate.", exists=True, readable=True)
    ],
) -> None:
    """Parse + validate a council config. Exits non-zero on any invariant violation."""
    try:
        cfg = load_council_config(config)
    except CouncilConfigError as e:
        _console.print(f"[red]invalid:[/red] {e}")
        raise typer.Exit(code=1) from e
    _console.print(f"[green]ok[/green] — {len(cfg.agents)} agents, council {cfg.name!r}")
    for agent in cfg.agents:
        _console.print(
            f"  [cyan]{agent.name}[/cyan] ({agent.role.value}) → {agent.model} "
            f"[dim](family={agent.family}, t={agent.temperature})[/dim]"
        )


def main() -> None:  # pragma: no cover — tiny wrapper
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
