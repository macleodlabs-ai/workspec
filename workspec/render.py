"""Terminal rendering for WorkSpec verdicts, using rich.

Kept separate from the engine so the judgment logic has no presentation
concerns. The viral value of WorkSpec is a sharp, readable diagnosis — so this
module earns its keep.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from workspec.models import Severity, Spec, Verdict

_SEVERITY_STYLE = {
    Severity.BLOCKER: "bold red",
    Severity.WARNING: "yellow",
    Severity.NOTE: "dim cyan",
}
_SEVERITY_LABEL = {
    Severity.BLOCKER: "BLOCKER",
    Severity.WARNING: "WARNING",
    Severity.NOTE: "NOTE",
}


def render_verdict(
    verdict: Verdict,
    spec: Spec,
    source_name: str,
    console: Console | None = None,
) -> None:
    """Print a full verdict report to the terminal."""
    console = console or Console()

    if verdict.passed:
        header = Text(" ✅  MANAGER-READY ", style="bold white on green")
    else:
        header = Text(" ❌  NOT MANAGER-READY ", style="bold white on red")

    console.print()
    console.print(
        Panel(
            header,
            title=f"WorkSpec · {spec.title}",
            subtitle=f"checked: {source_name}",
            border_style="green" if verdict.passed else "red",
        )
    )

    console.print(Text(verdict.summary, style="italic"))
    console.print()

    if verdict.findings:
        table = Table(
            show_header=True,
            header_style="bold",
            expand=True,
            title="Findings",
            title_style="bold",
        )
        table.add_column("Severity", width=10, no_wrap=True)
        table.add_column("Problem", ratio=3)
        table.add_column("Suggested fix", ratio=3)

        # Order: blockers first, then warnings, then notes.
        ordered = verdict.blockers + verdict.warnings + verdict.notes
        for f in ordered:
            sev = Text(_SEVERITY_LABEL[f.severity], style=_SEVERITY_STYLE[f.severity])
            problem = Text(f.problem)
            if f.evidence:
                problem.append(f"\n↳ {f.evidence}", style="dim italic")
            table.add_row(sev, problem, Text(f.suggested_fix))
        console.print(table)
    else:
        console.print(Text("No issues found.", style="green"))

    # Tally line.
    console.print()
    tally = Text()
    tally.append(f"{len(verdict.blockers)} blocker(s)  ", style="bold red")
    tally.append(f"{len(verdict.warnings)} warning(s)  ", style="yellow")
    tally.append(f"{len(verdict.notes)} note(s)", style="dim cyan")
    console.print(tally)

    if verdict.rewrite_prompt:
        console.print()
        console.print(
            Panel(
                verdict.rewrite_prompt,
                title="📋  Rewrite prompt (paste into your AI assistant)",
                border_style="cyan",
            )
        )
    console.print()
