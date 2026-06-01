"""Unit tests for terminal rendering (captured, no real terminal needed)."""

from __future__ import annotations

from rich.console import Console

from workspec.models import Finding, Severity, Spec, Verdict
from workspec.render import render_verdict


def _spec() -> Spec:
    return Spec(type="memo", title="Decision Memo")


def _render(verdict: Verdict) -> str:
    console = Console(record=True, width=100)
    render_verdict(verdict, _spec(), "report.md", console=console)
    return console.export_text()


def test_render_passed_verdict() -> None:
    out = _render(Verdict(passed=True, summary="All good.", findings=[]))
    assert "MANAGER-READY" in out
    assert "All good." in out
    assert "No issues found." in out
    assert "Decision Memo" in out
    assert "report.md" in out


def test_render_failed_verdict_with_findings_and_rewrite() -> None:
    verdict = Verdict(
        passed=False,
        summary="Missing owner.",
        findings=[
            Finding(
                severity=Severity.BLOCKER,
                rule="must_include",
                problem="No named owner.",
                evidence="the whole doc",
                suggested_fix="Name an owner.",
            ),
            Finding(
                severity=Severity.WARNING,
                rule="general",
                problem="Vague timeline.",
                evidence="",
                suggested_fix="Give a date.",
            ),
        ],
        rewrite_prompt="Rewrite with an owner and a date.",
    )
    out = _render(verdict)
    assert "NOT MANAGER-READY" in out
    assert "BLOCKER" in out
    assert "WARNING" in out
    assert "No named owner." in out
    assert "1 blocker(s)" in out
    assert "Rewrite prompt" in out


def test_render_without_console_argument() -> None:
    # Should not raise when constructing its own Console.
    render_verdict(Verdict(passed=True, summary="ok"), _spec(), "x.md")
