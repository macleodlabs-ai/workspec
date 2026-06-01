"""Unit tests for the typed data models (no network)."""

from __future__ import annotations

from workspec.models import AIPolicy, Finding, Severity, Spec, Verdict


def test_severity_values() -> None:
    assert Severity.BLOCKER.value == "blocker"
    assert Severity.WARNING.value == "warning"
    assert Severity.NOTE.value == "note"


def test_aipolicy_defaults() -> None:
    policy = AIPolicy()
    assert policy.ai_allowed is True
    assert policy.requires_source_check is True
    assert policy.human_accountable is True


def test_spec_defaults_and_factory() -> None:
    spec = Spec(type="memo", title="Memo")
    assert spec.description == ""
    assert spec.must_include == []
    assert spec.must_not_include == []
    assert spec.acceptance_tests == []
    assert isinstance(spec.ai_policy, AIPolicy)
    # default_factory must not share the same list across instances
    spec.must_include.append("owner")
    assert Spec(type="x", title="y").must_include == []


def test_spec_render_for_prompt_with_items() -> None:
    spec = Spec(
        type="decision_memo",
        title="Decision Memo",
        description="A memo asking for a decision.",
        must_include=["a named decision", "options"],
        must_not_include=["hedging"],
        acceptance_tests=["Each claim cites a source."],
    )
    out = spec.render_for_prompt()
    assert "SPEC TYPE: decision_memo" in out
    assert "TITLE: Decision Memo" in out
    assert "- a named decision" in out
    assert "- hedging" in out
    assert "- Each claim cites a source." in out
    assert "AI allowed: True" in out


def test_spec_render_for_prompt_empty_sections_say_none() -> None:
    out = Spec(type="t", title="T").render_for_prompt()
    assert "(none)" in out
    assert "DESCRIPTION: (none)" in out


def _finding(sev: Severity) -> Finding:
    return Finding(severity=sev, rule="r", problem="p", evidence="", suggested_fix="f")


def test_verdict_severity_partitions() -> None:
    verdict = Verdict(
        passed=False,
        summary="nope",
        findings=[
            _finding(Severity.BLOCKER),
            _finding(Severity.WARNING),
            _finding(Severity.WARNING),
            _finding(Severity.NOTE),
        ],
    )
    assert len(verdict.blockers) == 1
    assert len(verdict.warnings) == 2
    assert len(verdict.notes) == 1


def test_verdict_defaults() -> None:
    verdict = Verdict(passed=True, summary="ok")
    assert verdict.findings == []
    assert verdict.rewrite_prompt is None


def test_verdict_json_roundtrip() -> None:
    verdict = Verdict(passed=False, summary="s", findings=[_finding(Severity.BLOCKER)])
    restored = Verdict.model_validate_json(verdict.model_dump_json())
    assert restored == verdict
