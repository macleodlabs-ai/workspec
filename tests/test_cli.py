"""Unit tests for the CLI: resolution helpers and command dispatch.

The agents are monkeypatched so commands run end-to-end with no network.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import ClassVar

import pytest

from workspec import cli
from workspec.draft import Draft
from workspec.models import Finding, Severity, Verdict

# --- resolution helpers ---------------------------------------------------- #


def _args(provider=None, model=None) -> argparse.Namespace:
    return argparse.Namespace(provider=provider, model=model)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("WORKSPEC_MODEL", raising=False)
    monkeypatch.delenv("WORKSPEC_PROVIDER", raising=False)


def test_resolve_provider_default() -> None:
    assert cli._resolve_provider(_args()) == "anthropic"


def test_resolve_provider_flag_beats_env(monkeypatch) -> None:
    monkeypatch.setenv("WORKSPEC_PROVIDER", "openai")
    assert cli._resolve_provider(_args(provider="anthropic")) == "anthropic"


def test_resolve_provider_env(monkeypatch) -> None:
    monkeypatch.setenv("WORKSPEC_PROVIDER", "openai")
    assert cli._resolve_provider(_args()) == "openai"


def test_resolve_provider_invalid_env_raises(monkeypatch) -> None:
    monkeypatch.setenv("WORKSPEC_PROVIDER", "bogus")
    with pytest.raises(ValueError, match="Unknown provider"):
        cli._resolve_provider(_args())


def test_resolve_model_precedence(monkeypatch) -> None:
    # flag wins
    assert cli._resolve_model(_args(model="m-flag")) == "m-flag"
    # env next
    monkeypatch.setenv("WORKSPEC_MODEL", "m-env")
    assert cli._resolve_model(_args()) == "m-env"


def test_resolve_model_defaults_per_provider() -> None:
    assert cli._resolve_model(_args(provider="anthropic")) == cli.DEFAULT_MODEL
    assert cli._resolve_model(_args(provider="openai")) == cli.DEFAULT_OPENAI_MODEL


# --- command dispatch ------------------------------------------------------ #


class FakeWorkSpecAgent:
    last_kwargs: ClassVar[dict] = {}
    verdict: ClassVar[Verdict] = Verdict(passed=True, summary="ok", findings=[])

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs

    def check(self, spec, work):
        return type(self).verdict


class FakeDraftAgent:
    def __init__(self, **kwargs):
        pass

    def draft(self, spec, submission, instruction=""):
        return Draft(draft="Hi — confirming.", rationale="brief", open_questions=["[CONFIRM: x]"])


def test_rubrics_command_returns_zero(capsys) -> None:
    assert cli.main(["rubrics"]) == 0
    out = capsys.readouterr().out
    assert "email_reply" in out


def test_check_command_pass_returns_zero(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "WorkSpecAgent", FakeWorkSpecAgent)
    work = tmp_path / "work.md"
    work.write_text("Owned by Sam. Decision: ship.", encoding="utf-8")

    rc = cli.main(["check", str(work), "--rubric", "email_reply", "--model", "claude-haiku-4-5"])

    assert rc == 0
    assert FakeWorkSpecAgent.last_kwargs["model"] == "claude-haiku-4-5"
    assert FakeWorkSpecAgent.last_kwargs["provider"] == "anthropic"


def test_check_command_fail_returns_one(tmp_path: Path, monkeypatch) -> None:
    FakeWorkSpecAgent.verdict = Verdict(
        passed=False,
        summary="bad",
        findings=[
            Finding(
                severity=Severity.BLOCKER, rule="r", problem="p", evidence="", suggested_fix="f"
            )
        ],
    )
    monkeypatch.setattr(cli, "WorkSpecAgent", FakeWorkSpecAgent)
    work = tmp_path / "w.md"
    work.write_text("weak", encoding="utf-8")
    try:
        assert cli.main(["check", str(work), "--rubric", "email_reply"]) == 1
    finally:
        FakeWorkSpecAgent.verdict = Verdict(passed=True, summary="ok", findings=[])


def test_check_command_json_output(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "WorkSpecAgent", FakeWorkSpecAgent)
    work = tmp_path / "w.md"
    work.write_text("x", encoding="utf-8")
    cli.main(["check", str(work), "--rubric", "email_reply", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    assert "summary" in payload


def test_check_missing_work_file_returns_two() -> None:
    assert cli.main(["check", "/no/such/file.md", "--rubric", "email_reply"]) == 2


def test_check_missing_contract_returns_two(tmp_path: Path) -> None:
    work = tmp_path / "w.md"
    work.write_text("x", encoding="utf-8")
    assert cli.main(["check", str(work)]) == 2


def test_check_invalid_env_provider_returns_two(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WORKSPEC_PROVIDER", "bogus")
    monkeypatch.setattr(cli, "WorkSpecAgent", FakeWorkSpecAgent)
    work = tmp_path / "w.md"
    work.write_text("x", encoding="utf-8")
    assert cli.main(["check", str(work), "--rubric", "email_reply"]) == 2


def test_draft_command_outputs_draft(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "DraftAgent", FakeDraftAgent)
    sub = tmp_path / "msg.txt"
    sub.write_text("Can you confirm Friday?", encoding="utf-8")
    rc = cli.main(["draft", str(sub), "--rubric", "email_reply", "--profile-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "confirming" in out.lower()
    assert "CONFIRM" in out  # open question surfaced


def test_draft_missing_submission_returns_two() -> None:
    assert cli.main(["draft", "/no/such.txt", "--rubric", "email_reply"]) == 2
