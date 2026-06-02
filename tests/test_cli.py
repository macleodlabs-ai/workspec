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
from workspec.profile import LearnMetric, ProfileStore, VoiceProfile, VoiceTrait

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
    assert cli._resolve_model(_args(model="m-flag"), "anthropic") == "m-flag"
    # env next
    monkeypatch.setenv("WORKSPEC_MODEL", "m-env")
    assert cli._resolve_model(_args(), "anthropic") == "m-env"


def test_resolve_model_defaults_per_provider() -> None:
    assert cli._resolve_model(_args(), "anthropic") == cli.DEFAULT_MODEL
    assert cli._resolve_model(_args(), "openai") == cli.DEFAULT_OPENAI_MODEL


# --- command dispatch ------------------------------------------------------ #


class FakeWorkSpecAgent:
    last_kwargs: ClassVar[dict] = {}
    verdict: ClassVar[Verdict] = Verdict(passed=True, summary="ok", findings=[])
    raise_on_check: ClassVar[bool] = False
    raise_on_init: ClassVar[bool] = False

    def __init__(self, **kwargs):
        if type(self).raise_on_init:
            raise RuntimeError("no api key")
        type(self).last_kwargs = kwargs

    def check(self, spec, work):
        if type(self).raise_on_check:
            raise RuntimeError("model exploded")
        return type(self).verdict


class FakeDraftAgent:
    learned: ClassVar[list] = []
    applied: ClassVar[list] = []  # what draft() reports as applied_traits
    last_applied: ClassVar[list] = []  # what learn_from_edit() received
    raise_on_draft: ClassVar[bool] = False
    raise_on_init: ClassVar[bool] = False
    raise_on_learn: ClassVar[bool] = False

    def __init__(self, **kwargs):
        if type(self).raise_on_init:
            raise RuntimeError("no api key")

    def draft(self, spec, submission, instruction=""):
        if type(self).raise_on_draft:
            raise RuntimeError("model exploded")
        return Draft(
            draft="Hi — confirming.",
            rationale="brief",
            open_questions=["[CONFIRM: x]"],
            applied_traits=list(type(self).applied),
        )

    def learn_from_edit(self, draft, sent, feedback="", apply=True, applied_traits=None):
        if type(self).raise_on_learn:
            raise RuntimeError("model exploded")
        type(self).last_applied = list(applied_traits or [])
        return list(type(self).learned)


def test_rubrics_command_returns_zero(capsys) -> None:
    assert cli.main(["rubrics"]) == 0
    out = capsys.readouterr().out
    assert "email_reply" in out


def test_rubrics_command_empty(tmp_path: Path, monkeypatch, capsys) -> None:
    # Point the loader at a directory with no rubric files.
    monkeypatch.setattr("workspec.spec_loader._RUBRIC_DIR", tmp_path / "empty")
    assert cli.main(["rubrics"]) == 0
    assert "No built-in rubrics" in capsys.readouterr().out


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


def test_check_model_failure_returns_two(tmp_path: Path, monkeypatch) -> None:
    FakeWorkSpecAgent.raise_on_check = True
    monkeypatch.setattr(cli, "WorkSpecAgent", FakeWorkSpecAgent)
    work = tmp_path / "w.md"
    work.write_text("x", encoding="utf-8")
    try:
        assert cli.main(["check", str(work), "--rubric", "email_reply"]) == 2
    finally:
        FakeWorkSpecAgent.raise_on_check = False


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


def test_draft_defaults_to_email_reply(tmp_path: Path, monkeypatch, capsys) -> None:
    # No --rubric / --spec: the draft subparser defaults --rubric to email_reply,
    # so `workspec draft <file>` resolves to the built-in reply contract and succeeds.
    monkeypatch.setattr(cli, "DraftAgent", FakeDraftAgent)
    sub = tmp_path / "incoming.txt"
    sub.write_text("Can you confirm Friday?", encoding="utf-8")
    rc = cli.main(["draft", str(sub), "--profile-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "confirming" in out.lower()


def test_draft_missing_submission_returns_two() -> None:
    assert cli.main(["draft", "/no/such.txt", "--rubric", "email_reply"]) == 2


def test_draft_bad_contract_returns_two(tmp_path: Path) -> None:
    sub = tmp_path / "msg.txt"
    sub.write_text("hi", encoding="utf-8")
    assert cli.main(["draft", str(sub), "--spec", "/no/such/contract.yaml"]) == 2


def test_draft_agent_construction_failure_returns_two(tmp_path: Path, monkeypatch) -> None:
    FakeDraftAgent.raise_on_init = True
    monkeypatch.setattr(cli, "DraftAgent", FakeDraftAgent)
    sub = tmp_path / "msg.txt"
    sub.write_text("hi", encoding="utf-8")
    try:
        assert cli.main(["draft", str(sub), "--rubric", "email_reply"]) == 2
    finally:
        FakeDraftAgent.raise_on_init = False


def test_draft_json_output(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "DraftAgent", FakeDraftAgent)
    sub = tmp_path / "msg.txt"
    sub.write_text("Confirm Friday?", encoding="utf-8")
    rc = cli.main(["draft", str(sub), "--rubric", "email_reply", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["draft"]


def test_check_agent_construction_failure_returns_two(tmp_path: Path, monkeypatch) -> None:
    FakeWorkSpecAgent.raise_on_init = True
    monkeypatch.setattr(cli, "WorkSpecAgent", FakeWorkSpecAgent)
    work = tmp_path / "w.md"
    work.write_text("x", encoding="utf-8")
    try:
        assert cli.main(["check", str(work), "--rubric", "email_reply"]) == 2
    finally:
        FakeWorkSpecAgent.raise_on_init = False


def test_check_bad_contract_returns_two(tmp_path: Path) -> None:
    work = tmp_path / "w.md"
    work.write_text("x", encoding="utf-8")
    assert cli.main(["check", str(work), "--spec", "/no/such/contract.yaml"]) == 2


def test_draft_model_failure_returns_two(tmp_path: Path, monkeypatch) -> None:
    FakeDraftAgent.raise_on_draft = True
    monkeypatch.setattr(cli, "DraftAgent", FakeDraftAgent)
    sub = tmp_path / "msg.txt"
    sub.write_text("hi", encoding="utf-8")
    try:
        assert cli.main(["draft", str(sub), "--rubric", "email_reply"]) == 2
    finally:
        FakeDraftAgent.raise_on_draft = False


# --- learn-from-edit command ----------------------------------------------- #


def _edit_files(tmp_path: Path) -> tuple[str, str]:
    draft = tmp_path / "draft.txt"
    sent = tmp_path / "sent.txt"
    draft.write_text("Dear Sir, please find attached.", encoding="utf-8")
    sent.write_text("Hey! Attached.", encoding="utf-8")
    return str(draft), str(sent)


def test_learn_edit_reports_learned_traits(tmp_path: Path, monkeypatch, capsys) -> None:
    FakeDraftAgent.learned = [VoiceTrait(category="tone", rule="Be casual.", provenance="edit")]
    monkeypatch.setattr(cli, "DraftAgent", FakeDraftAgent)
    draft, sent = _edit_files(tmp_path)
    try:
        rc = cli.main(
            ["learn-from-edit", "--draft", draft, "--sent", sent, "--profile-dir", str(tmp_path)]
        )
    finally:
        FakeDraftAgent.learned = []
    out = capsys.readouterr().out
    assert rc == 0
    assert "learned 1 voice trait" in out
    assert "Be casual." in out


def test_learn_edit_dry_run(tmp_path: Path, monkeypatch, capsys) -> None:
    FakeDraftAgent.learned = [VoiceTrait(category="tone", rule="Be casual.", provenance="edit")]
    monkeypatch.setattr(cli, "DraftAgent", FakeDraftAgent)
    draft, sent = _edit_files(tmp_path)
    try:
        rc = cli.main(["learn-from-edit", "--draft", draft, "--sent", sent, "--dry-run"])
    finally:
        FakeDraftAgent.learned = []
    assert rc == 0
    assert "would learn" in capsys.readouterr().out


def test_learn_edit_no_traits(tmp_path: Path, monkeypatch, capsys) -> None:
    FakeDraftAgent.learned = []
    monkeypatch.setattr(cli, "DraftAgent", FakeDraftAgent)
    draft, sent = _edit_files(tmp_path)
    rc = cli.main(["learn-from-edit", "--draft", draft, "--sent", sent])
    assert rc == 0
    assert "No generalizable voice traits" in capsys.readouterr().out


def test_learn_edit_missing_file_returns_two(tmp_path: Path) -> None:
    sent = tmp_path / "sent.txt"
    sent.write_text("x", encoding="utf-8")
    assert cli.main(["learn-from-edit", "--draft", "/no/draft.txt", "--sent", str(sent)]) == 2


def test_learn_edit_agent_construction_failure_returns_two(tmp_path: Path, monkeypatch) -> None:
    FakeDraftAgent.raise_on_init = True
    monkeypatch.setattr(cli, "DraftAgent", FakeDraftAgent)
    draft, sent = _edit_files(tmp_path)
    try:
        assert cli.main(["learn-from-edit", "--draft", draft, "--sent", sent]) == 2
    finally:
        FakeDraftAgent.raise_on_init = False


def test_learn_edit_failure_returns_two(tmp_path: Path, monkeypatch) -> None:
    FakeDraftAgent.raise_on_learn = True
    monkeypatch.setattr(cli, "DraftAgent", FakeDraftAgent)
    draft, sent = _edit_files(tmp_path)
    try:
        assert cli.main(["learn-from-edit", "--draft", draft, "--sent", sent]) == 2
    finally:
        FakeDraftAgent.raise_on_learn = False


# --- profile command ------------------------------------------------------- #


def test_profile_empty(tmp_path: Path, capsys) -> None:
    rc = cli.main(["profile", "--profile-dir", str(tmp_path)])
    assert rc == 0
    assert "No voice profile yet" in capsys.readouterr().out


def test_profile_lists_traits(tmp_path: Path, capsys) -> None:
    store = ProfileStore(tmp_path)
    profile = VoiceProfile(owner="sam")
    profile.reinforce_or_add(category="tone", rule="Be warm.", provenance="edit")
    store.save(profile)

    rc = cli.main(["profile", "--profile-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Voice profile" in out
    assert "Be warm." in out


def test_profile_reset_existing(tmp_path: Path, capsys) -> None:
    store = ProfileStore(tmp_path)
    store.save(VoiceProfile())
    assert store.exists()

    rc = cli.main(["profile", "--reset", "--profile-dir", str(tmp_path)])
    assert rc == 0
    assert "deleted" in capsys.readouterr().out.lower()
    assert not store.exists()


def test_profile_reset_when_absent(tmp_path: Path, capsys) -> None:
    rc = cli.main(["profile", "--reset", "--profile-dir", str(tmp_path)])
    assert rc == 0
    assert "No profile to delete" in capsys.readouterr().out


# --- profile --stats (eval surface) ---------------------------------------- #


def test_profile_stats_empty(tmp_path: Path, capsys) -> None:
    rc = cli.main(["profile", "--stats", "--profile-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Voice profile stats" in out
    assert "0 trait" in out
    assert "No active traits" in out
    assert "No edit-ratio metrics" in out


def test_profile_stats_reports_counts_top_traits_and_trend(tmp_path: Path, capsys) -> None:
    store = ProfileStore(tmp_path)
    profile = VoiceProfile(owner="sam")
    profile.traits = [
        VoiceTrait(
            category="tone",
            rule="Lead with the answer.",
            weight=0.9,
            status="active",
            observations=4,
        ),
        VoiceTrait(category="length", rule="Keep it short.", weight=0.6, status="active"),
        VoiceTrait(category="phrasing", rule="Try this once.", status="provisional"),
        VoiceTrait(category="do_not", rule="No filler.", weight=0.1, status="retired"),
    ]
    # An improving trend: older drafts heavily edited, recent ones nearly untouched.
    profile.metrics = [LearnMetric(edit_ratio=r) for r in (0.40, 0.45, 0.90, 0.95)]
    store.save(profile)

    rc = cli.main(["profile", "--stats", "--profile-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "4 trait" in out
    assert "2 active" in out and "1 provisional" in out and "1 retired" in out
    # strongest active trait surfaces and outranks the weaker one
    assert "Lead with the answer." in out
    assert out.index("Lead with the answer.") < out.index("Keep it short.")
    # the recurring/retired traits do not appear in the top-active block
    assert "Try this once." not in out
    # edit-ratio trend is shown
    assert "Edit-ratio trend" in out


def test_profile_stats_shows_trend_delta(tmp_path: Path, capsys) -> None:
    """With more than `recent` metrics, the recent-vs-earlier delta is reported."""
    store = ProfileStore(tmp_path)
    profile = VoiceProfile(owner="sam")
    # 14 metrics: the older 4 are heavily edited, the recent 10 nearly untouched,
    # so the trend delta is positive (drafts need less editing now).
    older = [0.3, 0.3, 0.3, 0.3]
    recent = [0.95] * 10
    profile.metrics = [LearnMetric(edit_ratio=r) for r in older + recent]
    store.save(profile)

    rc = cli.main(["profile", "--stats", "--profile-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "vs earlier" in out
    assert "+0." in out  # positive delta rendered with a sign


def test_profile_stats_trend_declining(tmp_path: Path, capsys) -> None:
    """A worsening trend (drafts need more editing) renders the down arrow."""
    store = ProfileStore(tmp_path)
    profile = VoiceProfile(owner="sam")
    profile.metrics = [LearnMetric(edit_ratio=r) for r in [0.95, 0.95, 0.95, 0.95] + [0.3] * 10]
    store.save(profile)
    assert cli.main(["profile", "--stats", "--profile-dir", str(tmp_path)]) == 0
    assert "↓" in capsys.readouterr().out


def test_profile_stats_trend_flat(tmp_path: Path, capsys) -> None:
    """No change between recent and earlier means renders 'no change', not an arrow."""
    store = ProfileStore(tmp_path)
    profile = VoiceProfile(owner="sam")
    profile.metrics = [LearnMetric(edit_ratio=0.7) for _ in range(14)]
    store.save(profile)
    assert cli.main(["profile", "--stats", "--profile-dir", str(tmp_path)]) == 0
    assert "no change vs earlier" in capsys.readouterr().out


def test_profile_corrupt_returns_two(tmp_path: Path, capsys) -> None:
    """A hand-corrupted profile yields a clean exit 2, not an uncaught traceback."""
    store = ProfileStore(tmp_path)
    store.dir.mkdir(parents=True, exist_ok=True)
    store.path.write_text("{ this is not valid json", encoding="utf-8")
    rc = cli.main(["profile", "--profile-dir", str(tmp_path)])
    assert rc == 2
    assert "voice profile" in capsys.readouterr().err.lower()


def test_draft_writes_applied_traits_sidecar(tmp_path: Path, monkeypatch, capsys) -> None:
    """`draft` writes the active trait keys to a sidecar for the negative-signal loop."""
    FakeDraftAgent.applied = ["tone:Be warm", "signoff:Cheers"]
    monkeypatch.setattr(cli, "DraftAgent", FakeDraftAgent)
    sub = tmp_path / "msg.txt"
    sub.write_text("confirm friday?", encoding="utf-8")
    try:
        rc = cli.main(
            ["draft", str(sub), "--rubric", "email_reply", "--profile-dir", str(tmp_path)]
        )
    finally:
        FakeDraftAgent.applied = []
    sidecar = sub.with_suffix(".txt.traits")
    assert rc == 0
    assert sidecar.read_text(encoding="utf-8").splitlines() == ["tone:Be warm", "signoff:Cheers"]
    assert "applied traits written to" in capsys.readouterr().out


def test_learn_edit_applied_traits_from_file_and_literal(tmp_path: Path, monkeypatch) -> None:
    """`learn-from-edit --applied-traits` expands a sidecar file and literal keys."""
    monkeypatch.setattr(cli, "DraftAgent", FakeDraftAgent)
    sidecar = tmp_path / "msg.txt.traits"
    sidecar.write_text("tone:Be warm\n\nsignoff:Cheers\n", encoding="utf-8")  # blank line dropped
    draft, sent = tmp_path / "d.txt", tmp_path / "s.txt"
    draft.write_text("Dear Sir, attached.", encoding="utf-8")
    sent.write_text("Hi! attached.", encoding="utf-8")
    FakeDraftAgent.last_applied = []
    rc = cli.main(
        [
            "learn-from-edit",
            "--draft",
            str(draft),
            "--sent",
            str(sent),
            "--applied-traits",
            str(sidecar),
            "length:Be brief",
            "--profile-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    assert FakeDraftAgent.last_applied == ["tone:Be warm", "signoff:Cheers", "length:Be brief"]
