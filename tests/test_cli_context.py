"""CLI tests for the contextual addressing flags (--recipient and friends)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import ClassVar

from workspec import cli
from workspec.context import ContextKey
from workspec.contract import ContractDelta, ContractElement
from workspec.draft import Draft
from workspec.models import Verdict
from workspec.store import ContextStore


def _ns(**kw: object) -> argparse.Namespace:
    base: dict[str, object] = {"channel": None, "project": None, "recipient": None}
    base.update(kw)
    return argparse.Namespace(**base)


# --- key / store helpers --------------------------------------------------- #


def test_context_key_none_without_flags() -> None:
    assert cli._context_key(_ns()) is None


def test_context_key_from_recipient() -> None:
    key = cli._context_key(_ns(recipient="alice"))
    assert key == ContextKey(recipient="alice")


def test_context_key_carries_dormant_axes() -> None:
    key = cli._context_key(_ns(channel="email", project="atlas", recipient="bob"))
    assert key == ContextKey(channel="email", project="atlas", recipient="bob")


def test_context_store_roots_at_profile_dir(tmp_path: Path) -> None:
    store = cli._context_store(_ns(profile_dir=str(tmp_path)))
    assert store.base_dir == tmp_path


# --- end-to-end dispatch with --recipient ---------------------------------- #


class _FakeDraftAgent:
    last_key: ClassVar[object] = "unset"
    last_contract_key: ClassVar[object] = "unset"
    last_init: ClassVar[dict] = {}

    def __init__(self, **kwargs: object) -> None:
        type(self).last_init = kwargs
        self.store = kwargs.get("store")

    def draft(self, spec, submission, instruction="", key=None):
        type(self).last_key = key
        return Draft(draft="hi", rationale="", open_questions=[], applied_traits=[])

    def learn_from_edit(self, draft, sent, feedback="", apply=True, applied_traits=None, key=None):
        type(self).last_key = key
        return []

    def learn_contract_from_edit(self, draft, sent, feedback="", apply=True, key=None):
        type(self).last_contract_key = key
        return []


class _FakeWorkSpecAgent:
    last_key: ClassVar[object] = "unset"

    def __init__(self, **kwargs: object) -> None:
        pass

    def check(self, spec, work, key=None):
        type(self).last_key = key
        return Verdict(passed=True, summary="ok", findings=[])


def test_draft_passes_recipient_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "DraftAgent", _FakeDraftAgent)
    sub = tmp_path / "msg.txt"
    sub.write_text("hello", encoding="utf-8")
    rc = cli.main(
        [
            "draft",
            str(sub),
            "--rubric",
            "email_reply",
            "--recipient",
            "alice",
            "--profile-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    assert _FakeDraftAgent.last_key == ContextKey(recipient="alice")
    # With a context key the agent is wired to a ContextStore, not the legacy store.
    assert _FakeDraftAgent.last_init["store"] is not None
    assert _FakeDraftAgent.last_init["profile_store"] is None


def test_draft_without_recipient_uses_legacy_store(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "DraftAgent", _FakeDraftAgent)
    sub = tmp_path / "msg.txt"
    sub.write_text("hello", encoding="utf-8")
    cli.main(["draft", str(sub), "--rubric", "email_reply", "--profile-dir", str(tmp_path)])
    assert _FakeDraftAgent.last_init["store"] is None
    assert _FakeDraftAgent.last_init["profile_store"] is not None


def test_learn_edit_passes_recipient_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "DraftAgent", _FakeDraftAgent)
    draft = tmp_path / "d.txt"
    sent = tmp_path / "s.txt"
    draft.write_text("draft", encoding="utf-8")
    sent.write_text("sent", encoding="utf-8")
    rc = cli.main(
        [
            "learn-from-edit",
            "--draft",
            str(draft),
            "--sent",
            str(sent),
            "--recipient",
            "bob",
            "--profile-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    assert _FakeDraftAgent.last_key == ContextKey(recipient="bob")


def test_check_passes_recipient_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "WorkSpecAgent", _FakeWorkSpecAgent)
    work = tmp_path / "w.md"
    work.write_text("Owned by Sam. Decision: ship.", encoding="utf-8")
    rc = cli.main(["check", str(work), "--rubric", "email_reply", "--recipient", "carol"])
    assert rc == 0
    assert _FakeWorkSpecAgent.last_key == ContextKey(recipient="carol")


def test_learn_edit_also_learns_contract(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "DraftAgent", _FakeDraftAgent)
    draft = tmp_path / "d.txt"
    sent = tmp_path / "s.txt"
    draft.write_text("draft", encoding="utf-8")
    sent.write_text("sent", encoding="utf-8")
    cli.main(
        [
            "learn-from-edit",
            "--draft",
            str(draft),
            "--sent",
            str(sent),
            "--recipient",
            "bob",
            "--profile-dir",
            str(tmp_path),
        ]
    )
    # Contract learning ran on the same edit, scoped to the recipient.
    assert _FakeDraftAgent.last_contract_key == ContextKey(recipient="bob")


# --- contract confirm / reject + profile --proposals ----------------------- #


def _proposal(rule: str) -> ContractElement:
    return ContractElement(kind="must_include", rule=rule, status="active", confirmed=False)


def _seed_proposal(tmp_path: Path, rule: str, recipient: str | None = None) -> str:
    store = ContextStore(tmp_path)
    key = ContextKey(recipient=recipient) if recipient else ContextKey()
    element = _proposal(rule)
    store.save_contract(key, ContractDelta(elements=[element]))
    return element.key


def test_contract_confirm_gates_element(tmp_path: Path) -> None:
    rule = "State a next step."
    element_key = _seed_proposal(tmp_path, rule, recipient="bob")
    rc = cli.main(
        ["contract", "confirm", element_key, "--recipient", "bob", "--profile-dir", str(tmp_path)]
    )
    assert rc == 0
    delta = ContextStore(tmp_path).load_contract(ContextKey(recipient="bob"))
    assert [e.rule for e in delta.gating_elements()] == [rule]


def test_contract_reject_retires_element(tmp_path: Path) -> None:
    rule = "State a next step."
    element_key = _seed_proposal(tmp_path, rule)
    rc = cli.main(["contract", "reject", element_key, "--profile-dir", str(tmp_path)])
    assert rc == 0
    delta = ContextStore(tmp_path).load_contract(ContextKey())
    assert delta.proposals() == []
    assert delta.gating_elements() == []


def test_contract_confirm_unknown_key_errors(tmp_path: Path) -> None:
    rc = cli.main(["contract", "confirm", "must_include:ghost", "--profile-dir", str(tmp_path)])
    assert rc == 2


def test_profile_proposals_lists_proposal(tmp_path: Path, capsys) -> None:
    rule = "State a next step."
    _seed_proposal(tmp_path, rule, recipient="bob")
    rc = cli.main(["profile", "--proposals", "--recipient", "bob", "--profile-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Proposals" in out
    assert rule in out


def test_profile_proposals_empty(tmp_path: Path, capsys) -> None:
    rc = cli.main(["profile", "--proposals", "--profile-dir", str(tmp_path)])
    assert rc == 0
    assert "No learned contract yet" in capsys.readouterr().out


def test_profile_proposals_shows_gating_section(tmp_path: Path, capsys) -> None:
    store = ContextStore(tmp_path)
    gating = ContractElement(
        kind="must_include", rule="Name an owner.", status="active", confirmed=True
    )
    store.save_contract(ContextKey(), ContractDelta(elements=[gating]))
    rc = cli.main(["profile", "--proposals", "--profile-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Gating now" in out
    assert "Name an owner." in out


def test_contract_confirm_malformed_file_errors(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    path = store.contract_path(ContextKey())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not valid json", encoding="utf-8")
    rc = cli.main(["contract", "confirm", "must_include:x", "--profile-dir", str(tmp_path)])
    assert rc == 2


class _ContractLearningAgent(_FakeDraftAgent):
    """A fake agent whose contract learning returns a real, reportable element."""

    def learn_contract_from_edit(self, draft, sent, feedback="", apply=True, key=None):
        type(self).last_contract_key = key
        return [
            ContractElement(
                kind="must_include", rule="State a next step.", status="active", confirmed=False
            )
        ]


def test_learn_edit_reports_contract_elements(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "DraftAgent", _ContractLearningAgent)
    draft = tmp_path / "d.txt"
    sent = tmp_path / "s.txt"
    draft.write_text("draft", encoding="utf-8")
    sent.write_text("sent", encoding="utf-8")
    rc = cli.main(
        [
            "learn-from-edit",
            "--draft",
            str(draft),
            "--sent",
            str(sent),
            "--recipient",
            "bob",
            "--profile-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "contract element(s)" in out
    assert "State a next step." in out
    assert "proposal" in out
