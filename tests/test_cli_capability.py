"""CLI tests for `workspec capability set | show`."""

from __future__ import annotations

from pathlib import Path

from workspec import cli
from workspec.context import DEFAULT_CAPABILITY, ContextKey
from workspec.store import ContextStore


def test_capability_set_persists(tmp_path: Path) -> None:
    rc = cli.main(["capability", "set", "alice", "proven", "--profile-dir", str(tmp_path)])
    assert rc == 0
    cap = ContextStore(tmp_path).load_capability(ContextKey(recipient="alice"))
    assert cap.bucket == "proven"


def test_capability_set_requires_explicit_bucket(tmp_path: Path) -> None:
    # The dial is never inferred: omitting the bucket on `set` must error (exit 2)
    # rather than silently rating the recipient a default the owner never typed.
    rc = cli.main(["capability", "set", "alice", "--profile-dir", str(tmp_path)])
    assert rc == 2
    # Nothing was persisted.
    assert not ContextStore(tmp_path).capability_path(ContextKey(recipient="alice")).exists()


def test_capability_show_rated(tmp_path: Path, capsys) -> None:
    cli.main(["capability", "set", "alice", "new", "--profile-dir", str(tmp_path)])
    capsys.readouterr()
    rc = cli.main(["capability", "show", "alice", "--profile-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "alice" in out
    assert "new" in out
    # show reports both knobs.
    assert "check floor" in out
    assert "draft directive" in out


def test_capability_show_unrated_marks_default(tmp_path: Path, capsys) -> None:
    rc = cli.main(["capability", "show", "ghost", "--profile-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert DEFAULT_CAPABILITY in out
    assert "not yet rated" in out


def test_capability_show_malformed_errors(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    path = store.capability_path(ContextKey(recipient="alice"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not valid json", encoding="utf-8")
    rc = cli.main(["capability", "show", "alice", "--profile-dir", str(tmp_path)])
    assert rc == 2


def test_capability_set_then_show_proven_knobs(tmp_path: Path, capsys) -> None:
    cli.main(["capability", "set", "vet", "proven", "--profile-dir", str(tmp_path)])
    capsys.readouterr()
    cli.main(["capability", "show", "vet", "--profile-dir", str(tmp_path)])
    out = capsys.readouterr().out
    # proven floors minor gaps at note and asks for terse drafts.
    assert "note" in out
    assert "terse" in out
