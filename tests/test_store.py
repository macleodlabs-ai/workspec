"""Tests for ContextStore: per-scope addressing and legacy migration."""

from __future__ import annotations

from pathlib import Path

import pytest

from workspec.context import ContextKey
from workspec.contract import ContractDelta, ContractElement
from workspec.profile import (
    PROFILE_FILENAME,
    ProfileLoadError,
    ProfileStore,
    VoiceProfile,
    VoiceTrait,
)
from workspec.store import ContextStore


def _profile_with_trait(rule: str) -> VoiceProfile:
    return VoiceProfile(owner="me", traits=[VoiceTrait(category="tone", rule=rule)])


def test_voice_path_addresses_per_scope(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    assert store.voice_path(ContextKey()) == tmp_path / "voice" / "global.json"
    assert (
        store.voice_path(ContextKey(recipient="alice"))
        == tmp_path / "voice" / "recipient=alice.json"
    )


def test_reserved_dirs_exposed_but_unused(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    assert store.contract_dir == tmp_path / "contract"
    assert store.capability_dir == tmp_path / "capability"
    # Reserved dirs are not created merely by addressing them.
    assert not store.contract_dir.exists()
    assert not store.capability_dir.exists()


def test_save_and_load_round_trip_per_scope(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    store.save_voice(ContextKey(recipient="alice"), _profile_with_trait("be warm"))
    store.save_voice(ContextKey(), _profile_with_trait("be brief"))

    alice = store.load_voice(ContextKey(recipient="alice"))
    glob = store.load_voice(ContextKey())
    assert [t.rule for t in alice.traits] == ["be warm"]
    assert [t.rule for t in glob.traits] == ["be brief"]


def test_missing_scope_loads_empty(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    assert store.load_voice(ContextKey(recipient="ghost")).traits == []


def test_recipient_learning_does_not_leak_to_other_scopes(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    store.save_voice(ContextKey(recipient="alice"), _profile_with_trait("alice-only"))
    assert store.load_voice(ContextKey(recipient="bob")).traits == []
    assert store.load_voice(ContextKey()).traits == []


def test_legacy_migration_to_global(tmp_path: Path) -> None:
    # Seed a legacy single-file profile at the base-dir root.
    legacy = ProfileStore(tmp_path)
    legacy.save(_profile_with_trait("legacy voice"))
    assert (tmp_path / PROFILE_FILENAME).exists()

    store = ContextStore(tmp_path)
    glob = store.load_voice(ContextKey())

    # Migrated losslessly into the global scope; legacy file relocated.
    assert [t.rule for t in glob.traits] == ["legacy voice"]
    assert store.voice_path(ContextKey()).exists()
    assert not (tmp_path / PROFILE_FILENAME).exists()


def test_migration_runs_once_and_skips_when_global_exists(tmp_path: Path) -> None:
    # A global scope file already exists; a stray legacy file must NOT overwrite it.
    store = ContextStore(tmp_path)
    store.save_voice(ContextKey(), _profile_with_trait("real global"))
    ProfileStore(tmp_path).save(_profile_with_trait("stale legacy"))

    fresh = ContextStore(tmp_path)
    glob = fresh.load_voice(ContextKey())
    assert [t.rule for t in glob.traits] == ["real global"]
    # The legacy file is left untouched (not migrated over the existing global).
    assert (tmp_path / PROFILE_FILENAME).exists()


def test_no_migration_when_no_legacy(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    assert store.load_voice(ContextKey()).traits == []
    assert not (tmp_path / PROFILE_FILENAME).exists()


# --- contract persistence -------------------------------------------------- #


def _delta_with(rule: str, kind: str = "must_include") -> ContractDelta:
    return ContractDelta(
        owner="me",
        elements=[ContractElement(kind=kind, rule=rule)],  # type: ignore[list-item]
    )


def test_contract_path_addresses_per_scope(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    assert store.contract_path(ContextKey()) == tmp_path / "contract" / "global.json"
    assert (
        store.contract_path(ContextKey(recipient="alice"))
        == tmp_path / "contract" / "recipient=alice.json"
    )


def test_contract_round_trip_per_scope(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    store.save_contract(ContextKey(recipient="alice"), _delta_with("Name an owner."))
    store.save_contract(ContextKey(), _delta_with("State a decision."))

    alice = store.load_contract(ContextKey(recipient="alice"))
    glob = store.load_contract(ContextKey())
    assert [e.rule for e in alice.elements] == ["Name an owner."]
    assert [e.rule for e in glob.elements] == ["State a decision."]


def test_missing_contract_scope_loads_empty(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    assert store.load_contract(ContextKey(recipient="ghost")).elements == []


def test_contract_does_not_leak_across_scopes(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    store.save_contract(ContextKey(recipient="alice"), _delta_with("alice-only"))
    assert store.load_contract(ContextKey(recipient="bob")).elements == []
    assert store.load_contract(ContextKey()).elements == []


def test_malformed_contract_raises_clear_error(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    path = store.contract_path(ContextKey())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not valid json", encoding="utf-8")
    with pytest.raises(ProfileLoadError, match="contract delta"):
        store.load_contract(ContextKey())


def test_atomic_write_cleans_up_temp_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed atomic write removes its temp file and re-raises (no orphans)."""
    import workspec.store as store_module

    store = ContextStore(tmp_path)
    key = ContextKey(recipient="alice")

    def _boom(src: object, dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(store_module.os, "replace", _boom)
    with pytest.raises(OSError, match="disk full"):
        store.save_contract(key, _delta_with("Name an owner."))
    # The real file was never created and no .json.tmp orphan was left behind.
    assert not store.contract_path(key).exists()
    assert list(store.contract_dir.glob("*.tmp")) == []
