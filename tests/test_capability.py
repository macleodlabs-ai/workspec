"""Tests for the manual capability dial: the value type and its two knob maps."""

from __future__ import annotations

from pathlib import Path

import pytest

from workspec.capability import (
    Capability,
    scaffolding_directive,
    severity_floor,
)
from workspec.context import DEFAULT_CAPABILITY, ContextKey, SLStyle
from workspec.models import Severity
from workspec.profile import ProfileLoadError
from workspec.store import ContextStore

# --- value type ----------------------------------------------------------- #


def test_default_bucket_is_developing() -> None:
    assert Capability().bucket == DEFAULT_CAPABILITY == "developing"


def test_capability_roundtrips_through_json() -> None:
    cap = Capability(bucket="proven")
    assert Capability.model_validate_json(cap.model_dump_json()).bucket == "proven"


# --- scaffolding knob (draft) --------------------------------------------- #


def test_scaffolding_default_is_developing() -> None:
    assert scaffolding_directive() == scaffolding_directive("developing")


def test_scaffolding_distinct_per_bucket() -> None:
    lines = {scaffolding_directive(b) for b in ("new", "developing", "proven")}
    assert len(lines) == 3


@pytest.mark.parametrize(
    ("bucket", "needle"),
    [("new", "NEW"), ("developing", "DEVELOPING"), ("proven", "PROVEN")],
)
def test_scaffolding_names_the_bucket(bucket: SLStyle, needle: str) -> None:
    assert needle in scaffolding_directive(bucket)


def test_scaffolding_unknown_falls_back_to_default() -> None:
    bogus: SLStyle = "bogus"  # type: ignore[assignment]
    assert scaffolding_directive(bogus) == scaffolding_directive(DEFAULT_CAPABILITY)


# --- strictness knob (check) ---------------------------------------------- #


@pytest.mark.parametrize(
    ("bucket", "floor"),
    [
        ("new", Severity.BLOCKER),
        ("developing", Severity.WARNING),
        ("proven", Severity.NOTE),
    ],
)
def test_severity_floor_per_bucket(bucket: SLStyle, floor: Severity) -> None:
    assert severity_floor(bucket) is floor


def test_severity_floor_default_is_warning() -> None:
    assert severity_floor() is Severity.WARNING


def test_severity_floor_unknown_falls_back_to_default() -> None:
    bogus: SLStyle = "bogus"  # type: ignore[assignment]
    assert severity_floor(bogus) is severity_floor(DEFAULT_CAPABILITY)


def test_new_is_stricter_than_proven() -> None:
    # new keeps minor gaps as blockers; proven softens them to notes.
    order = {Severity.NOTE: 0, Severity.WARNING: 1, Severity.BLOCKER: 2}
    assert order[severity_floor("new")] > order[severity_floor("proven")]


# --- persistence via ContextStore ----------------------------------------- #


def test_unrated_recipient_loads_default(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    assert store.load_capability(ContextKey(recipient="alice")).bucket == DEFAULT_CAPABILITY


def test_capability_persists_per_recipient(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    store.save_capability(ContextKey(recipient="alice"), Capability(bucket="proven"))
    store.save_capability(ContextKey(recipient="bob"), Capability(bucket="new"))
    assert store.load_capability(ContextKey(recipient="alice")).bucket == "proven"
    assert store.load_capability(ContextKey(recipient="bob")).bucket == "new"
    # An unrated recipient is untouched by another's rating.
    assert store.load_capability(ContextKey(recipient="carol")).bucket == DEFAULT_CAPABILITY


def test_capability_path_under_capability_dir(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    path = store.capability_path(ContextKey(recipient="alice"))
    assert path.parent == store.capability_dir
    assert path.name == "recipient=alice.json"


def test_malformed_capability_raises(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    path = store.capability_path(ContextKey(recipient="alice"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not valid json", encoding="utf-8")
    with pytest.raises(ProfileLoadError):
        store.load_capability(ContextKey(recipient="alice"))
