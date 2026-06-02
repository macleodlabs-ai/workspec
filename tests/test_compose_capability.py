"""Tests for capability resolution in compose(): the backoff fold and knobs."""

from __future__ import annotations

from pathlib import Path

from workspec.capability import Capability, scaffolding_directive
from workspec.compose import _resolve_capability, compose
from workspec.context import DEFAULT_CAPABILITY, ContextKey
from workspec.models import Spec
from workspec.store import ContextStore


def _spec() -> Spec:
    return Spec(type="email_reply", title="Email reply", must_include=["a clear answer"])


def test_default_when_unrated(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    composed = compose(store, ContextKey(recipient="alice"), _spec())
    assert composed.sl_style == DEFAULT_CAPABILITY
    assert composed.scaffolding_directive == scaffolding_directive(DEFAULT_CAPABILITY)


def test_global_key_defaults(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    composed = compose(store, None, _spec())
    assert composed.sl_style == DEFAULT_CAPABILITY


def test_recipient_rating_drives_sl_style(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    store.save_capability(ContextKey(recipient="alice"), Capability(bucket="proven"))
    composed = compose(store, ContextKey(recipient="alice"), _spec())
    assert composed.sl_style == "proven"
    assert composed.scaffolding_directive == scaffolding_directive("proven")


def test_rating_does_not_leak_to_other_recipients(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    store.save_capability(ContextKey(recipient="alice"), Capability(bucket="new"))
    other = compose(store, ContextKey(recipient="bob"), _spec())
    assert other.sl_style == DEFAULT_CAPABILITY


def test_recipient_rating_overrides_global(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    store.save_capability(ContextKey(), Capability(bucket="new"))
    store.save_capability(ContextKey(recipient="alice"), Capability(bucket="proven"))
    # Most-specific scope wins in the backoff fold.
    assert _resolve_capability(store, ContextKey(recipient="alice")) == "proven"


def test_falls_back_to_global_rating(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    store.save_capability(ContextKey(), Capability(bucket="proven"))
    # No recipient rating, but the global scope is rated -> fall back to it.
    assert _resolve_capability(store, ContextKey(recipient="alice")) == "proven"
