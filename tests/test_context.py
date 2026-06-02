"""Tests for ContextKey addressing and the value types in workspec.context."""

from __future__ import annotations

from workspec.context import (
    DEFAULT_CAPABILITY,
    GLOBAL_SCOPE,
    CapabilitySignal,
    ContextKey,
)


def test_empty_key_is_global() -> None:
    key = ContextKey()
    assert key.is_global() is True
    assert key.scope_id == GLOBAL_SCOPE


def test_recipient_key_is_not_global() -> None:
    key = ContextKey(recipient="alice")
    assert key.is_global() is False
    assert key.scope_id == "recipient=alice"


def test_scope_id_is_order_independent() -> None:
    a = ContextKey(channel="email", recipient="alice")
    b = ContextKey(recipient="alice", channel="email")
    assert a.scope_id == b.scope_id == "channel=email__recipient=alice"


def test_scope_id_includes_all_set_axes() -> None:
    key = ContextKey(channel="slack", project="atlas", recipient="bob")
    assert key.scope_id == "channel=slack__project=atlas__recipient=bob"


def test_backoff_chain_global_only() -> None:
    chain = ContextKey().backoff_chain()
    assert chain == [ContextKey()]


def test_backoff_chain_recipient_then_global() -> None:
    key = ContextKey(recipient="alice")
    chain = key.backoff_chain()
    assert chain == [key, ContextKey()]
    # Most-specific first, global last.
    assert chain[0].scope_id == "recipient=alice"
    assert chain[-1].scope_id == GLOBAL_SCOPE


def test_capability_signal_defaults_to_developing() -> None:
    assert CapabilitySignal().bucket == "developing"
    assert DEFAULT_CAPABILITY == "developing"


def test_capability_signal_is_explicit_only() -> None:
    # The bucket is whatever was set; there is no inference path.
    assert CapabilitySignal(bucket="proven").bucket == "proven"
