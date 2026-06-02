"""End-to-end knob tests: draft scaffolding and check strictness vary by bucket.

Uses the fake provider so no network is touched; we assert on the *prompt* the
agent hands the provider, which is where the capability dial does its work.
"""

from __future__ import annotations

from pathlib import Path

from tests.helpers import FakeProvider
from workspec.capability import Capability, scaffolding_directive
from workspec.context import ContextKey
from workspec.draft import DraftAgent
from workspec.engine import _strictness_clause
from workspec.models import Spec
from workspec.store import ContextStore


def _spec() -> Spec:
    return Spec(type="email_reply", title="Email reply", must_include=["a clear answer"])


def _rate(store: ContextStore, recipient: str, bucket: str) -> None:
    store.save_capability(ContextKey(recipient=recipient), Capability(bucket=bucket))  # type: ignore[arg-type]


def _draft_prompt(provider: FakeProvider) -> str:
    """The user prompt of the single draft call recorded by the fake provider."""
    return provider.calls[-1]["user"]


# --- draft scaffolding ---------------------------------------------------- #


def test_new_gets_more_scaffolding_than_proven(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    _rate(store, "newbie", "new")
    _rate(store, "veteran", "proven")
    provider = FakeProvider()
    agent = DraftAgent(provider=provider, store=store)

    agent.draft(_spec(), "Can we ship Friday?", key=ContextKey(recipient="newbie"))
    new_prompt = _draft_prompt(provider)
    agent.draft(_spec(), "Can we ship Friday?", key=ContextKey(recipient="veteran"))
    proven_prompt = _draft_prompt(provider)

    # Same submission, different scaffolding directive injected per bucket.
    assert scaffolding_directive("new") in new_prompt
    assert scaffolding_directive("proven") in proven_prompt
    assert new_prompt != proven_prompt
    assert "maximize scaffolding" in new_prompt
    assert "terse" in proven_prompt


def test_unrated_draft_uses_default_directive(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    provider = FakeProvider()
    agent = DraftAgent(provider=provider, store=store)
    agent.draft(_spec(), "Status?", key=ContextKey(recipient="ghost"))
    assert scaffolding_directive("developing") in _draft_prompt(provider)


def test_legacy_path_injects_default_directive(tmp_path: Path) -> None:
    # No ContextStore -> legacy single-profile path still injects a directive,
    # using the default bucket, so the prompt template never has an empty slot.
    from workspec.profile import ProfileStore

    provider = FakeProvider()
    agent = DraftAgent(provider=provider, profile_store=ProfileStore(tmp_path))
    agent.draft(_spec(), "Status?")
    assert scaffolding_directive("developing") in _draft_prompt(provider)


# --- check strictness ----------------------------------------------------- #


def test_strictness_clause_distinct_per_bucket() -> None:
    clauses = {_strictness_clause(b) for b in ("new", "developing", "proven")}
    assert len(clauses) == 3


def test_check_prompt_strictness_varies_by_bucket(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    _rate(store, "newbie", "new")
    _rate(store, "veteran", "proven")
    from workspec.engine import WorkSpecAgent

    provider = FakeProvider()
    agent = WorkSpecAgent(provider=provider, store=store)

    agent.check(_spec(), "We are on track.", key=ContextKey(recipient="newbie"))
    new_prompt = provider.calls[-1]["user"]
    agent.check(_spec(), "We are on track.", key=ContextKey(recipient="veteran"))
    proven_prompt = provider.calls[-1]["user"]

    assert _strictness_clause("new") in new_prompt
    assert _strictness_clause("proven") in proven_prompt
    assert "blocker" in new_prompt
    assert "benefit of the doubt" in proven_prompt


def test_check_unrated_uses_developing_strictness(tmp_path: Path) -> None:
    from workspec.engine import WorkSpecAgent

    store = ContextStore(tmp_path)
    provider = FakeProvider()
    agent = WorkSpecAgent(provider=provider, store=store)
    agent.check(_spec(), "On track.", key=ContextKey(recipient="ghost"))
    assert _strictness_clause("developing") in provider.calls[-1]["user"]


def test_check_no_key_uses_developing_strictness(tmp_path: Path) -> None:
    from workspec.engine import WorkSpecAgent

    store = ContextStore(tmp_path)
    provider = FakeProvider()
    agent = WorkSpecAgent(provider=provider, store=store)
    agent.check(_spec(), "On track.")
    assert _strictness_clause("developing") in provider.calls[-1]["user"]
