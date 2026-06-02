"""Tests for the P0 resolver seam wired into the engine and drafter.

These assert the seam is behavior-identical on global-only data (the prompts the
provider sees are byte-for-byte unchanged) and that contextual learning is
scoped to the right file and never reads inbound prose.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers import FakeProvider
from workspec.compose import compose
from workspec.context import ContextKey
from workspec.draft import DraftAgent, ExtractedTrait, LearnedTraits
from workspec.engine import WorkSpecAgent
from workspec.models import Spec
from workspec.profile import ProfileStore, VoiceProfile, VoiceTrait
from workspec.store import ContextStore


def _spec() -> Spec:
    return Spec(type="email_reply", title="Email reply", must_include=["a clear answer"])


def _active_trait(rule: str) -> VoiceTrait:
    return VoiceTrait(category="tone", rule=rule, provenance="edit", weight=1.0, status="active")


# --- engine.check seam ----------------------------------------------------- #


def test_check_prompt_identical_with_and_without_key(tmp_path: Path) -> None:
    spec = _spec()
    work = "Owned by Sam. Decision: ship Friday."

    p1 = FakeProvider()
    WorkSpecAgent(provider=p1, store=ContextStore(tmp_path)).check(spec, work)
    p2 = FakeProvider()
    WorkSpecAgent(provider=p2, store=ContextStore(tmp_path)).check(spec, work, key=ContextKey())

    assert p1.calls[0]["user"] == p2.calls[0]["user"]
    assert p1.calls[0]["system"] == p2.calls[0]["system"]


def test_check_uses_default_store_when_unset() -> None:
    # No store configured: check still works (compose returns the spec unchanged).
    provider = FakeProvider()
    verdict = WorkSpecAgent(provider=provider).check(_spec(), "Some work to lint.")
    assert verdict.passed is True


# --- draft seam: byte-identical on global-only ----------------------------- #


def test_draft_prompt_identical_legacy_vs_context_store(tmp_path: Path) -> None:
    """A ContextStore on global-only data drafts the exact legacy prompt."""
    profile = VoiceProfile(owner="me", traits=[_active_trait("be concise")])

    legacy_dir = tmp_path / "legacy"
    ProfileStore(legacy_dir).save(profile)
    ctx_store = ContextStore(tmp_path / "ctx")
    ctx_store.save_voice(ContextKey(), profile)

    p_legacy = FakeProvider()
    DraftAgent(provider=p_legacy, profile_store=ProfileStore(legacy_dir)).draft(
        _spec(), "Can you confirm Friday?"
    )
    p_ctx = FakeProvider()
    DraftAgent(provider=p_ctx, store=ctx_store).draft(_spec(), "Can you confirm Friday?")

    assert p_legacy.calls[0]["user"] == p_ctx.calls[0]["user"]


def test_draft_applied_traits_identical_global_only(tmp_path: Path) -> None:
    profile = VoiceProfile(traits=[_active_trait("be warm")])
    ProfileStore(tmp_path / "legacy").save(profile)
    ctx_store = ContextStore(tmp_path / "ctx")
    ctx_store.save_voice(ContextKey(), profile)

    legacy = DraftAgent(provider=FakeProvider(), profile_store=ProfileStore(tmp_path / "legacy"))
    contextual = DraftAgent(provider=FakeProvider(), store=ctx_store)
    assert (
        legacy.draft(_spec(), "Q?").applied_traits == contextual.draft(_spec(), "Q?").applied_traits
    )


# --- learn_from_edit seam: scope + inbound invariant ----------------------- #


def test_learn_writes_to_recipient_scope(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    agent = DraftAgent(provider=FakeProvider(), store=store)
    agent.learn_from_edit(
        draft="long winded draft", sent="short", key=ContextKey(recipient="alice")
    )

    assert store.load_voice(ContextKey(recipient="alice")).traits
    # Global stays empty: recipient learning must not leak.
    assert store.load_voice(ContextKey()).traits == []


def test_learn_defaults_to_global_scope(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    agent = DraftAgent(provider=FakeProvider(), store=store)
    agent.learn_from_edit(draft="long winded draft", sent="short")
    assert store.load_voice(ContextKey()).traits


def test_learn_never_reads_inbound_prose(tmp_path: Path) -> None:
    """The learn prompt is built only from the owner's draft and sent text."""
    provider = FakeProvider()
    agent = DraftAgent(provider=provider, store=ContextStore(tmp_path))
    inbound = "SECRET-INBOUND-PROSE-SHOULD-NEVER-APPEAR"
    agent.learn_from_edit(draft="my long draft", sent="my short sent", feedback="too formal")

    learn_call = provider.calls[0]
    assert inbound not in learn_call["user"]
    assert "my long draft" in learn_call["user"]
    assert "my short sent" in learn_call["user"]


def test_learn_dry_run_writes_nothing(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    agent = DraftAgent(provider=FakeProvider(), store=store)
    traits = agent.learn_from_edit(
        draft="long", sent="short", apply=False, key=ContextKey(recipient="bob")
    )
    assert traits  # extracted
    assert store.load_voice(ContextKey(recipient="bob")).traits == []  # but not persisted


# --- cross-recipient promotion (Decision 6) -------------------------------- #


def _trait_provider(rule: str, category: str = "signoff") -> FakeProvider:
    """A provider whose learn step always extracts one fixed, generalizable trait.

    Fixing the extracted trait lets a test graduate the *same* rule across several
    recipients deterministically, without depending on the embedding server.
    """
    provider = FakeProvider()
    provider.responses[LearnedTraits] = LearnedTraits(
        traits=[ExtractedTrait(category=category, rule=rule, evidence="consistent edit")],
        summary="stable habit",
    )
    return provider


def _graduate_for(agent: DraftAgent, recipient: str, times: int = 3) -> None:
    """Drive enough draft→sent edits to graduate the recipient's trait to active."""
    for i in range(times):
        agent.learn_from_edit(
            draft=f"draft variant {i}",
            sent=f"sent variant {i}",
            key=ContextKey(recipient=recipient),
        )


def test_cross_recipient_recurrence_promotes_to_global(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    rule = "Sign off with 'Cheers'."
    agent = DraftAgent(provider=_trait_provider(rule), store=store)

    # Same trait independently graduates across three distinct recipients.
    for recipient in ("alice", "bob", "carol"):
        _graduate_for(agent, recipient)

    glob = store.load_voice(ContextKey())
    promoted = [t for t in glob.traits if t.rule == rule]
    assert promoted, "trait recurring across recipients should promote to global"
    assert promoted[0].status == "active"


def test_unpromoted_recipient_trait_does_not_leak_to_other_recipient(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    rule = "Sign off with 'Cheers'."
    agent = DraftAgent(provider=_trait_provider(rule), store=store)

    # Only one recipient graduates it -> below the promotion bar.
    _graduate_for(agent, "alice")

    # It stays in alice's scope; neither global nor a different recipient sees it.
    assert any(t.rule == rule for t in store.load_voice(ContextKey(recipient="alice")).traits)
    assert all(t.rule != rule for t in store.load_voice(ContextKey()).traits)
    bob = compose(store, ContextKey(recipient="bob"), _spec())
    assert all(t.rule != rule for t in bob.profile.traits)


def test_recipient_override_beats_global_on_contradiction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recipient trait that contradicts a global one wins in the folded view.

    Pin the lexical dedup path (point the embedding server at a dead address) so
    the assertion is deterministic regardless of whether Ollama is reachable: the
    two rules share no tokens, so dedup does not collapse them and the antonym
    contradiction (formal vs casual) is what resolves the fold.
    """
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:1")
    store = ContextStore(tmp_path)
    store.save_voice(
        ContextKey(),
        VoiceProfile(
            traits=[
                VoiceTrait(
                    category="tone",
                    rule="Write formal replies.",
                    provenance="seed",
                    weight=0.7,
                    status="active",
                )
            ]
        ),
    )
    store.save_voice(
        ContextKey(recipient="alice"),
        VoiceProfile(
            traits=[
                VoiceTrait(
                    category="tone",
                    rule="Prefer casual wording.",
                    provenance="edit",
                    weight=1.0,
                    status="active",
                )
            ]
        ),
    )

    composed = compose(store, ContextKey(recipient="alice"), _spec())
    rules = {t.rule for t in composed.profile.traits if t.status == "active"}
    # The recipient's casual trait survives; the contradicting global formal trait
    # is retired by the fold (child overrides parent on contradiction).
    assert "Prefer casual wording." in rules
    assert "Write formal replies." not in rules
