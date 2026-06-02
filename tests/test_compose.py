"""Tests for compose(): the per-scope voice fold and byte-identical fallback."""

from __future__ import annotations

from pathlib import Path

from workspec.compose import ComposedContext, compose
from workspec.context import DEFAULT_CAPABILITY, ContextKey
from workspec.contract import ContractDelta, ContractElement
from workspec.models import Spec
from workspec.profile import VoiceProfile, VoiceTrait
from workspec.store import ContextStore


def _spec() -> Spec:
    return Spec(type="email_reply", title="Email reply", must_include=["a clear answer"])


def _gating(rule: str, kind: str = "must_include") -> ContractElement:
    return ContractElement(kind=kind, rule=rule, status="active", confirmed=True)  # type: ignore[arg-type]


def _proposal(rule: str, kind: str = "must_include") -> ContractElement:
    return ContractElement(kind=kind, rule=rule, status="active", confirmed=False)  # type: ignore[arg-type]


def _active_trait(rule: str, category: str = "tone") -> VoiceTrait:
    # An active, edit-derived trait so it renders into the prompt block.
    return VoiceTrait(category=category, rule=rule, provenance="edit", weight=1.0, status="active")


def test_returns_base_spec_unchanged(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    spec = _spec()
    composed = compose(store, None, spec)
    assert isinstance(composed, ComposedContext)
    assert composed.spec is spec
    assert composed.sl_style == DEFAULT_CAPABILITY


def test_global_only_block_is_byte_identical_to_legacy(tmp_path: Path) -> None:
    """compose() on global-only data reproduces the legacy inline voice block."""
    profile = VoiceProfile(owner="me", traits=[_active_trait("be concise")])
    store = ContextStore(tmp_path)
    store.save_voice(ContextKey(), profile)

    composed = compose(store, None, _spec())

    legacy_block = "=== VOICE PROFILE ===\n" + profile.render_for_prompt()
    assert composed.voice_block == legacy_block
    assert composed.applied_traits == profile.active_trait_keys()


def test_empty_global_block_matches_legacy_empty(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    composed = compose(store, None, _spec())
    legacy_block = "=== VOICE PROFILE ===\n" + VoiceProfile().render_for_prompt()
    assert composed.voice_block == legacy_block
    assert composed.applied_traits == []


def test_none_key_equals_explicit_global(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    store.save_voice(ContextKey(), VoiceProfile(traits=[_active_trait("be warm")]))
    assert compose(store, None, _spec()).voice_block == (
        compose(store, ContextKey(), _spec()).voice_block
    )


def test_recipient_layer_adds_to_global(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    store.save_voice(
        ContextKey(), VoiceProfile(traits=[_active_trait("Keep paragraphs short.", "structure")])
    )
    store.save_voice(
        ContextKey(recipient="alice"),
        VoiceProfile(traits=[_active_trait("Sign off with 'Cheers'.", "signoff")]),
    )

    composed = compose(store, ContextKey(recipient="alice"), _spec())
    rules = {t.rule for t in composed.profile.traits}
    assert "Keep paragraphs short." in rules
    assert "Sign off with 'Cheers'." in rules


def test_recipient_traits_do_not_leak_to_global(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    store.save_voice(
        ContextKey(), VoiceProfile(traits=[_active_trait("Keep paragraphs short.", "structure")])
    )
    store.save_voice(
        ContextKey(recipient="alice"),
        VoiceProfile(traits=[_active_trait("Sign off with 'Cheers'.", "signoff")]),
    )

    glob = compose(store, None, _spec())
    assert {t.rule for t in glob.profile.traits} == {"Keep paragraphs short."}


# --- contract fold --------------------------------------------------------- #


def test_no_contract_returns_spec_by_identity(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    spec = _spec()
    assert compose(store, None, spec).spec is spec


def test_unconfirmed_proposal_does_not_change_spec(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    store.save_contract(ContextKey(), ContractDelta(elements=[_proposal("State a decision.")]))
    spec = _spec()
    composed = compose(store, None, spec)
    # Propose-first: an un-confirmed proposal never changes the effective spec.
    assert composed.spec is spec
    assert composed.applied_contract == []


def test_confirmed_element_shapes_effective_spec(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    store.save_contract(ContextKey(), ContractDelta(elements=[_gating("State a decision.")]))
    composed = compose(store, None, _spec())
    assert "State a decision." in composed.spec.must_include
    assert "must_include:State a decision." in composed.applied_contract


def test_recipient_contract_does_not_leak_to_global(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    store.save_contract(
        ContextKey(recipient="alice"), ContractDelta(elements=[_gating("Name an owner.")])
    )
    glob = compose(store, None, _spec())
    assert "Name an owner." not in glob.spec.must_include


def test_recipient_contract_folds_over_global(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    store.save_contract(ContextKey(), ContractDelta(elements=[_gating("State a decision.")]))
    store.save_contract(
        ContextKey(recipient="alice"), ContractDelta(elements=[_gating("Name an owner.")])
    )
    composed = compose(store, ContextKey(recipient="alice"), _spec())
    must = composed.spec.must_include
    assert "State a decision." in must
    assert "Name an owner." in must
