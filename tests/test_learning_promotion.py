"""Tests for cross-recipient promotion — earning a trait into the shared layer.

Promotion is the *only* sanctioned path from a recipient scope to the global
scope (Decision 6): un-promoted recipient traits must never leak, and a trait is
promoted only once it has independently graduated across enough distinct
recipients. These tests pin the count logic, the store-walking promotion, and
the negative cases (un-graduated, too few recipients, no leak before the bar).
"""

from __future__ import annotations

from pathlib import Path

from workspec.context import ContextKey
from workspec.learning import promotion
from workspec.profile import VoiceProfile, VoiceTrait
from workspec.store import ContextStore


def _active(rule: str, category: str = "signoff") -> VoiceTrait:
    return VoiceTrait(category=category, rule=rule, provenance="edit", weight=1.0, status="active")


def _provisional(rule: str, category: str = "signoff") -> VoiceTrait:
    return VoiceTrait(
        category=category, rule=rule, provenance="edit", weight=0.4, status="provisional"
    )


def _seed_recipients(store: ContextStore, mapping: dict[str, VoiceTrait | None]) -> None:
    for recipient, trait in mapping.items():
        traits = [trait] if trait is not None else []
        store.save_voice(ContextKey(recipient=recipient), VoiceProfile(traits=traits))


# --- count_graduated_recipients (pure) ------------------------------------- #


def test_count_only_counts_active_matches() -> None:
    rule = "Sign off with 'Cheers'."
    profiles = {
        "alice": VoiceProfile(traits=[_active(rule)]),
        "bob": VoiceProfile(traits=[_active("Sign off with 'Cheers'.")]),
        "carol": VoiceProfile(traits=[_provisional(rule)]),  # not graduated -> excluded
        "dan": VoiceProfile(traits=[]),  # nothing
    }
    assert promotion.count_graduated_recipients(profiles, _active(rule)) == 2


def test_count_ignores_unrelated_traits() -> None:
    target = _active("Sign off with 'Cheers'.")
    profiles = {
        "alice": VoiceProfile(traits=[_active("Open with a one-line summary.", "structure")]),
        "bob": VoiceProfile(traits=[_active("Be warm.", "tone")]),
    }
    assert promotion.count_graduated_recipients(profiles, target) == 0


# --- maybe_promote (store-aware) ------------------------------------------- #


def test_promotes_when_enough_recipients_graduated(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    rule = "Sign off with 'Cheers'."
    _seed_recipients(store, {"alice": _active(rule), "bob": _active(rule), "carol": _active(rule)})

    promoted = promotion.maybe_promote(store, _active(rule))

    assert promoted is not None
    glob = store.load_voice(ContextKey())
    assert any(t.rule == rule and t.status == "active" for t in glob.traits)
    assert "promoted" in promoted.evidence


def test_does_not_promote_below_threshold(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    rule = "Sign off with 'Cheers'."
    _seed_recipients(store, {"alice": _active(rule), "bob": _active(rule)})

    assert promotion.maybe_promote(store, _active(rule)) is None
    # Nothing leaked into the global layer.
    assert store.load_voice(ContextKey()).traits == []


def test_provisional_recipient_traits_do_not_count(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    rule = "Sign off with 'Cheers'."
    # Three recipients but one is only provisional -> only two graduated.
    _seed_recipients(
        store, {"alice": _active(rule), "bob": _active(rule), "carol": _provisional(rule)}
    )
    assert promotion.maybe_promote(store, _active(rule)) is None
    assert store.load_voice(ContextKey()).traits == []


def test_promotion_reinforces_existing_global_not_duplicate(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    rule = "Sign off with 'Cheers'."
    store.save_voice(ContextKey(), VoiceProfile(traits=[_active(rule)]))
    _seed_recipients(store, {"alice": _active(rule), "bob": _active(rule), "carol": _active(rule)})

    promotion.maybe_promote(store, _active(rule))

    glob = store.load_voice(ContextKey())
    # Still one shared trait for this rule — reinforced, not duplicated.
    matching = [t for t in glob.traits if t.rule == rule]
    assert len(matching) == 1


def test_empty_rule_never_promotes(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    assert promotion.maybe_promote(store, _active("   ")) is None


def test_no_recipient_scopes_no_promotion(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    assert promotion.maybe_promote(store, _active("Sign off with 'Cheers'.")) is None
