"""Unit tests for the voice profile, reinforcement, and persistence."""

from __future__ import annotations

from pathlib import Path

from workspec.profile import (
    PROVENANCE_WEIGHT,
    ProfileStore,
    VoiceProfile,
    VoiceTrait,
)


def test_voicetrait_defaults() -> None:
    t = VoiceTrait(category="tone", rule="Be concise.")
    assert t.provenance == "seed"
    assert t.weight == 0.7
    assert t.hits == 1
    assert t.updated_at  # stamped


def test_render_for_prompt_empty() -> None:
    out = VoiceProfile().render_for_prompt()
    assert "No learned voice profile yet" in out


def test_render_for_prompt_groups_and_orders() -> None:
    profile = VoiceProfile(
        traits=[
            VoiceTrait(category="tone", rule="Warm but brief.", weight=0.9),
            VoiceTrait(category="do_not", rule="Never say 'circle back'.", weight=1.0),
            VoiceTrait(category="signoff", rule="Sign off 'Cheers'.", weight=0.6),
        ]
    )
    out = profile.render_for_prompt()
    assert "HOW THIS PERSON WRITES:" in out
    assert "NEVER DO (hard constraints):" in out
    assert "circle back" in out
    # strongest positive trait appears before the weaker one
    assert out.index("Warm but brief.") < out.index("Cheers")


def test_render_for_prompt_min_weight_filters_all() -> None:
    profile = VoiceProfile(traits=[VoiceTrait(category="tone", rule="x", weight=0.5)])
    assert "neutrally" in profile.render_for_prompt(min_weight=0.9).lower()


def test_reinforce_or_add_adds_new_trait() -> None:
    profile = VoiceProfile()
    trait = profile.reinforce_or_add(category="tone", rule="Be concise.", provenance="edit")
    assert trait in profile.traits
    assert trait.provenance == "edit"
    assert trait.weight == PROVENANCE_WEIGHT["edit"]


def test_reinforce_existing_increments_hits_and_weight() -> None:
    profile = VoiceProfile()
    first = profile.reinforce_or_add(
        category="tone", rule="Be very concise please", provenance="feedback"
    )
    w0 = first.weight
    # Highly overlapping rule, same category -> treated as the same trait
    again = profile.reinforce_or_add(
        category="tone", rule="Be very concise please now", provenance="edit"
    )
    assert again is first
    assert first.hits == 2
    assert first.weight > w0
    assert first.provenance == "edit"  # upgraded to stronger provenance
    assert len(profile.traits) == 1


def test_feedback_weight_never_exceeds_ceiling() -> None:
    """A feedback-only trait must stay at or below its 0.9 provenance ceiling,
    no matter how many times it is reinforced (it must not tie an 'edit' trait)."""
    profile = VoiceProfile()
    trait = profile.reinforce_or_add(
        category="tone", rule="Keep it warm and friendly", provenance="feedback"
    )
    assert trait.weight == PROVENANCE_WEIGHT["feedback"]  # 0.9
    for _ in range(50):
        profile.reinforce_or_add(
            category="tone", rule="Keep it warm and friendly", provenance="feedback"
        )
    assert trait.weight <= PROVENANCE_WEIGHT["feedback"]
    assert trait.weight == 0.9


def test_edit_weight_approaches_one() -> None:
    """An edit trait (ceiling 1.0) may climb toward 1.0 with reinforcement."""
    profile = VoiceProfile()
    trait = profile.reinforce_or_add(
        category="tone", rule="Lead with the answer", provenance="edit"
    )
    for _ in range(50):
        profile.reinforce_or_add(category="tone", rule="Lead with the answer", provenance="edit")
    assert trait.weight <= 1.0
    assert trait.weight > 0.95


def test_find_similar_is_order_independent() -> None:
    """Dedup must not depend on insertion order (symmetric Jaccard)."""
    short_rule = "be concise"
    long_rule = "be concise and direct in every reply"

    short_first = VoiceProfile()
    short_first.reinforce_or_add(category="tone", rule=short_rule, provenance="edit")
    short_first.reinforce_or_add(category="tone", rule=long_rule, provenance="edit")

    long_first = VoiceProfile()
    long_first.reinforce_or_add(category="tone", rule=long_rule, provenance="edit")
    long_first.reinforce_or_add(category="tone", rule=short_rule, provenance="edit")

    # Same dedup decision regardless of which order they arrived in.
    assert len(short_first.traits) == len(long_first.traits)


def test_find_similar_skips_empty_token_traits() -> None:
    """An empty rule has no tokens, so the union is empty and it can't match a
    pre-existing empty-rule trait; both are kept as distinct entries."""
    profile = VoiceProfile()
    profile.traits.append(VoiceTrait(category="tone", rule="", provenance="edit"))
    added = profile.reinforce_or_add(category="tone", rule="", provenance="edit")

    # The empty-union branch means no dedup match: a new, separate trait is added.
    assert len([t for t in profile.traits if t.category == "tone"]) == 2
    assert added.hits == 1


def test_reinforce_updates_evidence_when_provided() -> None:
    profile = VoiceProfile()
    profile.reinforce_or_add(category="tone", rule="Be very concise please", provenance="edit")
    again = profile.reinforce_or_add(
        category="tone",
        rule="Be very concise please now",
        provenance="edit",
        evidence="trimmed the preamble",
    )
    assert again.evidence == "trimmed the preamble"


def test_reinforce_different_category_is_separate() -> None:
    profile = VoiceProfile()
    profile.reinforce_or_add(category="tone", rule="Be concise.", provenance="edit")
    profile.reinforce_or_add(category="signoff", rule="Be concise.", provenance="edit")
    assert len(profile.traits) == 2


def test_profile_store_roundtrip(tmp_path: Path) -> None:
    store = ProfileStore(tmp_path / ".workspec")
    assert not store.exists()
    assert store.load().traits == []  # missing file -> empty profile

    profile = VoiceProfile(owner="sam")
    profile.reinforce_or_add(category="tone", rule="Be concise.", provenance="edit")
    store.save(profile)

    assert store.exists()
    loaded = ProfileStore(tmp_path / ".workspec").load()
    assert loaded.owner == "sam"
    assert len(loaded.traits) == 1
    assert loaded.traits[0].rule == "Be concise."
