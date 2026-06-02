"""Unit tests for the voice profile, reinforcement, and persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from workspec.learning.recurrence import GRADUATION_OBSERVATIONS, PROVISIONAL_WEIGHT_CAP
from workspec.profile import (
    PROVENANCE_WEIGHT,
    LearnMetric,
    ProfileLoadError,
    ProfileStore,
    VoiceProfile,
    VoiceTrait,
)


@pytest.fixture(autouse=True)
def _disable_semantic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the lexical dedup path so these tests are hermetic.

    ``reinforce_or_add`` tries ``semantic.semantic_match`` first; with a real
    Ollama reachable that would make lexical-dedup assertions environment-
    dependent. Semantic matching has its own suite (test_learning_semantic.py).
    """
    monkeypatch.setattr("workspec.profile.semantic.semantic_match", lambda *a, **k: None)


def test_voicetrait_defaults() -> None:
    t = VoiceTrait(category="tone", rule="Be concise.")
    assert t.provenance == "seed"
    assert t.weight == 0.7
    assert t.hits == 1
    assert t.updated_at  # stamped
    # v2 model: traits are born provisional, with one observation and a key.
    assert t.status == "provisional"
    assert t.observations == 1
    assert t.last_seen  # stamped
    assert t.key == "tone:Be concise."


def test_render_for_prompt_empty() -> None:
    out = VoiceProfile().render_for_prompt()
    assert "No learned voice profile yet" in out


def test_render_for_prompt_groups_and_orders() -> None:
    # Only 'active' traits render, so graduate them explicitly for this test.
    profile = VoiceProfile(
        traits=[
            VoiceTrait(category="tone", rule="Warm but brief.", weight=0.9, status="active"),
            VoiceTrait(
                category="do_not", rule="Never say 'circle back'.", weight=1.0, status="active"
            ),
            VoiceTrait(category="signoff", rule="Sign off 'Cheers'.", weight=0.6, status="active"),
        ]
    )
    out = profile.render_for_prompt()
    assert "HOW THIS PERSON WRITES:" in out
    assert "NEVER DO (hard constraints):" in out
    assert "circle back" in out
    # strongest positive trait appears before the weaker one
    assert out.index("Warm but brief.") < out.index("Cheers")


def test_render_for_prompt_min_weight_filters_all() -> None:
    profile = VoiceProfile(
        traits=[VoiceTrait(category="tone", rule="x", weight=0.5, status="active")]
    )
    assert "neutrally" in profile.render_for_prompt(min_weight=0.9).lower()


def test_reinforce_or_add_adds_new_trait() -> None:
    profile = VoiceProfile()
    trait = profile.reinforce_or_add(category="tone", rule="Be concise.", provenance="edit")
    assert trait in profile.traits
    assert trait.provenance == "edit"
    # Born provisional, so its stored weight is held under the provisional cap
    # even for an 'edit' (ceiling 1.0) trait until it recurs enough to graduate.
    assert trait.status == "provisional"
    assert trait.weight <= PROVENANCE_WEIGHT["edit"]
    assert trait.weight == PROVISIONAL_WEIGHT_CAP


def test_reinforce_existing_increments_hits_and_weight() -> None:
    profile = VoiceProfile()
    first = profile.reinforce_or_add(
        category="tone", rule="Be very concise please", provenance="feedback"
    )
    # While provisional, recurrence clamps weight to the provisional cap; weight
    # only climbs once the trait recurs enough to graduate to 'active'. Reinforce
    # past GRADUATION_OBSERVATIONS so this exercises the post-graduation growth.
    again = profile.reinforce_or_add(
        category="tone", rule="Be very concise please now", provenance="edit"
    )
    assert again is first
    w_active = first.weight
    profile.reinforce_or_add(category="tone", rule="Be very concise please now", provenance="edit")
    assert first.status == "active"
    assert first.hits == 3
    assert first.weight > w_active
    assert first.provenance == "edit"  # upgraded to stronger provenance
    assert len(profile.traits) == 1


def test_feedback_weight_never_exceeds_ceiling() -> None:
    """A feedback-only trait must stay at or below its 0.9 provenance ceiling,
    no matter how many times it is reinforced (it must not tie an 'edit' trait)."""
    profile = VoiceProfile()
    trait = profile.reinforce_or_add(
        category="tone", rule="Keep it warm and friendly", provenance="feedback"
    )
    # Born provisional and clamped under the cap; the 0.9 ceiling only bounds it
    # once it has graduated and its weight is free to climb.
    assert trait.weight == PROVISIONAL_WEIGHT_CAP
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


def test_find_similar_lexical_match_when_semantic_unavailable(monkeypatch) -> None:
    """When semantic matching is unavailable, the lexical Jaccard fallback still
    collapses a highly-overlapping same-category rule into the existing trait."""
    from workspec.learning import semantic

    # Force the semantic path to abstain so the lexical _find_similar carries it.
    monkeypatch.setattr(semantic, "semantic_match", lambda *a, **k: None)

    profile = VoiceProfile()
    first = profile.reinforce_or_add(
        category="tone", rule="Be very concise please", provenance="edit"
    )
    again = profile.reinforce_or_add(
        category="tone", rule="Be very concise please now", provenance="edit"
    )
    assert again is first
    assert len(profile.traits) == 1


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


def test_load_raises_on_corrupt_profile(tmp_path: Path) -> None:
    """A hand-edited/truncated profile surfaces ProfileLoadError, not a raw traceback."""
    store = ProfileStore(tmp_path)
    store.dir.mkdir(parents=True, exist_ok=True)
    store.path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ProfileLoadError):
        store.load()


def test_save_is_atomic_no_temp_left(tmp_path: Path) -> None:
    """save() round-trips and leaves no .tmp turd behind."""
    store = ProfileStore(tmp_path)
    profile = VoiceProfile(owner="sam")
    profile.reinforce_or_add(category="tone", rule="Be warm", provenance="edit")
    store.save(profile)
    assert ProfileStore(tmp_path).load().owner == "sam"
    assert not list(store.dir.glob("*.tmp"))


def test_reinforce_skips_retired_traits() -> None:
    """A retired trait is out of play: re-learning its rule must not revive it.

    Regression for a bug where a contradiction-retired trait got re-matched by
    lexical dedup, reinforced back up, and then retired the active trait that had
    beaten it. Reinforcement must create a fresh trait instead of touching the
    retired one.
    """
    profile = VoiceProfile()
    retired = VoiceTrait(category="tone", rule="Be terse", provenance="edit", status="retired")
    profile.traits.append(retired)

    added = profile.reinforce_or_add(category="tone", rule="Be terse", provenance="edit")

    assert added is not retired
    assert retired.status == "retired"  # untouched
    assert retired.hits == 1
    assert len([t for t in profile.traits if t.category == "tone"]) == 2


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


# --- graduation lifecycle -------------------------------------------------- #


def test_graduation_lifecycle_provisional_then_active_renders() -> None:
    """A brand-new trait is provisional and does NOT render; after
    GRADUATION_OBSERVATIONS reinforcements it becomes active and renders."""
    profile = VoiceProfile()
    trait = profile.reinforce_or_add(
        category="tone", rule="Lead with the answer.", provenance="edit"
    )
    # Born provisional with a single observation, weight under the provisional cap.
    assert trait.status == "provisional"
    assert trait.observations == 1
    assert trait.weight <= 0.5
    assert "Lead with the answer." not in profile.render_for_prompt()
    assert profile.active_trait_keys() == []

    # Reinforce until it reaches the graduation threshold.
    for _ in range(GRADUATION_OBSERVATIONS - 1):
        profile.reinforce_or_add(category="tone", rule="Lead with the answer.", provenance="edit")

    assert trait.status == "active"
    assert trait.observations == GRADUATION_OBSERVATIONS
    assert "Lead with the answer." in profile.render_for_prompt()
    assert trait.key in profile.active_trait_keys()


def test_profile_store_roundtrip_preserves_v2_fields(tmp_path: Path) -> None:
    """Save/load round-trips status, observations, last_seen, and metrics."""
    store = ProfileStore(tmp_path / ".workspec")
    profile = VoiceProfile(owner="sam")
    trait = profile.reinforce_or_add(category="length", rule="Keep it short.", provenance="edit")
    for _ in range(GRADUATION_OBSERVATIONS - 1):
        profile.reinforce_or_add(category="length", rule="Keep it short.", provenance="edit")
    assert trait.status == "active"
    profile.metrics.append(LearnMetric(edit_ratio=0.42))
    store.save(profile)

    loaded = ProfileStore(tmp_path / ".workspec").load()
    assert len(loaded.traits) == 1
    lt = loaded.traits[0]
    assert lt.status == "active"
    assert lt.observations == GRADUATION_OBSERVATIONS
    assert lt.last_seen == trait.last_seen
    assert lt.key == trait.key
    assert len(loaded.metrics) == 1
    assert loaded.metrics[0].edit_ratio == 0.42


# --- stats (eval surface) -------------------------------------------------- #


def test_stats_empty_profile() -> None:
    s = VoiceProfile().stats()
    assert s.total == 0
    assert s.counts == {"provisional": 0, "active": 0, "retired": 0}
    assert s.top_active == []
    assert s.metric_count == 0
    assert s.recent_edit_ratio is None
    assert s.edit_ratio_delta is None


def test_stats_counts_top_active_and_trend() -> None:
    profile = VoiceProfile(
        traits=[
            VoiceTrait(category="tone", rule="Strong.", weight=0.9, status="active"),
            VoiceTrait(category="length", rule="Weaker.", weight=0.4, status="active"),
            VoiceTrait(category="phrasing", rule="New.", status="provisional"),
            VoiceTrait(category="do_not", rule="Gone.", weight=0.1, status="retired"),
        ]
    )
    profile.metrics = [LearnMetric(edit_ratio=r) for r in (0.4, 0.4, 0.9, 0.9)]

    s = profile.stats(top=5, recent=2)
    assert s.total == 4
    assert s.counts == {"provisional": 1, "active": 2, "retired": 1}
    # only active traits, ranked by effective weight (strongest first)
    assert [t.rule for t in s.top_active] == ["Strong.", "Weaker."]
    assert s.metric_count == 4
    # recent two (0.9, 0.9) vs older two (0.4, 0.4) -> positive delta
    assert s.recent_edit_ratio == 0.9
    assert s.edit_ratio_delta is not None
    assert s.edit_ratio_delta > 0


def test_stats_trend_declines_when_editing_increases() -> None:
    """A falling edit ratio (drafts need MORE editing) yields a negative delta."""
    profile = VoiceProfile()
    profile.metrics = [LearnMetric(edit_ratio=r) for r in (0.9, 0.9, 0.4, 0.4)]
    s = profile.stats(recent=2)
    assert s.edit_ratio_delta is not None
    assert s.edit_ratio_delta < 0


def test_stats_top_respects_limit() -> None:
    profile = VoiceProfile(
        traits=[
            VoiceTrait(category="tone", rule=f"Rule {i}", weight=0.5 + i / 100, status="active")
            for i in range(5)
        ]
    )
    assert len(profile.stats(top=2).top_active) == 2
    assert profile.stats(top=0).top_active == []


def test_stats_single_metric_has_no_delta() -> None:
    """With only recent metrics and no older ones, the delta is undefined (None)."""
    profile = VoiceProfile()
    profile.metrics = [LearnMetric(edit_ratio=0.7)]
    s = profile.stats()
    assert s.recent_edit_ratio == 0.7
    assert s.edit_ratio_delta is None


# --- graft_trait (cross-profile carry; promotion & fold mechanism) --------- #


def test_graft_new_trait_preserves_earned_status() -> None:
    """Grafting an unmatched active trait keeps its earned status/weight intact."""
    profile = VoiceProfile()
    incoming = VoiceTrait(
        category="signoff",
        rule="Sign off with 'Cheers'.",
        provenance="edit",
        weight=0.95,
        status="active",
        observations=4,
    )
    grafted = profile.graft_trait(incoming)
    assert grafted.status == "active"
    assert grafted.weight == 0.95
    assert grafted is not incoming  # a copy, not the original object
    assert profile.traits == [grafted]


def test_graft_upgrades_existing_status_and_provenance() -> None:
    """Grafting a stronger trait onto a weaker match upgrades status+provenance."""
    profile = VoiceProfile(
        traits=[
            VoiceTrait(
                category="signoff",
                rule="Sign off with 'Cheers'.",
                provenance="seed",
                weight=0.4,
                status="provisional",
                observations=1,
            )
        ]
    )
    incoming = VoiceTrait(
        category="signoff",
        rule="Sign off with 'Cheers'.",
        provenance="edit",
        weight=0.9,
        status="active",
        observations=3,
    )
    grafted = profile.graft_trait(incoming)
    assert len(profile.traits) == 1  # reinforced, not duplicated
    assert grafted.status == "active"  # upgraded (line 315)
    assert grafted.provenance == "edit"  # upgraded (line 319)
    assert grafted.weight == 0.9
    assert grafted.observations == 4  # 1 existing + 3 incoming


def test_graft_does_not_downgrade_existing() -> None:
    """A weaker incoming trait never demotes a stronger existing one."""
    profile = VoiceProfile(
        traits=[
            VoiceTrait(
                category="signoff",
                rule="Sign off with 'Cheers'.",
                provenance="edit",
                weight=0.9,
                status="active",
            )
        ]
    )
    incoming = VoiceTrait(
        category="signoff",
        rule="Sign off with 'Cheers'.",
        provenance="seed",
        weight=0.3,
        status="provisional",
    )
    grafted = profile.graft_trait(incoming)
    assert grafted.status == "active"
    assert grafted.provenance == "edit"
    assert grafted.weight == 0.9


def test_graft_accumulated_observations_graduate() -> None:
    """Summed observations crossing the gate graduate the trait, as in reinforce."""
    profile = VoiceProfile(
        traits=[
            VoiceTrait(
                category="signoff",
                rule="Sign off with 'Cheers'.",
                provenance="edit",
                weight=0.4,
                status="provisional",
                observations=2,
            )
        ]
    )
    incoming = VoiceTrait(
        category="signoff",
        rule="Sign off with 'Cheers'.",
        provenance="edit",
        weight=0.4,
        status="provisional",
        observations=1,
    )
    grafted = profile.graft_trait(incoming)
    assert grafted.observations == 3
    assert grafted.status == "active"  # graduated by recurrence, not left provisional
