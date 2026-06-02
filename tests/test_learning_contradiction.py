"""Unit tests for contradiction resolution (``workspec.learning.contradiction``).

Fully deterministic: the default heuristic is lexical and the strength ordering
is driven by explicit weights/observations/timestamps. No network involved.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from workspec.learning import contradiction
from workspec.learning.contradiction import (
    _default_contradicts,
    detect_and_resolve,
)
from workspec.profile import VoiceProfile, VoiceTrait


def _iso(days_ago: float = 0.0) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


def _trait(
    rule: str,
    *,
    category: str = "tone",
    weight: float = 0.7,
    status: str = "active",
    observations: int = 1,
    last_seen: str | None = None,
) -> VoiceTrait:
    return VoiceTrait(
        category=category,  # type: ignore[arg-type]
        rule=rule,
        weight=weight,
        status=status,  # type: ignore[arg-type]
        observations=observations,
        last_seen=last_seen or _iso(),
    )


# --- default heuristic: antonyms ------------------------------------------- #


def test_antonym_pair_is_detected() -> None:
    assert _default_contradicts("Be warm.", "Be cold and terse.")
    assert _default_contradicts("Keep replies short.", "Write long, thorough replies.")
    assert _default_contradicts("Be formal.", "Keep it casual.")
    assert _default_contradicts("Use bullets.", "Write in prose.")


def test_antonym_is_symmetric() -> None:
    assert _default_contradicts("Be cold.", "Be warm.")
    assert _default_contradicts("Write long replies.", "Keep it short.")


def test_unrelated_rules_do_not_contradict() -> None:
    assert not _default_contradicts("Be warm.", "Sign off with 'Cheers'.")
    assert not _default_contradicts("Use bullets.", "Be warm.")


def test_double_negation_antonym_cancels() -> None:
    # "be warm" vs "do not be cold" both want warmth -> no conflict.
    assert not _default_contradicts("Be warm.", "Do not be cold.")


# --- default heuristic: negation of shared token --------------------------- #


def test_negation_of_shared_token_conflicts() -> None:
    assert _default_contradicts("Use bullet points.", "Do not use bullet points.")
    assert _default_contradicts("Open with a greeting.", "Never open with a greeting.")


def test_same_polarity_shared_token_no_conflict() -> None:
    assert not _default_contradicts("Use bullet points.", "Use bullet points often.")


def test_both_negated_shared_token_no_conflict() -> None:
    assert not _default_contradicts("Do not use bullet points.", "Never use bullet points.")


def test_empty_rule_does_not_contradict() -> None:
    assert not _default_contradicts("", "Be warm.")
    assert not _default_contradicts("Be warm.", "   ")


# --- detect_and_resolve: scoping ------------------------------------------- #


def test_retires_weaker_existing_trait() -> None:
    weak = _trait("Be cold and terse.", weight=0.4)
    strong = _trait("Be warm.", weight=0.9)
    profile = VoiceProfile(traits=[weak, strong])

    retired = detect_and_resolve(profile, strong)

    assert retired == [weak]
    assert weak.status == "retired"
    assert strong.status == "active"


def test_new_trait_retired_when_weaker() -> None:
    strong_existing = _trait("Be warm.", weight=0.95, observations=5)
    weak_new = _trait("Be cold.", weight=0.3)
    profile = VoiceProfile(traits=[strong_existing, weak_new])

    retired = detect_and_resolve(profile, weak_new)

    # The new trait lost, so it is retired but NOT returned (return = others).
    assert retired == []
    assert weak_new.status == "retired"
    assert strong_existing.status == "active"


def test_ties_favour_new_trait() -> None:
    ls = _iso()
    existing = _trait("Be cold.", weight=0.7, observations=2, last_seen=ls)
    new = _trait("Be warm.", weight=0.7, observations=2, last_seen=ls)
    profile = VoiceProfile(traits=[existing, new])

    retired = detect_and_resolve(profile, new)

    assert retired == [existing]
    assert existing.status == "retired"
    assert new.status == "active"


def test_different_category_ignored() -> None:
    other = _trait("Be cold.", category="phrasing", weight=0.2)
    new = _trait("Be warm.", category="tone", weight=0.9)
    profile = VoiceProfile(traits=[other, new])

    retired = detect_and_resolve(profile, new)

    assert retired == []
    assert other.status == "active"


def test_non_active_traits_ignored() -> None:
    provisional = _trait("Be cold.", weight=0.2, status="provisional")
    already_retired = _trait("Be cold and brief.", weight=0.2, status="retired")
    new = _trait("Be warm.", weight=0.9)
    profile = VoiceProfile(traits=[provisional, already_retired, new])

    retired = detect_and_resolve(profile, new)

    assert retired == []
    assert provisional.status == "provisional"
    assert already_retired.status == "retired"


def test_new_trait_not_compared_to_itself() -> None:
    # A self-contradicting rule must never retire itself just by being present.
    new = _trait("Use bullets, do not use bullets.", weight=0.9)
    profile = VoiceProfile(traits=[new])

    retired = detect_and_resolve(profile, new)

    assert retired == []
    assert new.status == "active"


def test_no_contradiction_leaves_everything_active() -> None:
    a = _trait("Be warm.", weight=0.8)
    b = _trait("Sign off with 'Cheers'.", category="signoff", weight=0.8)
    new = _trait("Use bullet points.", category="formatting", weight=0.8)
    profile = VoiceProfile(traits=[a, b, new])

    retired = detect_and_resolve(profile, new)

    assert retired == []
    assert all(t.status == "active" for t in (a, b, new))


# --- detect_and_resolve: tie-break ordering -------------------------------- #


def test_observations_break_weight_tie() -> None:
    weak = _trait("Be cold.", weight=0.7, observations=1)
    new = _trait("Be warm.", weight=0.7, observations=4)
    profile = VoiceProfile(traits=[weak, new])

    retired = detect_and_resolve(profile, new)

    assert retired == [weak]
    assert weak.status == "retired"


def test_recency_breaks_weight_and_observation_tie() -> None:
    stale = _trait("Be cold.", weight=0.7, observations=2, last_seen=_iso(days_ago=30))
    fresh = _trait("Be warm.", weight=0.7, observations=2, last_seen=_iso(days_ago=0))
    profile = VoiceProfile(traits=[stale, fresh])

    retired = detect_and_resolve(profile, fresh)

    assert retired == [stale]
    assert stale.status == "retired"
    assert fresh.status == "active"


def test_multiple_conflicts_all_resolved() -> None:
    weak1 = _trait("Be cold.", weight=0.3)
    weak2 = _trait("Be terse and cold.", weight=0.4)
    strong_new = _trait("Be warm.", weight=0.95)
    profile = VoiceProfile(traits=[weak1, weak2, strong_new])

    retired = detect_and_resolve(profile, strong_new)

    assert retired == [weak1, weak2]
    assert weak1.status == "retired"
    assert weak2.status == "retired"
    assert strong_new.status == "active"


# --- detect_and_resolve: injectable predicate ------------------------------ #


def test_injected_predicate_overrides_default() -> None:
    # Default heuristic would NOT flag these as contradictory; force it to.
    a = _trait("Sign off with 'Cheers'.", category="signoff", weight=0.4)
    new = _trait("Sign off with 'Best'.", category="signoff", weight=0.9)
    profile = VoiceProfile(traits=[a, new])

    called: list[tuple[str, str]] = []

    def always(x: str, y: str) -> bool:
        called.append((x, y))
        return True

    retired = detect_and_resolve(profile, new, contradicts=always)

    assert retired == [a]
    assert a.status == "retired"
    assert called == [(new.rule, a.rule)]


def test_injected_predicate_can_suppress_all() -> None:
    weak = _trait("Be cold.", weight=0.2)
    new = _trait("Be warm.", weight=0.9)
    profile = VoiceProfile(traits=[weak, new])

    retired = detect_and_resolve(profile, new, contradicts=lambda _a, _b: False)

    assert retired == []
    assert weak.status == "active"


def test_predicate_called_new_rule_first() -> None:
    other = _trait("Be cold.", weight=0.5)
    new = _trait("Be warm.", weight=0.9)
    profile = VoiceProfile(traits=[other, new])

    seen: list[tuple[str, str]] = []

    def record(a: str, b: str) -> bool:
        seen.append((a, b))
        return contradiction._default_contradicts(a, b)

    detect_and_resolve(profile, new, contradicts=record)

    # Contract: contradicts(new_trait.rule, other.rule).
    assert seen == [(new.rule, other.rule)]


def test_empty_profile_returns_empty() -> None:
    new = _trait("Be warm.", weight=0.9)
    assert detect_and_resolve(VoiceProfile(), new) == []
