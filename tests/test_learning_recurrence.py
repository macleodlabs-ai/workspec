"""Unit tests for recurrence gating (``workspec.learning.recurrence``).

Recurrence is pure and deterministic — graduation depends only on a trait's own
``observations`` and ``status``, so these tests need no network or fixtures.
"""

from __future__ import annotations

import pytest

from workspec.learning import recurrence
from workspec.learning.recurrence import (
    GRADUATION_OBSERVATIONS,
    PROVISIONAL_WEIGHT_CAP,
    maybe_graduate,
)
from workspec.profile import VoiceTrait


def _trait(**kwargs: object) -> VoiceTrait:
    base: dict[str, object] = {"category": "tone", "rule": "Be concise."}
    base.update(kwargs)
    return VoiceTrait(**base)  # type: ignore[arg-type]


# --- constants ------------------------------------------------------------ #


def test_constants_match_contract() -> None:
    assert GRADUATION_OBSERVATIONS == 3
    assert PROVISIONAL_WEIGHT_CAP == 0.5


# --- graduation threshold ------------------------------------------------- #


def test_below_threshold_stays_provisional() -> None:
    t = _trait(status="provisional", observations=GRADUATION_OBSERVATIONS - 1)
    maybe_graduate(t)
    assert t.status == "provisional"


def test_at_threshold_graduates_to_active() -> None:
    t = _trait(status="provisional", observations=GRADUATION_OBSERVATIONS)
    maybe_graduate(t)
    assert t.status == "active"


def test_above_threshold_graduates_to_active() -> None:
    t = _trait(status="provisional", observations=GRADUATION_OBSERVATIONS + 5)
    maybe_graduate(t)
    assert t.status == "active"


def test_single_observation_stays_provisional() -> None:
    # Default observations == 1: a lone lucky edit must not mint an active rule.
    t = _trait(status="provisional")
    assert t.observations == 1
    maybe_graduate(t)
    assert t.status == "provisional"


# --- provisional weight clamp --------------------------------------------- #


def test_provisional_weight_clamped_down() -> None:
    t = _trait(status="provisional", observations=1, weight=0.9)
    maybe_graduate(t)
    assert t.status == "provisional"
    assert t.weight == PROVISIONAL_WEIGHT_CAP


def test_provisional_weight_at_cap_unchanged() -> None:
    t = _trait(status="provisional", observations=1, weight=PROVISIONAL_WEIGHT_CAP)
    maybe_graduate(t)
    assert t.weight == PROVISIONAL_WEIGHT_CAP


def test_provisional_weight_below_cap_unchanged() -> None:
    t = _trait(status="provisional", observations=1, weight=0.3)
    maybe_graduate(t)
    assert t.weight == 0.3


def test_graduating_trait_keeps_high_weight() -> None:
    # Once it graduates it is no longer provisional, so the cap no longer applies.
    t = _trait(status="provisional", observations=GRADUATION_OBSERVATIONS, weight=0.9)
    maybe_graduate(t)
    assert t.status == "active"
    assert t.weight == 0.9


# --- status invariants ---------------------------------------------------- #


def test_retired_trait_not_unretired_even_if_observed() -> None:
    t = _trait(status="retired", observations=GRADUATION_OBSERVATIONS + 10, weight=0.9)
    maybe_graduate(t)
    assert t.status == "retired"
    # Retired traits are left entirely untouched (no clamp either).
    assert t.weight == 0.9


def test_already_active_stays_active() -> None:
    t = _trait(status="active", observations=GRADUATION_OBSERVATIONS, weight=0.8)
    maybe_graduate(t)
    assert t.status == "active"
    assert t.weight == 0.8


def test_active_trait_weight_not_clamped() -> None:
    # An active trait above the provisional cap must keep its weight.
    t = _trait(status="active", observations=1, weight=0.9)
    maybe_graduate(t)
    assert t.status == "active"
    assert t.weight == 0.9


# --- purity / determinism ------------------------------------------------- #


def test_returns_none_and_mutates_in_place() -> None:
    t = _trait(status="provisional", observations=GRADUATION_OBSERVATIONS)
    result = maybe_graduate(t)
    assert result is None


def test_deterministic_across_repeated_calls() -> None:
    t = _trait(status="provisional", observations=1, weight=0.9)
    maybe_graduate(t)
    snapshot = (t.status, t.weight)
    for _ in range(5):
        maybe_graduate(t)
    assert (t.status, t.weight) == snapshot


def test_does_not_touch_unrelated_fields() -> None:
    t = _trait(
        status="provisional",
        observations=GRADUATION_OBSERVATIONS,
        weight=0.4,
        rule="Be concise.",
        provenance="edit",
        hits=7,
    )
    last_seen, updated_at = t.last_seen, t.updated_at
    maybe_graduate(t)
    assert t.rule == "Be concise."
    assert t.provenance == "edit"
    assert t.hits == 7
    assert t.observations == GRADUATION_OBSERVATIONS
    assert t.last_seen == last_seen
    assert t.updated_at == updated_at


# --- progression through repeated observation ----------------------------- #


def test_progression_from_one_to_graduation() -> None:
    """Simulate a trait recurring across edits, graduating at the threshold."""
    t = _trait(status="provisional", observations=1, weight=0.5)
    for obs in range(1, GRADUATION_OBSERVATIONS):
        t.observations = obs
        maybe_graduate(t)
        assert t.status == "provisional", f"should still be provisional at obs={obs}"
    t.observations = GRADUATION_OBSERVATIONS
    maybe_graduate(t)
    assert t.status == "active"


@pytest.mark.parametrize("obs", [0, 1, 2])
def test_low_observation_counts_never_graduate(obs: int) -> None:
    t = _trait(status="provisional", observations=obs)
    maybe_graduate(t)
    assert t.status == "provisional"


def test_module_exposes_expected_symbols() -> None:
    assert callable(recurrence.maybe_graduate)
    assert hasattr(recurrence, "GRADUATION_OBSERVATIONS")
    assert hasattr(recurrence, "PROVISIONAL_WEIGHT_CAP")
