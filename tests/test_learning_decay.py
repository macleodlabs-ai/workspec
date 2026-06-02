"""Unit tests for recency decay (``workspec.learning.decay``).

Deterministic: every test pins an explicit ``now`` and ``last_seen`` so no wall
clock is involved. The decay is exponential with a fixed half-life, so the
expected values are computed from the same closed form the implementation uses.
"""

from __future__ import annotations

import itertools
from datetime import datetime, timedelta, timezone

import pytest

from workspec.learning.decay import DECAY_HALFLIFE_DAYS, effective_weight
from workspec.profile import VoiceTrait


def _trait(weight: float = 0.8, last_seen: str | None = None) -> VoiceTrait:
    """A trait fixed at ``last_seen`` (defaults to a known UTC instant)."""
    if last_seen is None:
        last_seen = datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat()
    return VoiceTrait(category="tone", rule="Be concise.", weight=weight, last_seen=last_seen)


def test_constant_value() -> None:
    # The half-life is part of the seam contract — pin it explicitly.
    assert DECAY_HALFLIFE_DAYS == 90.0


def test_fresh_trait_keeps_full_weight() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t = _trait(weight=0.8, last_seen=now.isoformat())
    assert effective_weight(t, now) == pytest.approx(0.8)


def test_one_halflife_halves_weight() -> None:
    seen = datetime(2026, 1, 1, tzinfo=timezone.utc)
    now = seen + timedelta(days=DECAY_HALFLIFE_DAYS)
    t = _trait(weight=0.8, last_seen=seen.isoformat())
    assert effective_weight(t, now) == pytest.approx(0.4)


def test_two_halflives_quarters_weight() -> None:
    seen = datetime(2026, 1, 1, tzinfo=timezone.utc)
    now = seen + timedelta(days=2 * DECAY_HALFLIFE_DAYS)
    t = _trait(weight=0.8, last_seen=seen.isoformat())
    assert effective_weight(t, now) == pytest.approx(0.2)


def test_partial_age_matches_closed_form() -> None:
    seen = datetime(2026, 1, 1, tzinfo=timezone.utc)
    now = seen + timedelta(days=30)
    t = _trait(weight=0.6, last_seen=seen.isoformat())
    expected = 0.6 * 0.5 ** (30 / DECAY_HALFLIFE_DAYS)
    assert effective_weight(t, now) == pytest.approx(expected)


def test_decay_is_monotonic_in_age() -> None:
    seen = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t = _trait(weight=0.9, last_seen=seen.isoformat())
    values = [
        effective_weight(t, seen + timedelta(days=days)) for days in (0, 10, 45, 90, 180, 365)
    ]
    assert values == sorted(values, reverse=True)
    # Strictly decreasing once age is positive (first step is 0 -> 10 days).
    assert all(a > b for a, b in itertools.pairwise(values[1:]))


def test_never_exceeds_stored_weight_for_future_last_seen() -> None:
    # last_seen in the future -> negative age -> clamp to full weight, not amplify.
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    future = now + timedelta(days=365)
    t = _trait(weight=0.7, last_seen=future.isoformat())
    assert effective_weight(t, now) == pytest.approx(0.7)


def test_zero_weight_stays_zero() -> None:
    seen = datetime(2026, 1, 1, tzinfo=timezone.utc)
    now = seen + timedelta(days=45)
    t = _trait(weight=0.0, last_seen=seen.isoformat())
    assert effective_weight(t, now) == pytest.approx(0.0)


def test_does_not_mutate_trait() -> None:
    seen = datetime(2026, 1, 1, tzinfo=timezone.utc)
    now = seen + timedelta(days=200)
    t = _trait(weight=0.8, last_seen=seen.isoformat())
    before = t.model_dump()
    effective_weight(t, now)
    assert t.model_dump() == before


def test_unparseable_last_seen_returns_stored_weight() -> None:
    t = _trait(weight=0.65, last_seen="not-a-timestamp")
    now = datetime(2026, 6, 1, tzinfo=timezone.utc)
    assert effective_weight(t, now) == pytest.approx(0.65)


def test_empty_last_seen_returns_stored_weight() -> None:
    t = _trait(weight=0.5, last_seen="")
    now = datetime(2026, 6, 1, tzinfo=timezone.utc)
    assert effective_weight(t, now) == pytest.approx(0.5)


def test_naive_last_seen_assumed_utc() -> None:
    # A naive ISO string (no offset) is treated as UTC, matching the aware path.
    seen_naive = "2026-01-01T00:00:00"
    now = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=DECAY_HALFLIFE_DAYS)
    t = _trait(weight=0.8, last_seen=seen_naive)
    assert effective_weight(t, now) == pytest.approx(0.4)


def test_naive_now_assumed_utc() -> None:
    # A naive ``now`` is coerced to UTC so it can subtract an aware last_seen.
    seen = datetime(2026, 1, 1, tzinfo=timezone.utc)
    now_naive = datetime(2026, 1, 1) + timedelta(days=DECAY_HALFLIFE_DAYS)
    t = _trait(weight=0.8, last_seen=seen.isoformat())
    assert effective_weight(t, now_naive) == pytest.approx(0.4)


def test_default_now_is_current_time() -> None:
    # With no explicit now, a far-past last_seen decays well below stored weight,
    # and a just-now last_seen stays at (approximately) full weight.
    old = _trait(weight=0.9, last_seen=datetime(2000, 1, 1, tzinfo=timezone.utc).isoformat())
    assert effective_weight(old) < 0.1

    fresh = _trait(weight=0.9, last_seen=datetime.now(timezone.utc).isoformat())
    assert effective_weight(fresh) == pytest.approx(0.9, abs=1e-3)
