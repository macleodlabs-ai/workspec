"""Recency decay — a trait's *effective* weight fades since it was last seen.

Stale style drifts out of the prompt unless re-reinforced, but nothing is
mutated: the stored ``weight`` is preserved and only the *effective* weight used
for ranking/labeling decays. Re-reinforcement refreshes ``last_seen`` and so
restores effective weight.

The decay is a simple exponential half-life: after ``DECAY_HALFLIFE_DAYS`` the
effective weight is half the stored weight, after two half-lives a quarter, and
so on. A freshly-seen trait (or one with a future ``last_seen``) keeps its full
stored weight; an unparseable ``last_seen`` is treated as "no decay" so a
malformed profile never silently zeroes a trait out.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from workspec.profile import VoiceTrait

#: Half-life (days) of a trait's effective weight since it was last reinforced.
DECAY_HALFLIFE_DAYS = 90.0


def effective_weight(trait: VoiceTrait, now: datetime | None = None) -> float:
    """Return ``trait``'s decayed effective weight (non-destructive).

    Computed as ``weight * 0.5 ** (age_days / DECAY_HALFLIFE_DAYS)`` where age is
    measured from ``trait.last_seen`` to ``now`` (default: current UTC time).
    Used for ranking and labeling in ``render_for_prompt``.

    The result never exceeds the stored ``weight``: a future ``last_seen``
    (negative age) is clamped to zero age, so the trait keeps full weight rather
    than being amplified. The trait is never mutated. If ``last_seen`` cannot be
    parsed as an ISO timestamp, the stored ``weight`` is returned unchanged.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    try:
        last_seen = datetime.fromisoformat(trait.last_seen)
    except (ValueError, TypeError):
        return trait.weight

    # Compare in a single frame of reference: a naive ``last_seen`` is assumed to
    # be UTC, and ``now`` is coerced to match so the subtraction is well-defined.
    if last_seen.tzinfo is None:
        last_seen = last_seen.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    age_days = (now - last_seen).total_seconds() / 86_400.0
    if age_days <= 0.0:
        return trait.weight  # fresh (or future) — no decay, never amplified

    return trait.weight * 0.5 ** (age_days / DECAY_HALFLIFE_DAYS)
