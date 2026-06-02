"""Recurrence gating — a trait earns strength by recurring, not from one sample.

New traits are born ``provisional`` with their weight held under a low cap; they
graduate to ``active`` only after enough independent observations. This keeps a
single lucky edit from minting a strong, always-applied rule.

Phase 1 stub: :func:`maybe_graduate` is a no-op, so freshly added traits stay
provisional (and therefore do not render) until Phase 2 lands the real logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from workspec.profile import VoiceTrait

#: Distinct observations required for a provisional trait to graduate to active.
GRADUATION_OBSERVATIONS = 3
#: While provisional, a trait's stored weight is clamped to at most this value.
PROVISIONAL_WEIGHT_CAP = 0.5


def maybe_graduate(trait: VoiceTrait) -> None:
    """Graduate ``trait`` provisional → active once it has recurred enough.

    A provisional trait becomes ``active`` when ``observations >=
    GRADUATION_OBSERVATIONS``; while still provisional its weight is clamped to
    ``PROVISIONAL_WEIGHT_CAP``. Retired traits are left untouched (a penalized
    trait is never silently un-retired). Mutates ``trait`` in place.

    Pure and deterministic: depends only on the trait's own ``observations`` and
    ``status``.
    """
    if trait.status == "retired":
        return
    if trait.observations >= GRADUATION_OBSERVATIONS:
        trait.status = "active"
    if trait.status == "provisional" and trait.weight > PROVISIONAL_WEIGHT_CAP:
        trait.weight = PROVISIONAL_WEIGHT_CAP
