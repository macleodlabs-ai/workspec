"""Voice-learning v2 — the self-correcting layer over the v1 voice profile.

One module per soundness feature, each independently owned and tested:

  * ``recurrence``    — provisional → active gating by repeated observation.
  * ``decay``         — non-destructive recency decay of effective weight.
  * ``contradiction`` — retire the weaker side of conflicting same-category traits.
  * ``semantic``      — embedding-based dedup of paraphrased rules (lexical fallback).
  * ``negative``      — penalize traits that informed a draft but were edited back out.

Phase 1 ships these as safe stubs (behavior-preserving no-ops); Phase 2 fills in
the bodies. Modules here depend only on :mod:`workspec.profile` types, and import
those under ``TYPE_CHECKING`` to keep the dependency one-directional and avoid
circular imports with ``profile.py``.
"""

from __future__ import annotations

from workspec.learning.contradiction import detect_and_resolve
from workspec.learning.decay import DECAY_HALFLIFE_DAYS, effective_weight
from workspec.learning.negative import (
    NEGATIVE_DECREMENT,
    RETIRE_FLOOR,
    apply_negative_signal,
)
from workspec.learning.recurrence import (
    GRADUATION_OBSERVATIONS,
    PROVISIONAL_WEIGHT_CAP,
    maybe_graduate,
)
from workspec.learning.semantic import SIMILARITY_THRESHOLD, semantic_match

__all__ = [
    "DECAY_HALFLIFE_DAYS",
    "GRADUATION_OBSERVATIONS",
    "NEGATIVE_DECREMENT",
    "PROVISIONAL_WEIGHT_CAP",
    "RETIRE_FLOOR",
    "SIMILARITY_THRESHOLD",
    "apply_negative_signal",
    "detect_and_resolve",
    "effective_weight",
    "maybe_graduate",
    "semantic_match",
]
