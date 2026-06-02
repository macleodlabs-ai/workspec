"""Voice-learning v2 — the self-correcting layer over the v1 voice profile.

One module per soundness feature, each independently owned and tested:

  * ``recurrence``    — provisional → active gating by repeated observation.
  * ``decay``         — non-destructive recency decay of effective weight.
  * ``contradiction`` — retire the weaker side of conflicting same-category traits.
  * ``semantic``      — embedding-based dedup of paraphrased rules (lexical fallback).
  * ``negative``      — penalize traits that informed a draft but were edited back out.

Callers import the submodules directly (e.g. ``from workspec.learning import
decay``); each depends only on :mod:`workspec.profile` types, imported under
``TYPE_CHECKING`` to keep the dependency one-directional and dodge a circular
import with ``profile.py``.
"""
