"""Contradiction resolution — retire the weaker side of conflicting traits.

When two ``active`` traits in the same category pull opposite ways ("be warm"
vs "be terse"), the prompt becomes incoherent. This module detects such
conflicts against a newly reinforced trait and retires the weaker/older side by
setting ``status='retired'`` (non-destructive: the trait is kept for auditing).

The conflict test is pluggable via ``contradicts(a_rule, b_rule) -> bool``. The
default heuristic is intentionally lightweight: it fires when one rule negates a
salient token of the other, or when the two rules sit on opposite ends of a
known antonym pair (warm/cold, short/long, formal/casual, bullets/prose, ...).
It is a *cue* detector, not a semantic model — false negatives are preferred to
spuriously retiring an unrelated trait. The lexical primitives (tokenizer,
negation cues, antonym table, conflict tests) are shared via :mod:`._text`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from workspec.learning import _text, decay

if TYPE_CHECKING:
    from collections.abc import Callable

    from workspec.profile import VoiceProfile, VoiceTrait


def _default_contradicts(a_rule: str, b_rule: str) -> bool:
    """Lightweight negation/antonym-cue contradiction heuristic."""
    a_tokens = _text.tokens(a_rule)
    b_tokens = _text.tokens(b_rule)
    if not a_tokens or not b_tokens:
        return False
    return _text.negation_conflict(a_tokens, b_tokens) or _text.antonym_conflict(a_tokens, b_tokens)


def detect_and_resolve(
    profile: VoiceProfile,
    new_trait: VoiceTrait,
    *,
    contradicts: Callable[[str, str], bool] | None = None,
) -> list[VoiceTrait]:
    """Retire active same-category traits that contradict ``new_trait``.

    Among the profile's ``active`` traits sharing ``new_trait``'s category (and
    excluding ``new_trait`` itself), find those that contradict it using the
    injectable ``contradicts(a_rule, b_rule) -> bool`` predicate (default: a
    negation/antonym-cue heuristic). For each conflict the weaker side is retired
    by setting ``status='retired'``; strength is compared by
    ``(decay.effective_weight, observations, recency)``, highest wins. ``new_trait``
    is never retired when it is the stronger side. Returns the retired traits.
    """
    predicate = contradicts or _default_contradicts
    retired: list[VoiceTrait] = []

    for other in profile.traits:
        if other is new_trait:
            continue
        if other.category != new_trait.category or other.status != "active":
            continue
        if not predicate(new_trait.rule, other.rule):
            continue
        loser = _weaker(new_trait, other)
        loser.status = "retired"
        if loser is not new_trait:
            retired.append(loser)
    return retired


def _weaker(new_trait: VoiceTrait, other: VoiceTrait) -> VoiceTrait:
    """Return the weaker of two conflicting traits.

    Ranked by effective (decayed) weight, then observation count, then recency
    (``last_seen``). Ties resolve in favour of ``new_trait`` — the freshly
    reinforced signal — so ``other`` loses on an exact tie.
    """
    new_rank = (decay.effective_weight(new_trait), new_trait.observations, _last_seen_dt(new_trait))
    other_rank = (decay.effective_weight(other), other.observations, _last_seen_dt(other))
    return other if new_rank >= other_rank else new_trait


def _last_seen_dt(trait: VoiceTrait) -> datetime:
    """Parse ``trait.last_seen`` to an aware datetime for safe recency comparison.

    ``last_seen`` is a free-form string that may be hand-edited; lexical comparison
    of mixed naive/aware ISO timestamps mis-sorts. Parse it (assuming UTC for naive
    values), falling back to ``datetime.min`` (UTC) when the value is unparseable.
    """
    try:
        parsed = datetime.fromisoformat(trait.last_seen)
    except (ValueError, TypeError):
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed
