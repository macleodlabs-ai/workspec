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
spuriously retiring an unrelated trait.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from workspec.learning import decay

if TYPE_CHECKING:
    from collections.abc import Callable

    from workspec.profile import VoiceProfile, VoiceTrait

#: Tokens that flip the polarity of a following instruction.
_NEGATION_CUES: frozenset[str] = frozenset(
    {"no", "not", "never", "dont", "don't", "avoid", "stop", "without", "no longer"}
)

#: Symmetric antonym pairs. A rule on one side contradicts a rule on the other.
_ANTONYM_PAIRS: tuple[tuple[str, str], ...] = (
    ("warm", "cold"),
    ("warm", "cool"),
    ("warm", "terse"),
    ("friendly", "terse"),
    ("short", "long"),
    ("brief", "verbose"),
    ("concise", "verbose"),
    ("concise", "detailed"),
    ("formal", "casual"),
    ("formal", "informal"),
    ("bullets", "prose"),
    ("bulleted", "prose"),
    ("bullet", "paragraph"),
    ("more", "less"),
    ("add", "remove"),
    ("include", "omit"),
    ("include", "exclude"),
    ("open", "close"),
    ("polite", "blunt"),
    ("soft", "direct"),
)

_WORD_RE = re.compile(r"[a-z']+")


def _tokens(rule: str) -> list[str]:
    """Lowercased word tokens of ``rule`` (apostrophes kept for don't/can't)."""
    return _WORD_RE.findall(rule.lower())


def _is_negated(tokens: list[str], index: int) -> bool:
    """Whether the token at ``index`` is preceded by a nearby negation cue."""
    start = max(0, index - 3)
    return any(tokens[i] in _NEGATION_CUES for i in range(start, index))


def _negation_conflict(a_tokens: list[str], b_tokens: list[str]) -> bool:
    """True if a salient shared token is asserted on one side, negated on other.

    "Use bullet points" vs "Do not use bullet points" share ``bullet``/``points``;
    one side negates it while the other does not, so they conflict.
    """
    a_set = set(a_tokens)
    b_set = set(b_tokens)
    shared = (a_set & b_set) - _NEGATION_CUES
    for token in shared:
        if len(token) < 3:
            continue  # skip stopword-ish glue ("be", "it", "to")
        a_neg = any(_is_negated(a_tokens, i) for i, t in enumerate(a_tokens) if t == token)
        b_neg = any(_is_negated(b_tokens, i) for i, t in enumerate(b_tokens) if t == token)
        if a_neg != b_neg:
            return True
    return False


def _antonym_conflict(a_tokens: list[str], b_tokens: list[str]) -> bool:
    """True if the rules sit on opposite ends of a known antonym pair.

    Either ordering counts (the pairs are treated symmetrically); a side carrying
    a negation flips its polarity, so "be warm" vs "do not be cold" do *not*
    conflict (both end up wanting warmth).
    """
    a_set = set(a_tokens)
    b_set = set(b_tokens)
    for left, right in _ANTONYM_PAIRS:
        for x, y in ((left, right), (right, left)):
            if x in a_set and y in b_set:
                a_neg = _is_negated(a_tokens, a_tokens.index(x))
                b_neg = _is_negated(b_tokens, b_tokens.index(y))
                if a_neg == b_neg:
                    return True
    return False


def _default_contradicts(a_rule: str, b_rule: str) -> bool:
    """Lightweight negation/antonym-cue contradiction heuristic."""
    a_tokens = _tokens(a_rule)
    b_tokens = _tokens(b_rule)
    if not a_tokens or not b_tokens:
        return False
    return _negation_conflict(a_tokens, b_tokens) or _antonym_conflict(a_tokens, b_tokens)


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
    new_rank = (decay.effective_weight(new_trait), new_trait.observations, new_trait.last_seen)
    other_rank = (decay.effective_weight(other), other.observations, other.last_seen)
    return other if new_rank >= other_rank else new_trait
