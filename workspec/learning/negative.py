"""Negative signal — penalize traits that informed a draft but were edited out.

The learning loop is only sound if it can be told it was wrong. When a trait
that shaped a draft has its guidance reversed in what the person actually sent,
that trait is penalized (weight and observations down) and retired if it falls
below a floor. This closes the loop the positive path opened.

A reversal is inferred without a model: a trait is penalized when either

  * the injectable ``contradicts(rule, edit_description)`` predicate fires
    against the change the person made (default: a negation/antonym-cue
    heuristic, same spirit as :mod:`workspec.learning.contradiction`), or
  * the trait's *signature tokens* — distinctive words from its rule that were
    present in the draft — were stripped out in the sent text, i.e. the draft
    followed the trait and the person undid it.

Both checks are pure functions of the inputs, so the penalty is deterministic.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from workspec.profile import VoiceProfile, VoiceTrait

#: Amount a penalized trait's weight is decremented per negative signal.
NEGATIVE_DECREMENT = 0.15
#: Weight below which a penalized trait is retired.
RETIRE_FLOOR = 0.2

# Negation cues that flip a rule's polarity (mirrors the contradiction heuristic).
_NEGATIONS = frozenset(
    {"no", "not", "never", "don't", "dont", "avoid", "without", "stop", "drop", "remove"}
)

# Common antonym pairs whose presence on opposite sides signals a reversal.
_ANTONYMS: frozenset[frozenset[str]] = frozenset(
    {
        frozenset({"warm", "cold"}),
        frozenset({"warm", "terse"}),
        frozenset({"formal", "casual"}),
        frozenset({"formal", "informal"}),
        frozenset({"long", "short"}),
        frozenset({"long", "brief"}),
        frozenset({"verbose", "concise"}),
        frozenset({"verbose", "terse"}),
        frozenset({"add", "remove"}),
        frozenset({"include", "omit"}),
        frozenset({"open", "close"}),
        frozenset({"start", "end"}),
    }
)

# Words too common to count as a trait's distinctive signature.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "do",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "so",
        "than",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "use",
        "using",
        "with",
        "you",
        "your",
        "keep",
        "make",
        "more",
        "less",
        "very",
        "when",
        "always",
        "every",
    }
)


def _tokens(text: str) -> set[str]:
    """Lowercased word tokens, stripped of surrounding punctuation."""
    return {w for w in re.findall(r"[a-z']+", text.lower()) if w}


def _signature_tokens(rule: str) -> set[str]:
    """Distinctive content words of ``rule`` (drop stopwords/negations/short)."""
    return {t for t in _tokens(rule) if len(t) > 2 and t not in _STOPWORDS and t not in _NEGATIONS}


def _default_contradicts(rule: str, change: str) -> bool:
    """Heuristic ``contradicts`` for a rule vs a description of what changed.

    Fires when the change carries a negation cue against the rule's signature
    (e.g. rule "open warmly" and the change strips the warm opener), or when an
    antonym of a rule word shows up in the change. Mirrors the
    :mod:`workspec.learning.contradiction` cue style.
    """
    rule_sig = _signature_tokens(rule)
    if not rule_sig:
        return False
    change_tokens = _tokens(change)
    # Negation cue in the change that targets a rule signature word.
    if change_tokens & _NEGATIONS and (rule_sig & change_tokens):
        return True
    # Antonym pair split across rule and change.
    for pair in _ANTONYMS:
        a, b = tuple(pair)
        if (a in rule_sig and b in change_tokens) or (b in rule_sig and a in change_tokens):
            return True
    return False


def _was_reversed(
    trait: VoiceTrait,
    draft_tokens: set[str],
    sent_tokens: set[str],
    removed_tokens: set[str],
    changed_tokens: set[str],
    contradicts: Callable[[str, str], bool],
) -> bool:
    """True when ``trait``'s guidance appears undone in the draft→sent edit.

    Two independent signals, either of which counts as a reversal:

    1. **Contradiction** — the *change* (text the person removed from, or added
       to, the draft) contradicts the rule under the ``contradicts`` predicate.
       This catches both "took the rule's wording back out" and "introduced
       opposing language" (e.g. swapping a long passage for a short one).
    2. **Signature stripping** — the rule's distinctive tokens were present in
       the draft and are gone from the sent text, i.e. the trait shaped the
       draft and the edit removed exactly that shaping.
    """
    sig = _signature_tokens(trait.rule)
    if not sig:
        return False

    # 1. The change (what was removed and/or added) contradicts the rule.
    change_text = " ".join(sorted(changed_tokens))
    if changed_tokens and contradicts(trait.rule, change_text):
        return True

    # 2. The rule's signature was honored in the draft, then stripped out.
    in_draft = sig & draft_tokens
    return bool(in_draft and not (sig & sent_tokens) and (in_draft & removed_tokens))


def apply_negative_signal(
    profile: VoiceProfile,
    applied_keys: list[str],
    draft: str,
    sent: str,
    *,
    contradicts: Callable[[str, str], bool] | None = None,
) -> list[VoiceTrait]:
    """Penalize applied traits whose guidance was reversed between draft and sent.

    For each trait whose ``.key`` is in ``applied_keys`` and whose guidance was
    reversed in ``sent`` vs ``draft`` (via the injectable ``contradicts``
    predicate or the signature-stripping heuristic), decrement ``weight`` by
    ``NEGATIVE_DECREMENT`` and ``observations`` by 1 (floored at 0); retire it
    (``status='retired'``) if ``weight < RETIRE_FLOOR``. Returns the affected
    traits in profile order.

    Pure and deterministic given its inputs: it inspects only the token-level
    difference between ``draft`` and ``sent``.
    """
    if not applied_keys:
        return []

    predicate = contradicts or _default_contradicts
    keys = set(applied_keys)

    draft_tokens = _tokens(draft)
    sent_tokens = _tokens(sent)
    removed_tokens = draft_tokens - sent_tokens
    added_tokens = sent_tokens - draft_tokens
    # The full "edit" the contradicts predicate reasons about: what the person
    # took out plus what they put in its place.
    changed_tokens = removed_tokens | added_tokens

    affected: list[VoiceTrait] = []
    for trait in profile.traits:
        if trait.key not in keys or trait.status == "retired":
            continue
        if not _was_reversed(
            trait, draft_tokens, sent_tokens, removed_tokens, changed_tokens, predicate
        ):
            continue
        trait.weight = max(0.0, trait.weight - NEGATIVE_DECREMENT)
        trait.observations = max(0, trait.observations - 1)
        if trait.weight < RETIRE_FLOOR:
            trait.status = "retired"
        affected.append(trait)

    return affected
