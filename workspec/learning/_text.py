"""Shared lexical heuristics for the contradiction & negative-signal detectors.

Both modules ask the same low-level questions — what are a rule's meaningful
tokens, is a token negated, do two texts sit on opposite ends of an antonym
pair — so the tokenizer, the negation-cue set, the antonym table, the stopword
list, and the two conflict primitives live here once. The *higher-level*
predicates differ (rule-vs-rule for contradiction, rule-vs-edit for negative
signal) and stay in their own modules; only these primitives are shared.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[a-z']+")

#: Tokens that flip the polarity of a nearby instruction.
NEGATIONS: frozenset[str] = frozenset(
    {"no", "not", "never", "dont", "don't", "avoid", "stop", "without", "drop", "remove"}
)

#: Symmetric antonym pairs — a rule on one side opposes a rule on the other.
ANTONYMS: tuple[frozenset[str], ...] = (
    frozenset({"warm", "cold"}),
    frozenset({"warm", "cool"}),
    frozenset({"warm", "terse"}),
    frozenset({"friendly", "terse"}),
    frozenset({"short", "long"}),
    frozenset({"brief", "verbose"}),
    frozenset({"concise", "verbose"}),
    frozenset({"concise", "detailed"}),
    frozenset({"verbose", "terse"}),
    frozenset({"formal", "casual"}),
    frozenset({"formal", "informal"}),
    frozenset({"bullets", "prose"}),
    frozenset({"bulleted", "prose"}),
    frozenset({"bullet", "paragraph"}),
    frozenset({"add", "remove"}),
    frozenset({"include", "omit"}),
    frozenset({"include", "exclude"}),
    frozenset({"open", "close"}),
    frozenset({"start", "end"}),
    frozenset({"polite", "blunt"}),
    frozenset({"soft", "direct"}),
)

#: Words too common to count as a rule's distinctive signature.
STOPWORDS: frozenset[str] = frozenset(
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


def tokens(text: str) -> list[str]:
    """Lowercased word tokens (apostrophes kept for don't/can't)."""
    return _WORD_RE.findall(text.lower())


def token_set(text: str) -> set[str]:
    """Set of lowercased word tokens."""
    return set(_WORD_RE.findall(text.lower()))


def signature_tokens(rule: str) -> set[str]:
    """Distinctive content words of ``rule`` (drop stopwords/negations/short)."""
    return {t for t in tokens(rule) if len(t) > 2 and t not in STOPWORDS and t not in NEGATIONS}


#: How many tokens back a negation cue still flips a later token's polarity.
NEGATION_WINDOW = 3


def is_negated(toks: list[str], index: int) -> bool:
    """Whether the token at ``index`` is preceded by a nearby negation cue."""
    start = max(0, index - NEGATION_WINDOW)
    return any(toks[i] in NEGATIONS for i in range(start, index))


def negation_conflict(a_tokens: list[str], b_tokens: list[str]) -> bool:
    """True if a salient shared token is asserted on one side, negated on the other.

    "Use bullet points" vs "Do not use bullet points" share ``bullet``/``points``;
    one side negates it while the other does not, so they conflict.
    """
    shared = (set(a_tokens) & set(b_tokens)) - NEGATIONS
    for token in shared:
        if len(token) < 3:
            continue  # skip stopword-ish glue ("be", "it", "to")
        a_neg = any(is_negated(a_tokens, i) for i, t in enumerate(a_tokens) if t == token)
        b_neg = any(is_negated(b_tokens, i) for i, t in enumerate(b_tokens) if t == token)
        if a_neg != b_neg:
            return True
    return False


def antonym_conflict(a_tokens: list[str], b_tokens: list[str]) -> bool:
    """True if the rules sit on opposite ends of a known antonym pair.

    Either ordering counts (pairs are symmetric); a side carrying a negation flips
    its polarity, so "be warm" vs "do not be cold" do *not* conflict (both want
    warmth).
    """
    a_set = set(a_tokens)
    b_set = set(b_tokens)
    for pair in ANTONYMS:
        left, right = tuple(pair)
        for x, y in ((left, right), (right, left)):
            if x in a_set and y in b_set:
                a_neg = any(is_negated(a_tokens, i) for i, t in enumerate(a_tokens) if t == x)
                b_neg = any(is_negated(b_tokens, i) for i, t in enumerate(b_tokens) if t == y)
                if a_neg == b_neg:
                    return True
    return False
