"""Semantic dedup — paraphrases collapse to one reinforced trait.

"keep it short" and "be concise" are the same rule; lexical Jaccard misses that.
This module matches a candidate rule against existing same-category, non-retired
traits by embedding cosine similarity, so reinforcement lands on the right trait
instead of spawning a near-duplicate.

Embeddings come from Ollama (``nomic-embed-text`` at ``$OLLAMA_BASE_URL``,
default ``http://localhost:11434``). There is no hard dependency on a running
server: when embeddings are unreachable this returns ``None`` and the caller
falls back to the lexical ``_find_similar``. It must never raise.

Stdlib only — embeddings are fetched with :mod:`urllib`, no new dependency and
no SDK client. The model name and endpoint are overridable via the environment
(``WORKSPEC_EMBED_MODEL``, ``OLLAMA_BASE_URL``) so tests and ops can retarget it.
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from workspec.profile import VoiceProfile, VoiceTrait

#: Embedding cosine similarity at or above which two rules are "the same".
#: Calibrated for nomic-embed-text with the ``clustering:`` task prefix below:
#: genuine paraphrases score ~0.76-0.84, unrelated rules ~0.62-0.70, so 0.74
#: separates them. Without the prefix nomic compresses the range (~0.5-0.6 for
#: paraphrases) and dedup never fires — both pieces matter together.
SIMILARITY_THRESHOLD = 0.74

#: nomic-embed-text requires a task prefix; "clustering:" is the right one for
#: semantic-similarity grouping and is what the threshold above is tuned against.
_EMBED_PREFIX = "clustering: "

#: Ollama embedding model; nomic-embed-text is small, fast, and widely available.
_DEFAULT_EMBED_MODEL = "nomic-embed-text"
#: Where Ollama lives; matches the convention used by the live test backends.
_DEFAULT_BASE_URL = "http://localhost:11434"
#: Network budget per embedding call, in seconds. Kept tight so an unreachable
#: server degrades to the lexical fallback quickly rather than stalling a draft.
_EMBED_TIMEOUT = 8.0


def _base_url() -> str:
    return os.environ.get("OLLAMA_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")


def _embed_model() -> str:
    return os.environ.get("WORKSPEC_EMBED_MODEL", _DEFAULT_EMBED_MODEL)


def _embed(text: str) -> list[float] | None:
    """Embed ``text`` via Ollama's ``/api/embeddings``; ``None`` on any failure.

    Never raises: a missing server, a missing model, a malformed response, or a
    timeout all collapse to ``None`` so :func:`semantic_match` can defer to the
    lexical fallback. Stdlib :mod:`urllib` only — no new dependency.
    """
    payload = json.dumps({"model": _embed_model(), "prompt": _EMBED_PREFIX + text}).encode("utf-8")
    request = urllib.request.Request(
        f"{_base_url()}/api/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=_EMBED_TIMEOUT) as response:
            data = json.load(response)
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None
    vector = data.get("embedding") if isinstance(data, dict) else None
    if not isinstance(vector, list) or not vector:
        return None
    try:
        return [float(value) for value in vector]
    except (TypeError, ValueError):
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors; ``0.0`` if degenerate."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def semantic_match(
    profile: VoiceProfile,
    rule: str,
    category: str,
    *,
    threshold: float = SIMILARITY_THRESHOLD,
) -> VoiceTrait | None:
    """Return an existing trait semantically equivalent to ``rule``, else ``None``.

    Considers only same-``category``, non-``retired`` traits, and returns the one
    whose rule has embedding cosine similarity ``>= threshold`` (best match).
    Returns ``None`` when embeddings are unreachable so the caller can fall back
    to lexical dedup. Never raises on a missing/unhealthy embedding server.

    Embeddings are fetched lazily and cached within this single call: the
    candidate ``rule`` is embedded once, and each comparison trait once, so a
    profile with many traits in the category still makes the minimum number of
    requests.
    """
    if not rule.strip():
        return None  # nothing to embed; let the caller's lexical path decide
    candidates = [
        trait
        for trait in profile.traits
        if trait.category == category and trait.status != "retired" and trait.rule.strip()
    ]
    if not candidates:
        return None

    query_vector = _embed(rule)
    if query_vector is None:
        return None  # embeddings unavailable -> defer to lexical fallback

    cache: dict[str, list[float] | None] = {}
    best_trait: VoiceTrait | None = None
    best_score = threshold
    for trait in candidates:
        if trait.rule not in cache:
            cache[trait.rule] = _embed(trait.rule)
        trait_vector = cache[trait.rule]
        if trait_vector is None:
            continue
        score = _cosine(query_vector, trait_vector)
        if score >= best_score:
            best_score = score
            best_trait = trait
    return best_trait
