"""Unit tests for ``workspec.learning.semantic`` — embedding-based dedup.

The deterministic tests monkeypatch ``semantic._embed`` so they exercise the
matching / cosine / fallback logic without any network. One live test hits a
real Ollama ``nomic-embed-text`` server and is skipped when it is unreachable,
mirroring the skip style in ``tests/test_integration_backends.py``.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from workspec.learning import semantic
from workspec.profile import VoiceProfile, VoiceTrait

# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


def _trait(rule: str, category: str = "length", status: str = "active") -> VoiceTrait:
    return VoiceTrait(category=category, rule=rule, status=status)  # type: ignore[arg-type]


def _fake_embeddings(mapping: dict[str, list[float]]):
    """Build a deterministic ``_embed`` substitute from a text -> vector map.

    Unknown text returns ``None`` (as a real server would for an empty result).
    """

    def fake(text: str) -> list[float] | None:
        return mapping.get(text)

    return fake


# --------------------------------------------------------------------------- #
# Cosine math
# --------------------------------------------------------------------------- #


def test_cosine_identical_is_one() -> None:
    assert semantic._cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_orthogonal_is_zero() -> None:
    assert semantic._cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_opposite_is_negative_one() -> None:
    assert semantic._cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_zero_vector_is_zero() -> None:
    assert semantic._cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_length_mismatch_is_zero() -> None:
    assert semantic._cosine([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0


# --------------------------------------------------------------------------- #
# semantic_match — matching behavior (embedding injected)
# --------------------------------------------------------------------------- #


def test_match_blank_rule_short_circuits(monkeypatch: pytest.MonkeyPatch) -> None:
    """A blank candidate rule returns None without ever embedding (no point)."""

    def boom(_text: str) -> list[float] | None:  # pragma: no cover - must not run
        raise AssertionError("_embed should not be called for a blank rule")

    monkeypatch.setattr(semantic, "_embed", boom)
    profile = VoiceProfile(traits=[_trait("Be concise.")])
    assert semantic.semantic_match(profile, "   ", "length") is None


def test_match_returns_paraphrase_above_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    profile = VoiceProfile(traits=[_trait("Be concise.")])
    # Near-parallel vectors -> cosine ~0.9986, comfortably over the 0.74 default.
    monkeypatch.setattr(
        semantic,
        "_embed",
        _fake_embeddings({"keep it short": [1.0, 0.0, 0.0], "Be concise.": [0.95, 0.05, 0.0]}),
    )
    match = semantic.semantic_match(profile, "keep it short", "length")
    assert match is not None
    assert match.rule == "Be concise."


def test_no_match_when_below_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    profile = VoiceProfile(traits=[_trait("Be concise.")])
    # Orthogonal -> cosine 0.0, well under threshold.
    monkeypatch.setattr(
        semantic,
        "_embed",
        _fake_embeddings({"be warm and chatty": [0.0, 1.0], "Be concise.": [1.0, 0.0]}),
    )
    assert semantic.semantic_match(profile, "be warm and chatty", "length") is None


def test_picks_highest_similarity_among_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    profile = VoiceProfile(
        traits=[
            _trait("Be concise."),
            _trait("Keep replies short and to the point."),
        ]
    )
    monkeypatch.setattr(
        semantic,
        "_embed",
        _fake_embeddings(
            {
                "keep it short": [1.0, 0.0],
                "Be concise.": [0.9, 0.44],  # cosine ~0.898
                "Keep replies short and to the point.": [0.99, 0.14],  # cosine ~0.990
            }
        ),
    )
    match = semantic.semantic_match(profile, "keep it short", "length")
    assert match is not None
    assert match.rule == "Keep replies short and to the point."


def test_respects_custom_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    profile = VoiceProfile(traits=[_trait("Be concise.")])
    monkeypatch.setattr(
        semantic,
        "_embed",
        _fake_embeddings({"keep it short": [1.0, 0.0], "Be concise.": [0.9, 0.44]}),  # ~0.898
    )
    # Below a strict threshold -> no match.
    assert semantic.semantic_match(profile, "keep it short", "length", threshold=0.95) is None
    # Above a loose threshold -> match.
    assert semantic.semantic_match(profile, "keep it short", "length", threshold=0.5) is not None


# --------------------------------------------------------------------------- #
# semantic_match — filtering candidates
# --------------------------------------------------------------------------- #


def test_ignores_other_categories(monkeypatch: pytest.MonkeyPatch) -> None:
    profile = VoiceProfile(traits=[_trait("Be concise.", category="tone")])
    monkeypatch.setattr(
        semantic,
        "_embed",
        _fake_embeddings({"keep it short": [1.0, 0.0], "Be concise.": [1.0, 0.0]}),
    )
    # Identical vectors, but wrong category -> not a candidate at all.
    assert semantic.semantic_match(profile, "keep it short", "length") is None


def test_ignores_retired_traits(monkeypatch: pytest.MonkeyPatch) -> None:
    profile = VoiceProfile(traits=[_trait("Be concise.", status="retired")])
    monkeypatch.setattr(
        semantic,
        "_embed",
        _fake_embeddings({"keep it short": [1.0, 0.0], "Be concise.": [1.0, 0.0]}),
    )
    assert semantic.semantic_match(profile, "keep it short", "length") is None


def test_considers_provisional_traits(monkeypatch: pytest.MonkeyPatch) -> None:
    profile = VoiceProfile(traits=[_trait("Be concise.", status="provisional")])
    monkeypatch.setattr(
        semantic,
        "_embed",
        _fake_embeddings({"keep it short": [1.0, 0.0], "Be concise.": [1.0, 0.0]}),
    )
    match = semantic.semantic_match(profile, "keep it short", "length")
    assert match is not None and match.rule == "Be concise."


def test_no_candidates_skips_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    profile = VoiceProfile(traits=[])

    def boom(_text: str) -> list[float] | None:  # pragma: no cover - must not run
        raise AssertionError("_embed must not be called when there are no candidates")

    monkeypatch.setattr(semantic, "_embed", boom)
    assert semantic.semantic_match(profile, "keep it short", "length") is None


def test_embeds_each_rule_once(monkeypatch: pytest.MonkeyPatch) -> None:
    # Same rule text appears twice (e.g. duplicate provisional rows) -> embedded once.
    profile = VoiceProfile(traits=[_trait("Be concise."), _trait("Be concise.")])
    calls: list[str] = []

    def counting(text: str) -> list[float] | None:
        calls.append(text)
        return {"keep it short": [1.0, 0.0], "Be concise.": [0.99, 0.1]}.get(text)

    monkeypatch.setattr(semantic, "_embed", counting)
    assert semantic.semantic_match(profile, "keep it short", "length") is not None
    # One call for the query, one for the (deduped) trait rule.
    assert calls.count("keep it short") == 1
    assert calls.count("Be concise.") == 1


# --------------------------------------------------------------------------- #
# semantic_match — graceful degradation (never raises)
# --------------------------------------------------------------------------- #


def test_returns_none_when_query_embedding_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    profile = VoiceProfile(traits=[_trait("Be concise.")])
    monkeypatch.setattr(semantic, "_embed", lambda _text: None)
    assert semantic.semantic_match(profile, "keep it short", "length") is None


def test_skips_traits_whose_embedding_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    profile = VoiceProfile(
        traits=[_trait("Be concise."), _trait("Keep replies short.")],
    )
    # Query embeds; first trait fails to embed (skipped), second matches.
    monkeypatch.setattr(
        semantic,
        "_embed",
        _fake_embeddings(
            {
                "keep it short": [1.0, 0.0],
                # "Be concise." absent -> None -> skipped
                "Keep replies short.": [0.99, 0.1],
            }
        ),
    )
    match = semantic.semantic_match(profile, "keep it short", "length")
    assert match is not None and match.rule == "Keep replies short."


def test_embed_swallows_network_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_urlerror(*_args: object, **_kwargs: object):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", raise_urlerror)
    assert semantic._embed("anything") is None


def test_embed_handles_malformed_json(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Resp:
        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

        def read(self) -> bytes:
            return b"not json"

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp())
    assert semantic._embed("anything") is None


def test_embed_handles_missing_embedding_key(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Resp:
        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"error": "model not found"}).encode("utf-8")

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp())
    assert semantic._embed("anything") is None


def test_embed_handles_non_numeric_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Resp:
        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"embedding": ["a", "b"]}).encode("utf-8")

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp())
    assert semantic._embed("anything") is None


# --------------------------------------------------------------------------- #
# Environment overrides
# --------------------------------------------------------------------------- #


def test_base_url_honors_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://example.test:9999/")
    assert semantic._base_url() == "http://example.test:9999"  # trailing slash stripped


def test_base_url_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    assert semantic._base_url() == "http://localhost:11434"


def test_embed_model_honors_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORKSPEC_EMBED_MODEL", "mxbai-embed-large")
    assert semantic._embed_model() == "mxbai-embed-large"


# --------------------------------------------------------------------------- #
# Live path — real Ollama embeddings (skipped when unreachable)
# --------------------------------------------------------------------------- #


def _nomic_available() -> bool:
    """True when an Ollama server with nomic-embed-text answers an embed call."""
    try:
        vector = semantic._embed("connectivity probe")
    except Exception:  # pragma: no cover - defensive; _embed already swallows
        return False
    return bool(vector)


@pytest.mark.integration
@pytest.mark.skipif(not _nomic_available(), reason="Ollama nomic-embed-text unreachable")
def test_semantic_match_live_prefers_related_trait() -> None:
    """Real embeddings: a near-paraphrase scores higher than an unrelated rule.

    With the ``clustering:`` task prefix, nomic-embed-text places genuine
    paraphrases above the 0.74 default and unrelated rules below it. This exercises
    the real network path and the ranking guarantee (best candidate wins); we pass
    a ``threshold`` derived from the observed score so the assertion is robust to
    minor model drift, and the unrelated rule must still rank strictly below.
    """
    related = _trait("Keep replies short and to the point.", category="length")
    unrelated = _trait("Always cc the legal team on every reply.", category="length")
    profile = VoiceProfile(traits=[related, unrelated])

    query = "Be concise and brief."
    query_vec = semantic._embed(query)
    assert query_vec is not None
    related_vec = semantic._embed(related.rule)
    unrelated_vec = semantic._embed(unrelated.rule)
    assert related_vec is not None and unrelated_vec is not None

    related_score = semantic._cosine(query_vec, related_vec)
    unrelated_score = semantic._cosine(query_vec, unrelated_vec)
    # The conciseness paraphrase is closer than the unrelated cc rule.
    assert related_score > unrelated_score

    # At a threshold the embeddings support, the best match is the related trait.
    match = semantic.semantic_match(profile, query, "length", threshold=related_score - 0.01)
    assert match is not None
    assert match.rule == related.rule


@pytest.mark.integration
@pytest.mark.skipif(not _nomic_available(), reason="Ollama nomic-embed-text unreachable")
def test_semantic_match_live_unrelated_returns_none_at_default() -> None:
    """Real embeddings: an unrelated rule does not collapse onto an existing trait.

    Uses the real ``SIMILARITY_THRESHOLD`` default; the unrelated rule's cosine
    is far below it, so no match is returned.
    """
    profile = VoiceProfile(
        traits=[_trait("Keep replies short and to the point.", category="length")]
    )
    assert semantic.semantic_match(profile, "Always cc the legal team.", "length") is None
