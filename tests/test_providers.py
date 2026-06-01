"""Unit tests for provider construction and the structured-output adapters.

The SDK clients are mocked, so no network calls happen — we test the factory
routing, key resolution, and the parse/refusal/empty-result handling.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from workspec.models import Severity, Verdict
from workspec.providers import (
    AnthropicProvider,
    OpenAIProvider,
    build_provider,
)

_VERDICT = Verdict(passed=True, summary="ok", findings=[])


# --- factory --------------------------------------------------------------- #


def test_build_provider_anthropic(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    p = build_provider("anthropic", model="claude-haiku-4-5")
    assert isinstance(p, AnthropicProvider)
    assert p.name == "anthropic"
    assert p.model == "claude-haiku-4-5"


def test_build_provider_anthropic_uses_default_model(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert build_provider("anthropic").model == "claude-opus-4-8"


def test_build_provider_openai_and_aliases(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    for name in ("openai", "openai-compatible", "compatible"):
        p = build_provider(name, model="gpt-5.5")
        assert isinstance(p, OpenAIProvider)
        assert p.model == "gpt-5.5"


def test_build_provider_openai_default_model(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert build_provider("openai").model == "gpt-5.5"


def test_build_provider_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown provider"):
        build_provider("gemini")


def test_base_provider_get_structured_is_abstract() -> None:
    from workspec.providers import VerdictProvider

    class Bare(VerdictProvider):
        def get_structured(self, system_prompt, user_prompt, schema):
            return super().get_structured(system_prompt, user_prompt, schema)

    with pytest.raises(NotImplementedError):
        Bare().get_structured("s", "u", Verdict)


def test_anthropic_missing_key_raises(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # avoid a repo .env supplying the key during the test
    monkeypatch.setattr("workspec.providers.load_dotenv", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="No Anthropic API key"):
        AnthropicProvider()


def test_openai_missing_key_and_base_url_raises(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr("workspec.providers.load_dotenv", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="No OpenAI API key"):
        OpenAIProvider()


def test_openai_base_url_only_is_allowed(monkeypatch) -> None:
    """A local server (e.g. Ollama) needs no key, just a base_url."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("workspec.providers.load_dotenv", lambda *a, **k: None)
    p = OpenAIProvider(base_url="http://localhost:11434/v1", api_key="ollama")
    assert isinstance(p, OpenAIProvider)


# --- Anthropic.get_structured --------------------------------------------- #


def _anthropic(monkeypatch) -> AnthropicProvider:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return AnthropicProvider(model="claude-haiku-4-5")


def test_anthropic_get_structured_returns_parsed(monkeypatch) -> None:
    p = _anthropic(monkeypatch)
    p._client = MagicMock()
    p._client.messages.parse.return_value = SimpleNamespace(
        parsed_output=_VERDICT, stop_reason="end_turn"
    )
    out = p.get_structured("sys", "user", Verdict)
    assert out is _VERDICT
    p._client.messages.parse.assert_called_once()


def test_anthropic_get_structured_none_raises(monkeypatch) -> None:
    p = _anthropic(monkeypatch)
    p._client = MagicMock()
    p._client.messages.parse.return_value = SimpleNamespace(
        parsed_output=None, stop_reason="end_turn"
    )
    with pytest.raises(RuntimeError, match="no parseable Verdict"):
        p.get_structured("sys", "user", Verdict)


def test_anthropic_truncation_raises_clear_error(monkeypatch) -> None:
    p = _anthropic(monkeypatch)
    p._client = MagicMock()
    p._client.messages.parse.return_value = SimpleNamespace(
        parsed_output=None, stop_reason="max_tokens"
    )
    with pytest.raises(RuntimeError, match="truncated"):
        p.get_structured("sys", "user", Verdict)


def test_anthropic_passes_temperature_zero(monkeypatch) -> None:
    p = _anthropic(monkeypatch)
    assert p.temperature == 0.0
    p._client = MagicMock()
    p._client.messages.parse.return_value = SimpleNamespace(
        parsed_output=_VERDICT, stop_reason="end_turn"
    )
    p.get_structured("sys", "user", Verdict)
    _, kwargs = p._client.messages.parse.call_args
    assert kwargs["temperature"] == 0.0


def test_anthropic_caches_system_prompt(monkeypatch) -> None:
    p = _anthropic(monkeypatch)
    p._client = MagicMock()
    p._client.messages.parse.return_value = SimpleNamespace(
        parsed_output=_VERDICT, stop_reason="end_turn"
    )
    p.get_structured("big system prompt", "user", Verdict)
    _, kwargs = p._client.messages.parse.call_args
    system = kwargs["system"]
    assert isinstance(system, list)
    assert system[0]["text"] == "big system prompt"
    assert system[0]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_client_gets_retries_and_timeout(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    import anthropic

    captured: dict[str, object] = {}

    def fake_anthropic(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(anthropic, "Anthropic", fake_anthropic)
    AnthropicProvider(model="claude-haiku-4-5", max_retries=5, timeout=30.0)
    assert captured["max_retries"] == 5
    assert captured["timeout"] == 30.0


def test_anthropic_model_not_found_message(monkeypatch) -> None:
    p = _anthropic(monkeypatch)
    p._client = MagicMock()

    class _NotFound(Exception):
        status_code = 404

    p._client.messages.parse.side_effect = _NotFound("model: not_found_error")
    with pytest.raises(RuntimeError, match="not found for provider 'anthropic'"):
        p.get_structured("sys", "user", Verdict)


def test_anthropic_unrelated_error_propagates(monkeypatch) -> None:
    p = _anthropic(monkeypatch)
    p._client = MagicMock()
    p._client.messages.parse.side_effect = ValueError("boom")
    with pytest.raises(ValueError, match="boom"):
        p.get_structured("sys", "user", Verdict)


def test_anthropic_runtimeerror_propagates_unwrapped(monkeypatch) -> None:
    # A RuntimeError raised inside the call must bubble up unchanged, not be
    # re-wrapped by the model-not-found helper.
    p = _anthropic(monkeypatch)
    p._client = MagicMock()
    p._client.messages.parse.side_effect = RuntimeError("upstream boom")
    with pytest.raises(RuntimeError, match="upstream boom"):
        p.get_structured("sys", "user", Verdict)


def test_get_verdict_wrapper_delegates(monkeypatch) -> None:
    p = _anthropic(monkeypatch)
    p._client = MagicMock()
    p._client.messages.parse.return_value = SimpleNamespace(
        parsed_output=_VERDICT, stop_reason="end_turn"
    )
    verdict = p.get_verdict("sys", "user")
    assert isinstance(verdict, Verdict)
    # the wrapper must request the Verdict schema
    _, kwargs = p._client.messages.parse.call_args
    assert kwargs["output_format"] is Verdict


# --- OpenAI.get_structured ------------------------------------------------ #


def _openai(monkeypatch) -> OpenAIProvider:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return OpenAIProvider(model="gpt-5.5")


def _completion(*, parsed=None, refusal=None, finish_reason="stop") -> SimpleNamespace:
    message = SimpleNamespace(parsed=parsed, refusal=refusal)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


def test_openai_get_structured_returns_parsed(monkeypatch) -> None:
    p = _openai(monkeypatch)
    p._client = MagicMock()
    p._client.chat.completions.parse.return_value = _completion(parsed=_VERDICT)
    assert p.get_structured("sys", "user", Verdict) is _VERDICT


def test_openai_get_structured_refusal_raises(monkeypatch) -> None:
    p = _openai(monkeypatch)
    p._client = MagicMock()
    p._client.chat.completions.parse.return_value = _completion(refusal="cannot help")
    with pytest.raises(RuntimeError, match="refused"):
        p.get_structured("sys", "user", Verdict)


def test_openai_get_structured_none_raises(monkeypatch) -> None:
    p = _openai(monkeypatch)
    p._client = MagicMock()
    p._client.chat.completions.parse.return_value = _completion(parsed=None, finish_reason="stop")
    with pytest.raises(RuntimeError, match="no parseable Verdict"):
        p.get_structured("sys", "user", Verdict)


def test_openai_truncation_raises_clear_error(monkeypatch) -> None:
    p = _openai(monkeypatch)
    p._client = MagicMock()
    p._client.chat.completions.parse.return_value = _completion(parsed=None, finish_reason="length")
    with pytest.raises(RuntimeError, match="truncated"):
        p.get_structured("sys", "user", Verdict)


def test_openai_passes_temperature_zero(monkeypatch) -> None:
    p = _openai(monkeypatch)
    assert p.temperature == 0.0
    p._client = MagicMock()
    p._client.chat.completions.parse.return_value = _completion(parsed=_VERDICT)
    p.get_structured("sys", "user", Verdict)
    _, kwargs = p._client.chat.completions.parse.call_args
    assert kwargs["temperature"] == 0.0


def test_openai_client_gets_retries_and_timeout(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    import openai

    captured: dict[str, object] = {}

    def fake_openai(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(openai, "OpenAI", fake_openai)
    OpenAIProvider(model="gpt-5.5", max_retries=4, timeout=20.0)
    assert captured["max_retries"] == 4
    assert captured["timeout"] == 20.0


def test_openai_model_not_found_message(monkeypatch) -> None:
    p = _openai(monkeypatch)
    p._client = MagicMock()

    class _NotFound(Exception):
        status_code = 404

    p._client.chat.completions.parse.side_effect = _NotFound("The model `x` does not exist")
    with pytest.raises(RuntimeError, match="not found for provider 'openai'"):
        p.get_structured("sys", "user", Verdict)


def test_openai_unrelated_error_propagates(monkeypatch) -> None:
    p = _openai(monkeypatch)
    p._client = MagicMock()
    p._client.chat.completions.parse.side_effect = ValueError("boom")
    with pytest.raises(ValueError, match="boom"):
        p.get_structured("sys", "user", Verdict)


def test_openai_runtimeerror_propagates_unwrapped(monkeypatch) -> None:
    p = _openai(monkeypatch)
    p._client = MagicMock()
    p._client.chat.completions.parse.side_effect = RuntimeError("upstream boom")
    with pytest.raises(RuntimeError, match="upstream boom"):
        p.get_structured("sys", "user", Verdict)


def test_severity_enum_imported() -> None:
    # guard against accidental import breakage used across the suite
    assert Severity.BLOCKER
