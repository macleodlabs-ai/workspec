"""LLM provider backends for WorkSpec.

Changes: load repo ``.env`` before resolving API keys (see ``workspec.env.load_dotenv``).

A provider's only job: take a system prompt + user prompt and return a validated
``Verdict``, using its SDK's native structured-output support. Each backend is a
single-purpose adapter; the engine picks one and never touches an SDK directly.

Two backends ship:

  * ``AnthropicProvider`` — uses ``messages.parse(output_format=...)`` (GA
    structured outputs), result on ``.parsed_output``.
  * ``OpenAIProvider`` — uses ``chat.completions.parse(response_format=...)``,
    result on ``.choices[0].message.parsed``. Because the OpenAI SDK takes a
    ``base_url``, this one backend also covers any OpenAI-compatible endpoint:
    Azure OpenAI, OpenRouter, Together, Groq, vLLM, Ollama, LM Studio, etc.

Both rely on the SDKs' native Pydantic support, so there is no prose parsing in
either path.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

from workspec.env import load_dotenv
from workspec.models import Verdict

T = TypeVar("T", bound=BaseModel)


class VerdictProvider(ABC):
    """Interface every backend implements.

    The core method is ``get_structured``, which returns any Pydantic model the
    caller asks for (Verdict for linting, Draft/LearnedTraits for drafting).
    ``get_verdict`` is a thin convenience wrapper kept for the lint engine.
    """

    #: Human-readable backend name, e.g. "anthropic" or "openai".
    name: str = "base"

    @abstractmethod
    def get_structured(self, system_prompt: str, user_prompt: str, schema: type[T]) -> T:
        """Return a validated instance of ``schema`` for the given prompts."""
        raise NotImplementedError

    def get_verdict(self, system_prompt: str, user_prompt: str) -> Verdict:
        """Convenience wrapper used by the lint engine."""
        return self.get_structured(system_prompt, user_prompt, Verdict)


def _raise_if_model_not_found(exc: Exception, model: str, provider: str) -> None:
    """Re-raise a clearer error if ``exc`` looks like a model-not-found failure.

    Both Anthropic and OpenAI surface an unknown model as a 404 / NotFoundError
    mentioning "model". We detect that broadly (by message, not exact type) and
    raise an actionable ``RuntimeError``. Any other exception is re-raised as-is.
    """
    message = str(exc).lower()
    status = getattr(exc, "status_code", None)
    looks_not_found = status == 404 or "not_found" in message or "404" in message
    if looks_not_found and "model" in message:
        raise RuntimeError(
            f"Model '{model}' not found for provider '{provider}'. "
            "Pass a valid --model (see README)."
        ) from exc
    raise exc


class AnthropicProvider(VerdictProvider):
    """Anthropic backend using GA structured outputs.

    Parameters
    ----------
    model:
        Anthropic model id, e.g. ``claude-opus-4-8`` or ``claude-haiku-4-5``.
    api_key:
        Falls back to ``ANTHROPIC_API_KEY``.
    max_tokens:
        Output budget for the verdict.
    temperature:
        Sampling temperature. Defaults to ``0.0`` for deterministic verdicts.
    max_retries:
        SDK-level retry count for transient failures.
    timeout:
        Per-request timeout in seconds, passed to the SDK client.
    """

    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-opus-4-8",
        api_key: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        max_retries: int = 2,
        timeout: float = 60.0,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "The 'anthropic' package is required for the Anthropic provider. "
                "It ships with WorkSpec by default — reinstall with: uv pip install -e ."
            ) from exc

        load_dotenv()
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "No Anthropic API key found. Set ANTHROPIC_API_KEY (export it in "
                "your shell), add it to a .env file in the project root, or pass "
                "api_key= to the provider."
            )
        self._client = anthropic.Anthropic(api_key=key, max_retries=max_retries, timeout=timeout)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def get_structured(self, system_prompt: str, user_prompt: str, schema: type[T]) -> T:
        try:
            response = self._client.messages.parse(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                # Mark the large static system prompt as an ephemeral cache block
                # so it is cached across calls (prompt caching).
                system=[
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_prompt}],
                output_format=schema,
            )
        except RuntimeError:
            raise
        except Exception as exc:
            _raise_if_model_not_found(exc, self.model, self.name)
            raise  # pragma: no cover - unreachable: helper always raises, keeps names bound

        parsed = response.parsed_output
        if parsed is None:
            if response.stop_reason == "max_tokens":
                raise RuntimeError(
                    f"Anthropic output was truncated before a complete {schema.__name__} "
                    "could be produced (stop_reason=max_tokens). Raise --max-tokens and retry."
                )
            raise RuntimeError(
                f"Anthropic returned no parseable {schema.__name__} "
                f"(stop_reason={response.stop_reason})."
            )
        return parsed


class OpenAIProvider(VerdictProvider):
    """OpenAI / OpenAI-compatible backend using ``chat.completions.parse``.

    Set ``base_url`` to target any OpenAI-compatible server. Examples::

        # OpenAI
        OpenAIProvider(model="gpt-5.5")

        # OpenRouter
        OpenAIProvider(model="openai/gpt-5.5",
                       base_url="https://openrouter.ai/api/v1",
                       api_key=os.environ["OPENROUTER_API_KEY"])

        # Local Ollama (no real key needed)
        OpenAIProvider(model="llama3.1",
                       base_url="http://localhost:11434/v1",
                       api_key="ollama")

    Parameters
    ----------
    model:
        Model id understood by the target endpoint.
    api_key:
        Falls back to ``OPENAI_API_KEY``.
    base_url:
        Optional OpenAI-compatible endpoint. Falls back to ``OPENAI_BASE_URL``
        if set, otherwise the default OpenAI API.
    max_tokens:
        Output budget for the verdict.
    temperature:
        Sampling temperature. Defaults to ``0.0`` for deterministic verdicts.
    max_retries:
        SDK-level retry count for transient failures.
    timeout:
        Per-request timeout in seconds, passed to the SDK client.

    Note
    ----
    The OpenAI chat API has no separate ``system`` parameter, so the system
    prompt is sent as a leading system-role message.
    """

    name = "openai"

    def __init__(
        self,
        model: str = "gpt-5.5",
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        max_retries: int = 2,
        timeout: float = 60.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "The 'openai' package is required for the OpenAI provider. "
                "It ships with WorkSpec by default — reinstall with: uv pip install -e ."
            ) from exc

        load_dotenv()
        key = api_key or os.environ.get("OPENAI_API_KEY")
        resolved_base = base_url or os.environ.get("OPENAI_BASE_URL")
        if not key and not resolved_base:
            raise RuntimeError(
                "No OpenAI API key found. Set OPENAI_API_KEY, or pass api_key= / "
                "base_url= (e.g. for a local server)."
            )

        # base_url is omitted (not passed as None) so the SDK's own default applies.
        if resolved_base:
            self._client = OpenAI(
                api_key=key or "not-needed",
                base_url=resolved_base,
                max_retries=max_retries,
                timeout=timeout,
            )
        else:
            self._client = OpenAI(
                api_key=key or "not-needed", max_retries=max_retries, timeout=timeout
            )
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def get_structured(self, system_prompt: str, user_prompt: str, schema: type[T]) -> T:
        try:
            completion = self._client.chat.completions.parse(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=schema,
            )
        except RuntimeError:
            raise
        except Exception as exc:
            _raise_if_model_not_found(exc, self.model, self.name)
            raise  # pragma: no cover - unreachable: helper always raises, keeps names bound

        message = completion.choices[0].message
        if getattr(message, "refusal", None):
            raise RuntimeError(f"Model refused: {message.refusal}")
        parsed = message.parsed
        if parsed is None:
            finish_reason = completion.choices[0].finish_reason
            if finish_reason == "length":
                raise RuntimeError(
                    f"OpenAI output was truncated before a complete {schema.__name__} "
                    "could be produced (finish_reason=length). Raise --max-tokens and retry."
                )
            raise RuntimeError(
                f"OpenAI-compatible endpoint returned no parseable {schema.__name__} "
                f"(finish_reason={finish_reason})."
            )
        return parsed


def build_provider(
    provider: str,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    max_tokens: int = 4096,
) -> VerdictProvider:
    """Factory: construct a provider by name.

    Parameters
    ----------
    provider:
        ``"anthropic"`` or ``"openai"`` (the latter covers any OpenAI-compatible
        endpoint via ``base_url``).
    model:
        Optional model override; each provider has a sensible default.
    api_key, base_url, max_tokens:
        Passed through to the chosen provider (``base_url`` is OpenAI-only).
    """
    provider = provider.lower()
    if provider == "anthropic":
        if model:
            return AnthropicProvider(model=model, api_key=api_key, max_tokens=max_tokens)
        return AnthropicProvider(api_key=api_key, max_tokens=max_tokens)
    if provider in ("openai", "openai-compatible", "compatible"):
        if model:
            return OpenAIProvider(
                model=model, api_key=api_key, base_url=base_url, max_tokens=max_tokens
            )
        return OpenAIProvider(api_key=api_key, base_url=base_url, max_tokens=max_tokens)
    raise ValueError(f"Unknown provider '{provider}'. Use 'anthropic' or 'openai'.")
