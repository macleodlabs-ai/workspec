"""Shared test helpers: a fake provider and live-backend detection.

Kept out of ``conftest.py`` so test modules can import these classes directly
without importing the conftest plugin module twice.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import BaseModel

from workspec.draft import Draft, ExtractedTrait, LearnedTraits
from workspec.models import Finding, Severity, Verdict
from workspec.providers import VerdictProvider

# --------------------------------------------------------------------------- #
# Fake provider for unit tests
# --------------------------------------------------------------------------- #


def default_for(schema: type[BaseModel]) -> BaseModel:
    """A minimal, schema-valid instance for each structured-output type."""
    if schema is Verdict:
        return Verdict(
            passed=True,
            summary="Looks manager-ready.",
            findings=[
                Finding(
                    severity=Severity.NOTE,
                    rule="general",
                    problem="Tiny nit.",
                    evidence="",
                    suggested_fix="Optional polish.",
                )
            ],
        )
    if schema is Draft:
        return Draft(
            draft="Thanks — confirming now.",
            rationale="Concise and direct.",
            open_questions=["[CONFIRM: deadline]"],
            used_profile=False,
        )
    if schema is LearnedTraits:
        return LearnedTraits(
            traits=[ExtractedTrait(category="tone", rule="Be concise.", evidence="trimmed filler")],
            summary="Prefers brevity.",
        )
    raise AssertionError(f"FakeProvider has no default for schema {schema!r}")


@dataclass
class FakeProvider(VerdictProvider):
    """A VerdictProvider that returns canned objects and records every call."""

    name: str = "fake"
    responses: dict[type[BaseModel], BaseModel] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def get_structured(  # type: ignore[override]
        self, system_prompt: str, user_prompt: str, schema: type[BaseModel]
    ) -> BaseModel:
        self.calls.append({"system": system_prompt, "user": user_prompt, "schema": schema})
        return self.responses.get(schema) or default_for(schema)


# --------------------------------------------------------------------------- #
# Live backend detection + parametrization
# --------------------------------------------------------------------------- #

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
_OLLAMA_PREFERRED = ["llama3.2:latest", "llama3.1:latest", "llama3:latest", "qwen3:32b"]


@dataclass
class Backend:
    """Connection details for one live provider, fed to WorkSpecAgent/DraftAgent."""

    id: str
    provider: str
    model: str
    api_key: str | None = None
    base_url: str | None = None


def _ollama_models() -> list[str]:
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/v1/models", timeout=2) as resp:
            data = json.load(resp)
    except (urllib.error.URLError, OSError, ValueError):
        return []
    return [m.get("id", "") for m in data.get("data", []) if m.get("id")]


def _ollama_model() -> str | None:
    override = os.environ.get("WORKSPEC_OLLAMA_MODEL")
    if override:
        return override
    models = _ollama_models()
    for preferred in _OLLAMA_PREFERRED:
        if preferred in models:
            return preferred
    chat = [m for m in models if "embed" not in m.lower()]
    return chat[0] if chat else None


def live_backends() -> list[Backend]:
    """Every backend with working credentials/endpoint, in a stable order."""
    backends: list[Backend] = []

    ollama_model = _ollama_model()
    if ollama_model:
        backends.append(
            Backend(
                id="ollama",
                provider="openai",
                model=ollama_model,
                api_key="ollama",
                base_url=f"{OLLAMA_BASE_URL}/v1",
            )
        )

    if os.environ.get("ANTHROPIC_API_KEY"):
        backends.append(
            Backend(
                id="anthropic",
                provider="anthropic",
                model=os.environ.get("ANTHROPIC_TEST_MODEL", "claude-haiku-4-5"),
            )
        )

    if os.environ.get("OPENAI_API_KEY"):
        backends.append(
            Backend(
                id="openai",
                provider="openai",
                model=os.environ.get("OPENAI_TEST_MODEL", "gpt-4o-mini"),
                base_url=os.environ.get("OPENAI_BASE_URL"),
            )
        )

    return backends


def backend_params() -> list[Any]:
    """pytest params for the ``backend`` fixture; a single skip param if none."""
    backends = live_backends()
    if backends:
        return [pytest.param(b, id=b.id) for b in backends]
    return [
        pytest.param(None, id="none", marks=pytest.mark.skip(reason="no live backend available"))
    ]
