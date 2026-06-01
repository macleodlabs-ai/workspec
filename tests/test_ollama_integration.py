"""Integration test: run a real WorkSpec check/draft through a local Ollama server.

Ollama exposes an OpenAI-compatible API, so WorkSpec talks to it via the `openai`
provider with ``--base-url http://localhost:11434/v1``. This exercises the full
structured-output path (``chat.completions.parse`` with a JSON schema) against a
real, locally-running model — no cloud credentials needed.

The test **auto-skips** when no Ollama server is reachable (or no chat model is
pulled), so it never breaks CI or a machine without Ollama. To run it:

    ollama serve                       # if not already running
    ollama pull llama3.2               # any chat model works
    uv run pytest tests/ -v

Overrides (env vars):
    OLLAMA_BASE_URL          default http://localhost:11434
    WORKSPEC_OLLAMA_MODEL    force a specific model id (else auto-pick)
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from workspec.engine import WorkSpecAgent
from workspec.models import Finding, Severity, Verdict
from workspec.spec_loader import load_spec

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
# Small, fast chat models preferred first; the test falls back to whatever is pulled.
_PREFERRED = ["llama3.2:latest", "llama3.1:latest", "llama3:latest", "qwen3:32b"]
_STATUS_UPDATE_SPEC = Path(__file__).resolve().parent.parent / "examples" / "status_update.yaml"


def _available_models() -> list[str]:
    """Return chat model ids from the Ollama OpenAI-compatible endpoint, or []."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/v1/models", timeout=2) as resp:
            data = json.load(resp)
    except (urllib.error.URLError, OSError, ValueError):
        return []
    return [m.get("id", "") for m in data.get("data", []) if m.get("id")]


def _pick_model() -> str | None:
    """Choose a model: explicit override, else first preferred, else first available."""
    models = _available_models()
    override = os.environ.get("WORKSPEC_OLLAMA_MODEL")
    if override:
        return override
    if not models:
        return None
    for preferred in _PREFERRED:
        if preferred in models:
            return preferred
    # Skip embedding-only models, which can't do chat completions.
    chat = [m for m in models if "embed" not in m.lower()]
    return chat[0] if chat else None


_MODEL = _pick_model()

pytestmark = pytest.mark.skipif(
    _MODEL is None,
    reason=f"No local Ollama chat model reachable at {OLLAMA_BASE_URL} "
    f"(start it with `ollama serve` and pull a model, e.g. `ollama pull llama3.2`).",
)


@pytest.fixture
def ollama_agent() -> WorkSpecAgent:
    return WorkSpecAgent(
        provider="openai",
        model=_MODEL,
        base_url=f"{OLLAMA_BASE_URL}/v1",
        api_key="ollama",  # Ollama ignores the key but the SDK requires one.
        max_tokens=1024,
    )


@pytest.mark.integration
def test_check_returns_typed_verdict_via_ollama(ollama_agent: WorkSpecAgent) -> None:
    """A local model returns a schema-valid Verdict through the structured-output path."""
    spec = load_spec(str(_STATUS_UPDATE_SPEC))
    work = "Status: things are going fine, making progress. Will keep you posted soon."

    verdict = ollama_agent.check(spec, work)

    # Assert the *contract* (structured outputs held), not the model's judgment —
    # small local models aren't reliable enough to assert specific pass/fail or to
    # always populate every free-text field, but the schema must validate.
    assert isinstance(verdict, Verdict)
    assert isinstance(verdict.passed, bool)
    assert verdict.summary.strip(), "verdict should carry a non-empty summary"
    assert isinstance(verdict.findings, list)
    for finding in verdict.findings:
        assert isinstance(finding, Finding)
        assert isinstance(finding.severity, Severity)
        assert isinstance(finding.problem, str)
        assert isinstance(finding.suggested_fix, str)


@pytest.mark.integration
def test_draft_returns_text_via_ollama() -> None:
    """The draft capability also round-trips through Ollama and yields a non-empty reply."""
    from workspec.draft import Draft, DraftAgent

    agent = DraftAgent(
        provider="openai",
        model=_MODEL,
        base_url=f"{OLLAMA_BASE_URL}/v1",
        api_key="ollama",
        max_tokens=1024,
    )
    spec = load_spec("email_reply")
    incoming = "Hi — can you confirm whether the report will be ready by Friday? Thanks."

    result = agent.draft(spec, incoming)

    assert isinstance(result, Draft)
    assert result.draft.strip(), "draft body should not be empty"
    assert isinstance(result.open_questions, list)
