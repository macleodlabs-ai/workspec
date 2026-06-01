"""Live integration tests — run real check/draft/learn against each backend.

Parametrized over every reachable backend (see ``conftest.live_backends``):

  * **ollama**    — a local Ollama server (OpenAI-compatible endpoint).
  * **anthropic** — when ``ANTHROPIC_API_KEY`` is set (uses a cheap model).
  * **openai**    — when ``OPENAI_API_KEY`` is set (uses a cheap model).

Each backend is skipped when its credentials/endpoint aren't available, so the
suite is safe to run anywhere. These hit a real model, so they're marked
``integration`` and assert the structured-output *contract*, not the model's
judgment (small/cheap models aren't reliable enough for that).

    uv run pytest -m integration -v        # only the live tests
    uv run pytest -m "not integration"     # skip them entirely
"""

from __future__ import annotations

import pytest

from tests.helpers import Backend
from workspec.draft import Draft, DraftAgent
from workspec.engine import WorkSpecAgent
from workspec.models import Finding, Severity, Verdict
from workspec.profile import VoiceTrait
from workspec.spec_loader import load_spec

pytestmark = pytest.mark.integration


def _check_agent(backend: Backend) -> WorkSpecAgent:
    return WorkSpecAgent(
        provider=backend.provider,
        model=backend.model,
        api_key=backend.api_key,
        base_url=backend.base_url,
        # Generous ceiling: a full verdict JSON for a rich spec can be long, and a
        # too-small budget truncates the structured output mid-string.
        max_tokens=4096,
    )


def _draft_agent(backend: Backend) -> DraftAgent:
    return DraftAgent(
        provider=backend.provider,
        model=backend.model,
        api_key=backend.api_key,
        base_url=backend.base_url,
        max_tokens=2048,
    )


def test_check_returns_valid_verdict(backend: Backend) -> None:
    spec = load_spec("decision_memo")
    work = "We should maybe do the thing. It'll probably be fine. Let's sync soon."

    verdict = _check_agent(backend).check(spec, work)

    # Assert the structured-output *contract* held — not the model's judgment or
    # how richly it filled free-text fields (small local models vary a lot).
    assert isinstance(verdict, Verdict)
    assert isinstance(verdict.passed, bool)
    assert isinstance(verdict.summary, str)
    assert isinstance(verdict.findings, list)
    for f in verdict.findings:
        assert isinstance(f, Finding)
        assert isinstance(f.severity, Severity)
        assert isinstance(f.problem, str)
        assert isinstance(f.suggested_fix, str)


def test_draft_returns_reply(backend: Backend) -> None:
    spec = load_spec("email_reply")
    incoming = "Hi — will the Q3 report be ready by Friday? Need it for the board. Thanks."

    result = _draft_agent(backend).draft(spec, incoming)

    assert isinstance(result, Draft)
    assert result.draft.strip()
    assert isinstance(result.open_questions, list)


def test_learn_from_edit_extracts_traits(backend: Backend) -> None:
    # Dry run (apply=False): exercises the learning path without touching disk.
    draft = "Dear Sir or Madam, I am writing to inform you that the report is attached herewith."
    sent = "Hey! Report's attached. Shout if you need anything."

    traits = _draft_agent(backend).learn_from_edit(draft=draft, sent=sent, apply=False)

    assert isinstance(traits, list)
    for t in traits:
        assert isinstance(t, VoiceTrait)
        assert t.provenance == "edit"
