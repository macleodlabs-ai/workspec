"""Unit tests for the lint engine (fake provider — no network)."""

from __future__ import annotations

import pytest

from tests.helpers import FakeProvider
from workspec.engine import DEFAULT_MODEL, DEFAULT_OPENAI_MODEL, WorkSpecAgent
from workspec.models import Spec, Verdict


def _spec() -> Spec:
    return Spec(type="memo", title="Memo", must_include=["owner"])


def test_check_returns_verdict_and_builds_prompt(fake_provider: FakeProvider) -> None:
    agent = WorkSpecAgent(provider=fake_provider)
    verdict = agent.check(_spec(), "Here is the work, owned by Sam.")
    assert isinstance(verdict, Verdict)
    assert len(fake_provider.calls) == 1
    call = fake_provider.calls[0]
    assert call["schema"] is Verdict
    assert "Here is the work" in call["user"]
    assert "SPEC TYPE: memo" in call["user"]


def test_check_empty_work_raises(fake_provider: FakeProvider) -> None:
    agent = WorkSpecAgent(provider=fake_provider)
    with pytest.raises(ValueError, match="empty"):
        agent.check(_spec(), "   ")


def test_agent_accepts_provider_instance(fake_provider: FakeProvider) -> None:
    agent = WorkSpecAgent(provider=fake_provider)
    assert agent.provider is fake_provider


def test_default_model_constants() -> None:
    assert DEFAULT_MODEL == "claude-opus-4-8"
    assert DEFAULT_OPENAI_MODEL == "gpt-5.5"
