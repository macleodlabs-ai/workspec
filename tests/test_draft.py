"""Unit tests for draft generation, learning, and helpers (fake provider)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers import FakeProvider
from workspec.draft import (
    Draft,
    DraftAgent,
    ExtractedTrait,
    LearnedTraits,
    _safe_cat,
    _unified_diff,
)
from workspec.models import Spec
from workspec.profile import ProfileStore, VoiceProfile, VoiceTrait


def _spec() -> Spec:
    return Spec(type="email_reply", title="Reply", must_include=["a clear answer"])


# --- helpers --------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("tone", "tone"),
        ("  ToNe ", "tone"),
        ("do_not", "do_not"),
        ("nonsense", "preference"),
        ("", "preference"),
    ],
)
def test_safe_cat(raw: str, expected: str) -> None:
    assert _safe_cat(raw) == expected


def test_unified_diff_marks_changes() -> None:
    diff = _unified_diff("hello there\nbye", "hello world\nbye")
    assert "-hello there" in diff
    assert "+hello world" in diff


def test_unified_diff_empty_when_identical() -> None:
    assert _unified_diff("same", "same") == ""


# --- draft ----------------------------------------------------------------- #


def test_draft_returns_typed_result(fake_provider: FakeProvider) -> None:
    agent = DraftAgent(provider=fake_provider)
    result = agent.draft(_spec(), "Can you confirm Friday?")
    assert isinstance(result, Draft)
    assert result.draft
    assert len(fake_provider.calls) == 1
    assert fake_provider.calls[0]["schema"] is Draft


def test_draft_empty_submission_raises(fake_provider: FakeProvider) -> None:
    with pytest.raises(ValueError, match="empty"):
        DraftAgent(provider=fake_provider).draft(_spec(), "  ")


def test_draft_instruction_threaded_into_prompt(fake_provider: FakeProvider) -> None:
    agent = DraftAgent(provider=fake_provider)
    agent.draft(_spec(), "Question?", instruction="keep it to three sentences")
    assert "keep it to three sentences" in fake_provider.calls[0]["user"]


def test_draft_used_profile_flag(tmp_path: Path, fake_provider: FakeProvider) -> None:
    store = ProfileStore(tmp_path / ".workspec")
    profile = VoiceProfile()
    profile.reinforce_or_add(category="tone", rule="Be warm.", provenance="edit")
    store.save(profile)

    agent = DraftAgent(provider=fake_provider, profile_store=store)
    result = agent.draft(_spec(), "Question?")
    assert result.used_profile is True
    # profile content should be injected into the prompt
    assert "Be warm." in fake_provider.calls[0]["user"]


# --- learn_from_edit ------------------------------------------------------- #


def test_learn_no_change_no_feedback_returns_empty(fake_provider: FakeProvider) -> None:
    agent = DraftAgent(provider=fake_provider)
    assert agent.learn_from_edit(draft="same text", sent="same text") == []
    assert fake_provider.calls == []  # never calls the model


def test_learn_dry_run_does_not_persist(tmp_path: Path) -> None:
    provider = FakeProvider(
        responses={
            LearnedTraits: LearnedTraits(
                traits=[ExtractedTrait(category="tone", rule="Be brief.", evidence="x")]
            )
        }
    )
    store = ProfileStore(tmp_path / ".workspec")
    agent = DraftAgent(provider=provider, profile_store=store)

    traits = agent.learn_from_edit(draft="long winded draft", sent="short", apply=False)

    assert len(traits) == 1
    assert isinstance(traits[0], VoiceTrait)
    assert traits[0].provenance == "edit"
    assert not store.exists()  # nothing written


def test_learn_provenance_is_feedback_when_no_diff(tmp_path: Path) -> None:
    """draft == sent but feedback given -> the only signal is feedback."""
    provider = FakeProvider(
        responses={
            LearnedTraits: LearnedTraits(
                traits=[ExtractedTrait(category="tone", rule="Be warmer.", evidence="")]
            )
        }
    )
    store = ProfileStore(tmp_path / ".workspec")
    agent = DraftAgent(provider=provider, profile_store=store)

    applied = agent.learn_from_edit(
        draft="identical body", sent="identical body", feedback="be warmer"
    )

    assert len(applied) == 1
    assert applied[0].provenance == "feedback"
    saved = ProfileStore(tmp_path / ".workspec").load()
    assert saved.traits[0].provenance == "feedback"


def test_learn_provenance_is_edit_when_diff(tmp_path: Path) -> None:
    """A real draft -> sent diff is the gold 'edit' signal, even with feedback."""
    provider = FakeProvider(
        responses={
            LearnedTraits: LearnedTraits(
                traits=[ExtractedTrait(category="tone", rule="Be brief.", evidence="")]
            )
        }
    )
    store = ProfileStore(tmp_path / ".workspec")
    agent = DraftAgent(provider=provider, profile_store=store)

    applied = agent.learn_from_edit(
        draft="long winded original draft", sent="short", feedback="too formal"
    )

    assert applied[0].provenance == "edit"


def test_learn_dry_run_provenance_feedback_when_no_diff(tmp_path: Path) -> None:
    """Dry-run provenance must match the apply branch (feedback when no diff)."""
    provider = FakeProvider(
        responses={
            LearnedTraits: LearnedTraits(
                traits=[ExtractedTrait(category="tone", rule="Be warmer.", evidence="")]
            )
        }
    )
    store = ProfileStore(tmp_path / ".workspec")
    agent = DraftAgent(provider=provider, profile_store=store)

    traits = agent.learn_from_edit(draft="same", sent="same", feedback="be warmer", apply=False)

    assert traits[0].provenance == "feedback"
    assert not store.exists()


def test_learn_persists_traits(tmp_path: Path) -> None:
    provider = FakeProvider(
        responses={
            LearnedTraits: LearnedTraits(
                traits=[
                    ExtractedTrait(category="tone", rule="Be brief.", evidence="trimmed"),
                    ExtractedTrait(category="weird-cat", rule="Use bullets.", evidence=""),
                ]
            )
        }
    )
    store = ProfileStore(tmp_path / ".workspec")
    agent = DraftAgent(provider=provider, profile_store=store)

    applied = agent.learn_from_edit(draft="long", sent="short", feedback="too formal")

    assert len(applied) == 2
    saved = ProfileStore(tmp_path / ".workspec").load()
    assert {t.rule for t in saved.traits} == {"Be brief.", "Use bullets."}
    # invalid category was coerced to a valid one
    assert all(
        t.category
        in {
            "tone",
            "structure",
            "preference",
            "formatting",
            "do_not",
            "phrasing",
            "salutation",
            "signoff",
            "length",
        }
        for t in saved.traits
    )
