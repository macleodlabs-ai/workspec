"""Tests for the contract extractor: prompt shape, kind normalization, absorb."""

from __future__ import annotations

from workspec.contract import ContractDelta
from workspec.contract_extractor import (
    ExtractedContract,
    ExtractedElement,
    absorb_extracted,
    build_extract_prompt,
)
from workspec.learning.recurrence import GRADUATION_OBSERVATIONS


def test_prompt_includes_draft_sent_diff_and_feedback() -> None:
    system, user = build_extract_prompt("the draft", "the sent", "@@ diff @@", "add a deadline")
    # Opposite intent to voice: the system prompt is about STRUCTURE, not style.
    assert "STRUCTURE" in system
    assert "must_include" in system
    assert "the draft" in user
    assert "the sent" in user
    assert "@@ diff @@" in user
    assert "add a deadline" in user


def test_prompt_omits_feedback_block_when_absent() -> None:
    _, user = build_extract_prompt("d", "s", "diff")
    assert "explicit feedback" not in user


def test_prompt_handles_empty_diff() -> None:
    _, user = build_extract_prompt("d", "s", "")
    assert "no line-level diff" in user


def test_absorb_first_sighting_is_provisional() -> None:
    delta = ContractDelta()
    extracted = ExtractedContract(
        elements=[ExtractedElement(kind="must_include", rule="State a next step.")]
    )
    applied = absorb_extracted(delta, extracted, "edit")
    assert len(applied) == 1
    assert applied[0].status == "provisional"
    assert not applied[0].proposed


def test_absorb_recurrence_graduates_to_proposal() -> None:
    delta = ContractDelta()
    extracted = ExtractedContract(
        elements=[ExtractedElement(kind="must_include", rule="State a next step.")]
    )
    last = []
    for _ in range(GRADUATION_OBSERVATIONS):
        last = absorb_extracted(delta, extracted, "edit")
    assert last[0].proposed
    assert not last[0].gating  # propose-first: graduated != gating


def test_absorb_normalizes_unknown_kind_to_must_include() -> None:
    delta = ContractDelta()
    extracted = ExtractedContract(
        elements=[ExtractedElement(kind="garbage", rule="Something structural.")]
    )
    applied = absorb_extracted(delta, extracted, "edit")
    assert applied[0].kind == "must_include"


def test_absorb_preserves_all_kinds() -> None:
    delta = ContractDelta()
    extracted = ExtractedContract(
        elements=[
            ExtractedElement(kind="must_include", rule="Name an owner."),
            ExtractedElement(kind="must_not_include", rule="No apology opener."),
            ExtractedElement(kind="suppress_base", rule="a clear decision"),
        ]
    )
    applied = absorb_extracted(delta, extracted, "edit")
    assert [e.kind for e in applied] == ["must_include", "must_not_include", "suppress_base"]


def test_absorb_empty_extraction_is_noop() -> None:
    delta = ContractDelta()
    assert absorb_extracted(delta, ExtractedContract(), "edit") == []
    assert delta.elements == []
