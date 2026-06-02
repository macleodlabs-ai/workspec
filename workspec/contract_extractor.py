"""Contract extraction — mine the owner's draft→sent edit for *structural* habits.

This is the structural counterpart to the voice extractor in
:mod:`workspec.draft`. Where the voice extractor asks "what does this edit reveal
about their *style*", the contract extractor asks the opposite question: "what
*structural element* does this person reliably add to — or strip from — a proper
update?" Added structure becomes a candidate ``must_include``; stripped structure
becomes a candidate ``suppress_base`` (the base spec over-asked) or
``must_not_include`` (an anti-pattern they reject).

Like every learning path in WorkSpec this reads **only** the owner's own outbound
draft→sent edit and optional feedback (Decision 3); it never reads inbound prose.
A single edit never changes the contract: each candidate enters the delta as a
``provisional`` element and only graduates to an (un-confirmed) *proposal* after
the same 3-observation recurrence gate that governs voice traits (Decision 5).
The provider is reused — no new backend.
"""

from __future__ import annotations

from typing import cast, get_args

from pydantic import BaseModel, Field

from workspec.contract import ContractDelta, ContractElement, ContractKind
from workspec.profile import Provenance

_EXTRACT_SYSTEM = """\
You analyze how a person edited a draft reply before sending it, to learn the \
STRUCTURAL CONTRACT of a proper update from them — what a good update must \
contain, and what it must not. You are not learning their writing *style* (tone, \
phrasing); a separate process does that. You learn only STRUCTURE.

Compare the DRAFT (what an assistant wrote) with the SENT version (what the \
person actually sent) and extract durable, reusable STRUCTURAL elements:
  - must_include: a structural element the person ADDED and would expect every
    such update to carry (an explicit owner, a decision, a deadline, a next
    step, a data source, a risk call-out, an explicit ask).
  - must_not_include: an anti-pattern the person consistently STRIPPED OUT
    (hedging filler, an apology opener, a status with no decision).
  - suppress_base: a required element the person deliberately and acceptably
    OMITTED because for them it does not belong in this kind of update.

Extract only STRUCTURAL patterns that will generalize to the NEXT update. Ignore:
  - this message's specific facts (a date, a name, a number) — they don't
    generalize.
  - tone / phrasing / formatting changes — that is voice, not contract.
If the edit reveals no generalizable structural element, return an empty list.
Never invent elements to seem thorough.
"""

_EXTRACT_USER = """\
Extract generalizable STRUCTURAL contract elements from how this person edited \
the draft.
{feedback_block}
=== DRAFT (assistant wrote) ===
{draft}
=== SENT (person actually sent) ===
{sent}
=== UNIFIED DIFF (draft -> sent) ===
{diff}
=== END ===

Return only structural elements that will generalize to future updates.
"""


class ExtractedElement(BaseModel):
    """One structural element a model mined from a draft→sent edit."""

    kind: str = Field(description="One of: must_include, must_not_include, suppress_base.")
    rule: str = Field(description="The structural element, stated as a spec line.")
    evidence: str = Field(default="", description="What in the edit shows this.")


class ExtractedContract(BaseModel):
    """The structural contract elements a model extracted from an edit."""

    elements: list[ExtractedElement] = Field(default_factory=list)
    summary: str = Field(
        default="", description="One line on what the edit revealed about their contract."
    )


_VALID_KINDS: frozenset[str] = frozenset(get_args(ContractKind))


def _safe_kind(kind: str) -> ContractKind:
    """Normalize a model-provided kind, defaulting unknown values to must_include."""
    normalized = kind.strip().lower()
    if normalized in _VALID_KINDS:
        return cast(ContractKind, normalized)
    return "must_include"


def build_extract_prompt(draft: str, sent: str, diff: str, feedback: str = "") -> tuple[str, str]:
    """Return the ``(system, user)`` prompts for the contract extractor.

    Kept separate from the call site so the prompt shape can be unit-tested
    without a provider, and so :class:`~workspec.draft.DraftAgent` can reuse its
    own diff/ratio helpers when invoking it.
    """
    feedback_block = (
        f"\nThe person also gave this explicit feedback: {feedback}\n" if feedback else ""
    )
    user = _EXTRACT_USER.format(
        feedback_block=feedback_block,
        draft=draft.strip(),
        sent=sent.strip(),
        diff=diff or "(no line-level diff; see texts)",
    )
    return _EXTRACT_SYSTEM, user


def absorb_extracted(
    delta: ContractDelta,
    extracted: ExtractedContract,
    provenance: Provenance,
) -> list[ContractElement]:
    """Fold extracted elements into ``delta`` via the recurrence-gated lifecycle.

    Each extracted element is reinforced into ``delta``: a first sighting lands as
    ``provisional`` and only graduates to an (un-confirmed) proposal once it has
    recurred ``GRADUATION_OBSERVATIONS`` times (Decision 5). Returns the resulting
    in-delta elements in extraction order. ``delta`` is mutated in place but not
    persisted — the caller owns the save so a dry run changes nothing on disk.
    """
    return [
        delta.reinforce_or_add(
            kind=_safe_kind(e.kind), rule=e.rule, provenance=provenance, evidence=e.evidence
        )
        for e in extracted.elements
    ]
