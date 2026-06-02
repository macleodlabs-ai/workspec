"""Tests for the learned contract overlay: lifecycle, propose-first, apply_delta.

The contract overlay reuses the generic learning lifecycle (recurrence, decay,
contradiction, semantic) over a *structural* element instead of a voice trait.
These tests pin the propose-first invariant (a graduated element is a proposal,
not a gate, until confirmed) and the spec fold.
"""

from __future__ import annotations

import pytest

from workspec.contract import ContractDelta, ContractElement, apply_delta
from workspec.learning import semantic
from workspec.learning.recurrence import GRADUATION_OBSERVATIONS
from workspec.models import Spec


@pytest.fixture(autouse=True)
def _no_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the lexical dedup fallback so identity tests are deterministic.

    The semantic matcher is allowed to be unreachable (it degrades to lexical
    Jaccard); pinning it to ``None`` here makes the fallback path the one under
    test regardless of whether a local embedding server happens to be running.
    """
    monkeypatch.setattr(semantic, "semantic_match", lambda *a, **k: None)


def _spec() -> Spec:
    return Spec(
        type="status_update",
        title="Status update",
        must_include=["a clear decision"],
    )


def _graduate(delta: ContractDelta, kind: str, rule: str) -> ContractElement:
    """Reinforce ``rule`` enough times to graduate it to an active proposal."""
    element = delta.reinforce_or_add(kind, rule, "edit")  # type: ignore[arg-type]
    for _ in range(GRADUATION_OBSERVATIONS - 1):
        element = delta.reinforce_or_add(kind, rule, "edit")  # type: ignore[arg-type]
    return element


# --- lifecycle: provisional -> proposed -> confirmed ----------------------- #


def test_single_occurrence_stays_provisional() -> None:
    delta = ContractDelta()
    element = delta.reinforce_or_add("must_include", "State a next step.", "edit")
    assert element.status == "provisional"
    assert not element.gating
    assert not element.proposed
    assert delta.proposals() == []


def test_graduates_to_proposal_not_gating() -> None:
    delta = ContractDelta()
    element = _graduate(delta, "must_include", "State a next step.")
    # Graduated to active by recurrence, but a proposal — never silently gating.
    assert element.status == "active"
    assert element.proposed
    assert not element.gating
    assert delta.proposals() == [element]
    assert delta.gating_elements() == []


def test_confirm_makes_it_gate() -> None:
    delta = ContractDelta()
    element = _graduate(delta, "must_include", "State a next step.")
    confirmed = delta.confirm(element.key)
    assert confirmed is element
    assert element.confirmed
    assert element.gating
    assert not element.proposed
    assert delta.gating_elements() == [element]
    assert delta.proposals() == []


def test_confirm_unknown_or_already_confirmed_returns_none() -> None:
    delta = ContractDelta()
    assert delta.confirm("must_include:nope") is None
    element = _graduate(delta, "must_include", "State a next step.")
    delta.confirm(element.key)
    # A second confirm is a no-op (already confirmed).
    assert delta.confirm(element.key) is None


def test_confirm_provisional_is_rejected() -> None:
    delta = ContractDelta()
    element = delta.reinforce_or_add("must_include", "State a next step.", "edit")
    # Only an active proposal can be confirmed; a provisional one cannot.
    assert delta.confirm(element.key) is None
    assert not element.confirmed


def test_reject_retires_proposal() -> None:
    delta = ContractDelta()
    element = _graduate(delta, "must_include", "State a next step.")
    rejected = delta.reject(element.key)
    assert rejected is element
    assert element.status == "retired"
    assert delta.proposals() == []
    assert delta.gating_elements() == []


def test_reject_unknown_returns_none() -> None:
    delta = ContractDelta()
    assert delta.reject("must_include:nope") is None


def test_reject_confirmed_is_no_op() -> None:
    delta = ContractDelta()
    element = _graduate(delta, "must_include", "State a next step.")
    delta.confirm(element.key)
    # Once confirmed it is no longer a proposal, so reject does not touch it.
    assert delta.reject(element.key) is None
    assert element.gating


# --- apply_delta: only confirmed elements shape the spec ------------------- #


def test_apply_delta_identity_when_nothing_gates() -> None:
    spec = _spec()
    delta = ContractDelta()
    _graduate(delta, "must_include", "State a next step.")  # proposal, not confirmed
    # Propose-first no-op: returns the SAME object, byte-identical check.
    assert apply_delta(spec, delta) is spec


def test_apply_delta_adds_confirmed_must_include() -> None:
    spec = _spec()
    delta = ContractDelta()
    element = _graduate(delta, "must_include", "State a next step.")
    delta.confirm(element.key)

    out = apply_delta(spec, delta)
    assert out is not spec  # a copy; input never mutated
    assert "State a next step." in out.must_include
    assert "a clear decision" in out.must_include
    # The base spec is untouched.
    assert "State a next step." not in spec.must_include


def test_apply_delta_adds_confirmed_must_not_include() -> None:
    spec = _spec()
    delta = ContractDelta()
    element = _graduate(delta, "must_not_include", "Do not open with an apology.")
    delta.confirm(element.key)

    out = apply_delta(spec, delta)
    assert "Do not open with an apology." in out.must_not_include


def test_apply_delta_suppress_base_drops_required_line() -> None:
    spec = _spec()
    delta = ContractDelta()
    element = _graduate(delta, "suppress_base", "a clear decision")
    delta.confirm(element.key)

    out = apply_delta(spec, delta)
    assert "a clear decision" not in out.must_include


def test_apply_delta_dedupes_against_existing_line() -> None:
    spec = _spec()
    delta = ContractDelta()
    element = _graduate(delta, "must_include", "a clear decision")  # already in base
    delta.confirm(element.key)

    out = apply_delta(spec, delta)
    assert out.must_include.count("a clear decision") == 1


# --- queries / aliases ----------------------------------------------------- #


def test_traits_and_category_aliases_for_generic_learning() -> None:
    delta = ContractDelta()
    element = delta.reinforce_or_add("must_not_include", "No filler.", "edit")
    # The generic learning modules key off ``.traits`` / ``.category``.
    assert delta.traits is delta.elements
    assert element.category == element.kind == "must_not_include"


def test_find_match_lexical_dedup() -> None:
    delta = ContractDelta()
    delta.reinforce_or_add("must_include", "State an explicit next step", "edit")
    match = delta.find_match("State an explicit next step please", "must_include")
    assert match is not None
    assert match.kind == "must_include"


def test_reinforce_collapses_paraphrase_onto_one_element() -> None:
    delta = ContractDelta()
    delta.reinforce_or_add("must_include", "State an explicit next step", "edit")
    delta.reinforce_or_add("must_include", "State an explicit next step now", "edit")
    assert len(delta.elements) == 1
    assert delta.elements[0].observations == 2


def test_reinforce_upgrades_provenance_and_evidence() -> None:
    delta = ContractDelta()
    delta.reinforce_or_add("must_include", "State a next step.", "feedback", evidence="said so")
    # A stronger provenance (edit > feedback) upgrades the source of record, and a
    # fresh evidence note overwrites the old one.
    upgraded = delta.reinforce_or_add(
        "must_include", "State a next step.", "edit", evidence="from the diff"
    )
    assert upgraded.provenance == "edit"
    assert upgraded.evidence == "from the diff"


def test_find_similar_ignores_empty_rules() -> None:
    delta = ContractDelta()
    delta.elements.append(ContractElement(kind="must_include", rule=""))
    # Both rules empty -> empty union -> no match (the ``continue`` branch).
    assert delta._find_similar("", "must_include") is None


# --- graft_element (cross-scope carry) ------------------------------------- #


def _earned(rule: str, kind: str = "must_include", confirmed: bool = False) -> ContractElement:
    return ContractElement(
        kind=kind,  # type: ignore[arg-type]
        rule=rule,
        provenance="edit",
        weight=1.0,
        status="active",
        confirmed=confirmed,
        observations=3,
    )


def test_graft_appends_when_no_match() -> None:
    delta = ContractDelta()
    grafted = delta.graft_element(_earned("Name an owner."))
    assert grafted.rule == "Name an owner."
    assert grafted.status == "active"
    # A deep copy, not the same object.
    assert delta.elements == [grafted]


def test_graft_merges_and_keeps_confirmed_sticky() -> None:
    delta = ContractDelta(elements=[_earned("State a next step.", confirmed=True)])
    # An un-confirmed incoming match must not un-confirm the gating element.
    merged = delta.graft_element(_earned("State a next step.", confirmed=False))
    assert len(delta.elements) == 1
    assert merged.confirmed
    assert merged.observations == 6  # 3 + 3 summed


def test_graft_merge_upgrades_status_and_provenance() -> None:
    weak = ContractElement(
        kind="must_include",
        rule="State a next step.",
        provenance="feedback",
        weight=0.4,
        status="provisional",
        observations=1,
    )
    delta = ContractDelta(elements=[weak])
    incoming = _earned("State a next step.")  # active, edit, weight 1.0
    incoming.evidence = "promoted"
    merged = delta.graft_element(incoming)
    assert merged.status == "active"
    assert merged.provenance == "edit"
    assert merged.weight == 1.0
    assert merged.evidence == "promoted"


def test_graft_active_child_retires_conflicting_parent() -> None:
    # Antonym (short/long) conflict with low token overlap so they do NOT dedup
    # into one element but DO contradict.
    parent = ContractElement(
        kind="must_include",
        rule="Keep it short.",
        provenance="edit",
        weight=0.5,
        status="active",
        observations=3,
    )
    delta = ContractDelta(elements=[parent])
    child = ContractElement(
        kind="must_include",
        rule="Write a long reply.",
        provenance="edit",
        weight=1.0,
        status="active",
        observations=3,
    )
    delta.graft_element(child)
    statuses = {e.rule: e.status for e in delta.elements}
    # The stronger child wins; the weaker parent is retired (child overrides).
    assert statuses["Write a long reply."] == "active"
    assert statuses["Keep it short."] == "retired"


def test_graft_accumulated_observations_graduate_to_proposal() -> None:
    # Two provisional sightings of the same element, neither graduated alone, but
    # the summed observations cross the gate -> grafting must graduate it (to a
    # proposal: active, not yet confirmed), mirroring reinforce_or_add.
    existing = ContractElement(
        kind="must_include",
        rule="State a next step.",
        provenance="edit",
        weight=0.4,
        status="provisional",
        observations=2,
    )
    delta = ContractDelta(elements=[existing])
    incoming = ContractElement(
        kind="must_include",
        rule="State a next step.",
        provenance="edit",
        weight=0.4,
        status="provisional",
        observations=1,
    )
    merged = delta.graft_element(incoming)
    assert merged.observations == 3
    assert merged.status == "active"  # graduated by recurrence
    assert merged.proposed  # active but unconfirmed -> a proposal, not gating
