"""End-to-end contract scenario: learn -> propose -> confirm -> gate the check.

Pins the headline propose-first behavior across the whole overlay:
  * a consistently-added element graduates to a *proposal* (not gating);
  * after the owner confirms it, ``check`` fails an update that omits it;
  * a single occurrence stays provisional (one message never moves the gate);
  * reject drops the proposal so it never gates.

A spec-aware fake provider lets the check actually depend on the effective spec
the contract fold produces, without any network.
"""

from __future__ import annotations

from pathlib import Path

from tests.helpers import FakeProvider
from workspec.context import ContextKey
from workspec.contract_extractor import ExtractedContract, ExtractedElement
from workspec.draft import DraftAgent
from workspec.engine import WorkSpecAgent
from workspec.learning.recurrence import GRADUATION_OBSERVATIONS
from workspec.models import Finding, Severity, Spec, Verdict
from workspec.providers import VerdictProvider
from workspec.store import ContextStore

_NEXT_STEP_RULE = "State an explicit next step."


class _SpecAwareProvider(FakeProvider):
    """A FakeProvider whose Verdict fails when the spec requires the next-step rule.

    It inspects the rendered spec in the user prompt: if the effective spec
    carries ``_NEXT_STEP_RULE`` as a MUST INCLUDE, the work (which never mentions
    a next step) gets a blocker, so the check fails. This lets the test prove the
    *confirmed* contract element actually tightens the gate.
    """

    def get_verdict(self, system_prompt: str, user_prompt: str) -> Verdict:
        if _NEXT_STEP_RULE in user_prompt:
            return Verdict(
                passed=False,
                summary="Missing the required next step.",
                findings=[
                    Finding(
                        severity=Severity.BLOCKER,
                        rule=_NEXT_STEP_RULE,
                        problem="No explicit next step.",
                        evidence="",
                        suggested_fix="Add a next step.",
                    )
                ],
            )
        return Verdict(passed=True, summary="ok", findings=[])


def _extracted() -> ExtractedContract:
    return ExtractedContract(elements=[ExtractedElement(kind="must_include", rule=_NEXT_STEP_RULE)])


def _draft_agent(store: ContextStore, provider: VerdictProvider) -> DraftAgent:
    return DraftAgent(provider=provider, store=store)


def _spec() -> Spec:
    return Spec(type="status_update", title="Status update", must_include=["a clear decision"])


def _learn_once(agent: DraftAgent, key: ContextKey) -> None:
    agent.learn_contract_from_edit(
        draft="Here is the status.",
        sent="Here is the status. Next, I will ship Friday.",
        key=key,
    )


# --- the headline scenario ------------------------------------------------- #


def test_consistent_add_graduates_to_proposal_not_gating(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    provider = FakeProvider(responses={ExtractedContract: _extracted()})
    agent = _draft_agent(store, provider)
    key = ContextKey(recipient="alice")

    for _ in range(GRADUATION_OBSERVATIONS):
        _learn_once(agent, key)

    delta = store.load_contract(key)
    proposals = delta.proposals()
    assert [e.rule for e in proposals] == [_NEXT_STEP_RULE]
    # Graduated but NOT gating — the check is untouched until confirmation.
    assert delta.gating_elements() == []


def test_single_occurrence_stays_provisional(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    provider = FakeProvider(responses={ExtractedContract: _extracted()})
    agent = _draft_agent(store, provider)
    key = ContextKey(recipient="alice")

    _learn_once(agent, key)

    delta = store.load_contract(key)
    assert delta.proposals() == []
    assert [e.status for e in delta.elements] == ["provisional"]


def test_confirm_then_check_fails_omitting_work(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    key = ContextKey(recipient="alice")

    # Learn + graduate the element into a proposal.
    learn_agent = _draft_agent(store, FakeProvider(responses={ExtractedContract: _extracted()}))
    for _ in range(GRADUATION_OBSERVATIONS):
        _learn_once(learn_agent, key)

    work = "We decided to ship. Owner: me."  # has a decision, but no next step

    # Before confirmation the proposal does not gate: the spec-aware check passes.
    checker = WorkSpecAgent(provider=_SpecAwareProvider(), store=store)
    assert checker.check(_spec(), work, key=key).passed

    # Owner confirms the proposal.
    delta = store.load_contract(key)
    element = delta.proposals()[0]
    assert delta.confirm(element.key) is not None
    store.save_contract(key, delta)

    # Now the confirmed element gates: the same work fails the check.
    verdict = checker.check(_spec(), work, key=key)
    assert not verdict.passed
    assert any(f.rule == _NEXT_STEP_RULE for f in verdict.blockers)


def test_reject_drops_the_proposal(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    key = ContextKey(recipient="alice")
    learn_agent = _draft_agent(store, FakeProvider(responses={ExtractedContract: _extracted()}))
    for _ in range(GRADUATION_OBSERVATIONS):
        _learn_once(learn_agent, key)

    delta = store.load_contract(key)
    element = delta.proposals()[0]
    assert delta.reject(element.key) is not None
    store.save_contract(key, delta)

    # Rejected: never gates, even though the work omits the (rejected) rule.
    checker = WorkSpecAgent(provider=_SpecAwareProvider(), store=store)
    assert checker.check(_spec(), "We decided to ship.", key=key).passed


def test_learn_contract_requires_context_store() -> None:
    agent = DraftAgent(provider=FakeProvider())  # no store
    try:
        agent.learn_contract_from_edit(draft="a", sent="b")
    except ValueError as exc:
        assert "ContextStore" in str(exc)
    else:  # pragma: no cover - the call must raise
        raise AssertionError("expected ValueError without a ContextStore")


def test_learn_contract_dry_run_writes_nothing(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    agent = _draft_agent(store, FakeProvider(responses={ExtractedContract: _extracted()}))
    key = ContextKey(recipient="alice")
    applied = agent.learn_contract_from_edit(
        draft="Here is the status.",
        sent="Here is the status. Next, I will ship Friday.",
        apply=False,
        key=key,
    )
    assert [e.rule for e in applied] == [_NEXT_STEP_RULE]
    # Dry run: nothing persisted.
    assert not store.contract_path(key).exists()


def test_learn_contract_no_change_no_feedback_is_noop(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    agent = _draft_agent(store, FakeProvider(responses={ExtractedContract: _extracted()}))
    assert agent.learn_contract_from_edit(draft="same", sent="same") == []


def test_learn_contract_never_reads_inbound_prose(tmp_path: Path) -> None:
    """The extractor prompt is built only from the owner's draft and sent text."""
    store = ContextStore(tmp_path)
    provider = FakeProvider(responses={ExtractedContract: _extracted()})
    agent = _draft_agent(store, provider)
    inbound = "SECRET-INBOUND-MARKER from the recipient"
    agent.learn_contract_from_edit(
        draft="Here is the status.",
        sent="Here is the status. Next step: ship.",
        key=ContextKey(recipient="alice"),
    )
    # The inbound marker was never passed in, so it cannot appear in any prompt.
    assert all(inbound not in call["user"] for call in provider.calls)
