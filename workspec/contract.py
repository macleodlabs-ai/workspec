"""The learned contract overlay — WorkSpec's model of *what a proper update is*.

The :class:`~workspec.profile.VoiceProfile` learns *how* this person writes
(style, drives drafting). This module is its structural twin: it learns *what* a
proper update from this person must (or must not) contain, and that drives the
``check``. The two axes are deliberately separate (Decision 2): voice shapes the
draft, contract shapes the gate.

A :class:`ContractElement` is the structural analogue of a
:class:`~workspec.profile.VoiceTrait` — same provenance/weight/status/observation
lifecycle, so it can reuse the generic learning modules
(``recurrence``/``decay``/``contradiction``/``semantic``) verbatim. Three kinds:

  * ``must_include``      — a structural element the work must contain.
  * ``must_not_include``  — an anti-pattern the work must avoid.
  * ``suppress_base``     — a base-spec ``must_include`` line this person's
                            updates consistently (and acceptably) omit, so it
                            should stop being enforced for this context.

PROPOSE-FIRST (Decision 5): contract learning never silently starts failing the
check. A learned element graduates by recurrence to ``status='active'`` — but an
active element is only a *proposal* until the owner confirms it. Gating is
keyed off :attr:`ContractElement.gating` (active **and** confirmed); an
un-confirmed active element is surfaced as a proposal and changes nothing.

This learns only from the owner's own outbound draft→sent edits, feedback, and
owner-side check history (Decision 3); it never reads or trains on inbound prose.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Protocol, TypeVar

from pydantic import BaseModel, Field

from workspec.learning import contradiction, recurrence, semantic
from workspec.models import Spec
from workspec.profile import (
    _STATUS_RANK,
    PROVENANCE_WEIGHT,
    Provenance,
    TraitStatus,
)

#: What a contract element does to the effective spec.
#:  * ``must_include``     adds a required structural element.
#:  * ``must_not_include`` adds a forbidden anti-pattern.
#:  * ``suppress_base``    removes a base-spec ``must_include`` line that this
#:                         context consistently and acceptably omits.
ContractKind = Literal["must_include", "must_not_include", "suppress_base"]


class ContractElement(BaseModel):
    """One learned structural requirement about this person's updates.

    The structural twin of :class:`~workspec.profile.VoiceTrait`: it carries the
    identical provenance/weight/status/observation lifecycle so the generic
    learning modules apply unchanged, plus a :attr:`kind` (what it does to the
    spec) and a :attr:`confirmed` flag (the propose-first gate, Decision 5).
    """

    kind: ContractKind
    rule: str = Field(description="The structural element, stated as a spec line.")
    provenance: Provenance = "seed"
    weight: float = Field(default=0.7, ge=0.0, le=1.0)
    status: TraitStatus = Field(
        default="provisional",
        description="Lifecycle stage. Elements are born 'provisional' and "
        "graduate to 'active' (a *proposal*) once they recur; conflicting / "
        "rejected elements are 'retired'.",
    )
    confirmed: bool = Field(
        default=False,
        description="Whether the owner has confirmed this active element. Only a "
        "confirmed, active element gates the check (propose-first, Decision 5).",
    )
    observations: int = Field(
        default=1,
        description="Count of distinct learn events that produced this element; "
        "drives recurrence-based graduation.",
    )
    last_seen: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp of the most recent reinforcement; drives decay.",
    )
    evidence: str = Field(
        default="",
        description="Short note on where this came from, so the owner can audit "
        "the learned contract.",
    )
    hits: int = Field(default=1, description="How many times signal reinforced this element.")
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def key(self) -> str:
        """Stable identifier (``kind:rule``)."""
        return f"{self.kind}:{self.rule}"

    @property
    def category(self) -> str:
        """Alias of :attr:`kind` for the generic learning modules.

        ``semantic``/``contradiction`` are generic over "same-category" lifecycle
        rules and key off a ``category`` attribute; exposing :attr:`kind` under
        that name lets the contract overlay reuse them verbatim (Decision 7)
        without copying their logic.
        """
        return self.kind

    @property
    def gating(self) -> bool:
        """True only when this element should actually shape the effective spec.

        Propose-first: an element gates the check only once it has graduated to
        ``active`` *and* the owner has confirmed it. A provisional, retired, or
        un-confirmed-active element never alters the spec.
        """
        return self.status == "active" and self.confirmed

    @property
    def proposed(self) -> bool:
        """True when this element is a *surfaced proposal* awaiting confirmation.

        An ``active`` element the owner has not yet confirmed: graduated by
        recurrence but not yet gating. These are what ``--proposals`` lists and
        what ``contract confirm`` acts on.
        """
        return self.status == "active" and not self.confirmed


class ContractDelta(BaseModel):
    """The per-scope set of learned contract elements (the structural overlay).

    The structural twin of :class:`~workspec.profile.VoiceProfile`. Stored one
    file per scope via :class:`~workspec.store.ContextStore`; folded across the
    backoff chain by :func:`workspec.compose.compose` into the effective spec
    used for the check.
    """

    owner: str = Field(default="", description="Whose contract this is (free text).")
    elements: list[ContractElement] = Field(default_factory=list)
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def traits(self) -> list[ContractElement]:
        """Alias of :attr:`elements` for the generic learning modules.

        ``semantic``/``contradiction`` iterate a profile's ``traits``; exposing
        :attr:`elements` under that name lets the contract overlay reuse those
        modules verbatim (Decision 7).
        """
        return self.elements

    # --- queries ---------------------------------------------------------- #

    def gating_elements(self) -> list[ContractElement]:
        """Confirmed, active elements that actually shape the effective spec."""
        return [e for e in self.elements if e.gating]

    def proposals(self) -> list[ContractElement]:
        """Graduated-but-unconfirmed elements awaiting an owner decision."""
        return [e for e in self.elements if e.proposed]

    # --- identity --------------------------------------------------------- #

    def _find_similar(self, rule: str, kind: str) -> ContractElement | None:
        """Lexical Jaccard dedup: same kind + high token overlap == same element.

        Delegates to the generic :func:`lexical_match`, the shared lexical
        fallback, so contract dedup behaves the same way voice dedup does,
        order-independently and excluding retired elements. The contract side
        exposes ``.category`` / ``.traits`` aliases (Decision 7), so the generic
        helper applies verbatim instead of re-implementing the Jaccard test.
        """
        return lexical_match(self, rule, kind)

    def find_match(self, rule: str, kind: str) -> ContractElement | None:
        """Find a non-retired element equivalent to ``rule`` in ``kind``, else None.

        Uses the same identity test as :meth:`reinforce_or_add` — semantic
        (embedding) dedup first, then the lexical Jaccard fallback — reusing the
        shared :mod:`workspec.learning.semantic` matcher generically (it keys off
        a ``category`` attribute; the contract maps that to :attr:`kind`).
        """
        return semantic.semantic_match(self, rule, kind) or self._find_similar(rule, kind)

    # --- mutation --------------------------------------------------------- #

    def reinforce_or_add(
        self,
        kind: ContractKind,
        rule: str,
        provenance: Provenance,
        evidence: str = "",
    ) -> ContractElement:
        """Add a new element, or strengthen an existing similar one.

        Identical lifecycle to :meth:`VoiceProfile.reinforce_or_add`: a brand-new
        element is born ``provisional`` (its weight clamped under the provisional
        cap), graduating to ``active`` — and thereby becoming a *proposal* — only
        once it recurs enough (:mod:`workspec.learning.recurrence`). Reinforcement
        raises weight toward the provenance ceiling and may upgrade a weak
        element's provenance. An active element that recurs resolves
        contradictions, retiring the weaker conflicting side.
        """
        base_weight = PROVENANCE_WEIGHT.get(provenance, 0.5)
        existing = self.find_match(rule, kind)
        now = datetime.now(timezone.utc).isoformat()

        if existing is not None:
            existing.hits += 1
            existing.observations += 1
            existing.last_seen = now
            ceiling = max(existing.weight, base_weight)
            existing.weight = min(ceiling, existing.weight + (ceiling - existing.weight) / 3 + 0.05)
            if PROVENANCE_WEIGHT.get(provenance, 0) > PROVENANCE_WEIGHT.get(existing.provenance, 0):
                existing.provenance = provenance
            if evidence:
                existing.evidence = evidence
            existing.updated_at = now
            self.updated_at = now
            recurrence.maybe_graduate(existing)
            if existing.status == "active":
                contradiction.detect_and_resolve(self, existing)
            return existing

        element = ContractElement(
            kind=kind,
            rule=rule,
            provenance=provenance,
            weight=base_weight,
            status="provisional",
            observations=1,
            last_seen=now,
            evidence=evidence,
        )
        recurrence.maybe_graduate(element)
        self.elements.append(element)
        self.updated_at = now
        return element

    def graft_element(self, incoming: ContractElement) -> ContractElement:
        """Merge a fully-formed element into this delta, preserving its lifecycle.

        The structural twin of :meth:`VoiceProfile.graft_trait`: it carries an
        already-earned element across scopes intact (used by the backoff fold), so
        a more-specific scope's confirmed, gating element wins over a general one.
        A match here is reinforced (extra observations, max weight, never-downgraded
        status, and ``confirmed`` is sticky — a confirmed side stays confirmed);
        otherwise the incoming element is appended as a deep copy. An ``active``
        result resolves contradictions so a child scope overrides a conflicting
        parent (Decision 6).
        """
        now = datetime.now(timezone.utc).isoformat()
        existing = self.find_match(incoming.rule, incoming.kind)
        if existing is not None:
            existing.hits += 1
            existing.observations += incoming.observations
            existing.last_seen = now
            existing.weight = max(existing.weight, incoming.weight)
            if _STATUS_RANK[incoming.status] > _STATUS_RANK[existing.status]:
                existing.status = incoming.status
            existing.confirmed = existing.confirmed or incoming.confirmed
            if PROVENANCE_WEIGHT.get(incoming.provenance, 0) > PROVENANCE_WEIGHT.get(
                existing.provenance, 0
            ):
                existing.provenance = incoming.provenance
            if incoming.evidence:
                existing.evidence = incoming.evidence
            existing.updated_at = now
            self.updated_at = now
            # Accumulated observations can cross the graduation gate; graduate
            # before resolving contradictions so a freshly-active grafted element
            # (a proposal) can retire a conflicting one.
            recurrence.maybe_graduate(existing)
            if existing.status == "active":
                contradiction.detect_and_resolve(self, existing)
            return existing

        grafted = incoming.model_copy(deep=True)
        self.elements.append(grafted)
        self.updated_at = now
        if grafted.status == "active":
            contradiction.detect_and_resolve(self, grafted)
        return grafted

    def confirm(self, key: str) -> ContractElement | None:
        """Confirm the active element whose :attr:`~ContractElement.key` matches.

        Flips an active *proposal* to gating (``confirmed=True``) so it starts
        shaping the check. Returns the confirmed element, or ``None`` when no
        active element matches ``key`` (already confirmed, retired, or unknown).
        """
        for e in self.elements:
            if e.key == key and e.status == "active" and not e.confirmed:
                e.confirmed = True
                e.updated_at = datetime.now(timezone.utc).isoformat()
                self.updated_at = e.updated_at
                return e
        return None

    def reject(self, key: str) -> ContractElement | None:
        """Reject the proposed element whose :attr:`~ContractElement.key` matches.

        Retires an un-confirmed active proposal so it never gates the check.
        Non-destructive: the element is kept as ``retired`` for auditing. Returns
        the rejected element, or ``None`` when no proposal matches ``key``.
        """
        for e in self.elements:
            if e.key == key and e.proposed:
                e.status = "retired"
                e.updated_at = datetime.now(timezone.utc).isoformat()
                self.updated_at = e.updated_at
                return e
        return None


class _LearningItem(Protocol):
    """The generic shape a learnable rule must expose for :func:`lexical_match`.

    Both :class:`~workspec.profile.VoiceTrait` and :class:`ContractElement`
    satisfy this (the latter via its ``rule``/``category`` aliases), so the
    matcher stays one implementation instead of a per-collection copy.
    """

    category: str
    status: str
    rule: str


#: The concrete learnable-item type a collection holds. Bound to the duck-typed
#: shape so the matcher returns the *caller's* type (``VoiceTrait`` /
#: ``ContractElement``) rather than the erased protocol.
_ItemT = TypeVar("_ItemT", bound=_LearningItem)


class _LearningCollection(Protocol[_ItemT]):
    """A collection exposing its learnable items as ``.traits``.

    :class:`~workspec.profile.VoiceProfile` exposes this natively; the contract
    overlay aliases ``elements`` -> ``traits`` to match.
    """

    @property
    def traits(self) -> list[_ItemT]: ...


def lexical_match(profile: _LearningCollection[_ItemT], rule: str, category: str) -> _ItemT | None:
    """Lexical Jaccard dedup fallback, generic over any learning collection.

    The shared counterpart to :func:`workspec.learning.semantic.semantic_match`:
    where that matches by embedding similarity, this matches by symmetric Jaccard
    token overlap (``|a & b| / |a | b| >= 0.5``), order-independently and skipping
    ``retired`` rules. Like the semantic matcher it keys off the generic
    ``traits``/``category``/``status``/``rule`` attributes, so both the voice
    profile and the contract overlay (which aliases ``elements``->``traits`` and
    ``kind``->``category``) can delegate to one implementation instead of each
    keeping a byte-for-byte copy.
    """
    rule_tokens = set(rule.lower().split())
    for item in profile.traits:
        if item.category != category or item.status == "retired":
            continue  # retired rules are out of play (mirrors semantic_match)
        other_tokens = set(item.rule.lower().split())
        union = rule_tokens | other_tokens
        if not union:
            continue
        if len(rule_tokens & other_tokens) / len(union) >= 0.5:
            return item
    return None


def apply_delta(spec: Spec, delta: ContractDelta) -> Spec:
    """Fold a delta's *gating* elements into ``spec`` and return a new Spec.

    Only confirmed, active elements (:meth:`ContractDelta.gating_elements`) take
    effect — propose-first means an un-confirmed proposal never changes the spec.
    ``must_include`` / ``must_not_include`` elements are appended to the matching
    spec list (de-duplicated against existing lines); ``suppress_base`` elements
    drop a matching base ``must_include`` line.

    When nothing gates, the original ``spec`` is returned *by identity* — the
    no-confirmed-contract path is a true no-op, so the check is byte-identical to
    the pre-contract behavior. When something gates, a deep copy is returned and
    the input ``spec`` is never mutated.
    """
    gating = delta.gating_elements()
    if not gating:
        return spec
    out = spec.model_copy(deep=True)

    suppress = {e.rule.strip().lower() for e in gating if e.kind == "suppress_base"}
    if suppress:
        out.must_include = [
            line for line in out.must_include if line.strip().lower() not in suppress
        ]

    for e in gating:
        if e.kind == "must_include":
            _append_unique(out.must_include, e.rule)
        elif e.kind == "must_not_include":
            _append_unique(out.must_not_include, e.rule)
    return out


def _append_unique(lines: list[str], rule: str) -> None:
    """Append ``rule`` to ``lines`` unless an equal (case-insensitive) line exists."""
    norm = rule.strip().lower()
    if norm and norm not in {line.strip().lower() for line in lines}:
        lines.append(rule.strip())
