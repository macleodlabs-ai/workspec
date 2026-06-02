"""Fold per-scope learning into one effective context for a single request.

A draft or a check happens *in a context* (a :class:`~workspec.context.ContextKey`).
The learning for that context is spread across the backoff chain — most of it in
the global scope, some of it specific to the recipient. :func:`compose` walks
that chain and folds it into one :class:`ComposedContext`: the spec to check/
draft against, the voice block to inject, and the recipient's owner-set manual
capability style (Decision 4) — which tunes two knobs: the draft's scaffolding
directive and (via :func:`workspec.capability.severity_floor`, consumed by the
engine) the check's strictness on minor structural gaps. The bucket is read from
disk, never inferred from any signal.

The fold walks the full backoff chain (Decision 1): starting from the global
scope and layering each more-specific scope (recipient, channel, project, …) on
top wherever that scope has data. With global-only data on disk the folded
profile is *exactly* the global profile, so the rendered voice block and the
applied-trait keys are byte-identical to the pre-contextual behavior.

Two axes are folded (Decision 2): VOICE (style, drives the draft) and CONTRACT
(structure, drives the check). The contract fold layers each scope's
:class:`~workspec.contract.ContractDelta` across the backoff chain and applies
only its *confirmed, gating* elements to the base spec (propose-first, Decision
5): an un-confirmed proposal never changes the spec, so with no confirmed
contract data the returned spec is the base spec unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from workspec.capability import scaffolding_directive
from workspec.context import DEFAULT_CAPABILITY, ContextKey, SLStyle
from workspec.contract import ContractDelta, apply_delta
from workspec.models import Spec
from workspec.profile import VoiceProfile
from workspec.store import ContextStore

# Mirrors the framing the drafter has always wrapped the profile in. Kept here so
# the composed voice block is byte-identical to the legacy inline string.
_VOICE_HEADER = "=== VOICE PROFILE ==="


@dataclass
class ComposedContext:
    """The effective, folded view of one context for a single request.

    Attributes
    ----------
    spec:
        The effective contract to draft/check against: the base spec with the
        folded contract delta's *confirmed, gating* elements applied. Equal to the
        base spec when no confirmed contract element exists (propose-first).
    voice_block:
        The ready-to-inject voice guidance block (header + rendered profile),
        byte-identical to the legacy inline block on global-only data.
    sl_style:
        The recipient's owner-set manual capability bucket (default
        ``developing``), resolved up the backoff chain and NEVER inferred
        (Decision 4). Tunes both knobs below.
    scaffolding_directive:
        The single draft-prompt directive line implied by ``sl_style`` — maximal
        scaffolding for ``new`` down to terse output for ``proven``. Injected by
        the drafter; empty only if a caller blanks it.
    applied_traits:
        Keys of the active voice traits that informed this context, for the
        negative-signal loop. Equal to the global profile's keys on global-only
        data.
    profile:
        The folded :class:`VoiceProfile` the block was rendered from. Exposed so
        callers can re-render or inspect without re-folding.
    contract:
        The folded :class:`~workspec.contract.ContractDelta` the effective spec
        was built from. Exposed so callers can inspect proposals / gating
        elements without re-folding.
    applied_contract:
        Keys of the confirmed, gating contract elements that shaped the effective
        spec, for tracing why the check differs from the base spec.
    """

    spec: Spec
    voice_block: str
    sl_style: SLStyle = DEFAULT_CAPABILITY
    scaffolding_directive: str = scaffolding_directive(DEFAULT_CAPABILITY)
    applied_traits: list[str] = field(default_factory=list)
    profile: VoiceProfile = field(default_factory=VoiceProfile)
    contract: ContractDelta = field(default_factory=ContractDelta)
    applied_contract: list[str] = field(default_factory=list)


def _fold_voice(store: ContextStore, key: ContextKey) -> VoiceProfile:
    """Layer per-scope profiles from least- to most-specific into one profile.

    Walks :meth:`ContextKey.backoff_chain` in reverse (global first), starting
    from the global profile and reinforcing it with each more-specific scope's
    traits. On global-only data nothing more specific exists, so the result *is*
    the global profile, preserving today's rendering exactly.
    """
    chain = list(reversed(key.backoff_chain()))  # global first, most-specific last
    merged = store.load_voice(chain[0])
    for rung in chain[1:]:
        for trait in store.load_voice(rung).traits:
            merged.graft_trait(trait)
    return merged


def _fold_contract(store: ContextStore, key: ContextKey) -> ContractDelta:
    """Layer per-scope contract deltas from least- to most-specific into one delta.

    Mirrors :func:`_fold_voice`: walk the backoff chain global-first and graft each
    more-specific scope's elements on top, so a recipient's confirmed, gating
    element overrides a conflicting global one (child overrides parent, Decision
    6). On global-only data the result *is* the global delta. Grafting preserves
    each element's earned status and ``confirmed`` flag, so propose-first holds
    across the fold.
    """
    chain = list(reversed(key.backoff_chain()))  # global first, most-specific last
    merged = store.load_contract(chain[0])
    for rung in chain[1:]:
        for element in store.load_contract(rung).elements:
            merged.graft_element(element)
    return merged


def _resolve_capability(store: ContextStore, key: ContextKey) -> SLStyle:
    """Resolve the owner-set capability bucket for ``key`` (most-specific wins).

    Walks the backoff chain from most- to least-specific and returns the first
    scope the owner has actually rated; an unrated key falls through to the
    default bucket (``developing``). The bucket is read, never derived — there is
    no signal-based inference anywhere on this path (Decision 4).
    """
    for rung in key.backoff_chain():
        if store.capability_path(rung).exists():
            return store.load_capability(rung).bucket
    return DEFAULT_CAPABILITY


def compose(store: ContextStore, key: ContextKey | None, base_spec: Spec) -> ComposedContext:
    """Resolve the effective context for ``key`` against ``base_spec``.

    ``key=None`` means the global scope — the no-``--recipient`` path that must
    reproduce legacy behavior. The returned ``voice_block`` and ``applied_traits``
    are byte-identical to the pre-contextual drafter on global-only data.

    The returned ``spec`` is ``base_spec`` with the folded contract delta's
    *confirmed, gating* elements applied (propose-first, Decision 5): with no
    confirmed contract data the spec is ``base_spec`` unchanged, so the check is
    untouched until the owner confirms a proposal.
    """
    key = key or ContextKey()
    profile = _fold_voice(store, key)
    voice_block = _VOICE_HEADER + "\n" + profile.render_for_prompt()
    contract = _fold_contract(store, key)
    sl_style = _resolve_capability(store, key)
    return ComposedContext(
        spec=apply_delta(base_spec, contract),
        voice_block=voice_block,
        sl_style=sl_style,
        scaffolding_directive=scaffolding_directive(sl_style),
        applied_traits=profile.active_trait_keys(),
        profile=profile,
        contract=contract,
        applied_contract=[e.key for e in contract.gating_elements()],
    )
