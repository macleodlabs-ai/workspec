"""Contextual addressing for WorkSpec's learning.

WorkSpec used to learn one *global* voice profile. This module introduces a
:class:`ContextKey` that names *which* context a piece of learning belongs to —
a channel, a project, a recipient — so the engine can learn how this person
writes *to a specific person* without leaking that to everyone else.

``channel`` and ``project`` are encoded in the scope id and the backoff chain;
intermediate backoff rungs between recipient and global are not yet inserted, so
a key with only ``recipient`` set backs off to ``global`` and nothing else.

The two learned axes that ride on these keys:

  * VOICE  — *how* this person writes (style); drives drafting.
  * CONTRACT — *what* a proper update from this person contains; drives check.

This module defines only the addressing types and the lightweight style/
capability value types. The stores and the fold live in
:mod:`workspec.store` and :mod:`workspec.compose`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# The manual capability dial (Decision 4): a 3-bucket judgment the owner sets
# per recipient. It is NEVER inferred from edit ratios or any other signal.
SLStyle = Literal["new", "developing", "proven"]

# The default bucket for a recipient the owner has not rated yet.
DEFAULT_CAPABILITY: SLStyle = "developing"

# The global scope id — the bucket every key backs off to, and the home of the
# migrated legacy profile.
GLOBAL_SCOPE = "global"


@dataclass(frozen=True)
class CapabilitySignal:
    """The lightweight, in-memory capability value type (Decision 4).

    Carries only the owner-set :attr:`bucket`. It is *never* inferred from edit
    ratios, draft acceptance, or any other observed signal — the bucket is
    whatever the owner explicitly set, defaulting to :data:`DEFAULT_CAPABILITY`
    for an unrated recipient. The persisted twin is
    :class:`workspec.capability.Capability`; this is the value type the engine
    and tests pass around without touching the store.
    """

    bucket: SLStyle = DEFAULT_CAPABILITY


@dataclass(frozen=True)
class ContextKey:
    """Names the context a learned trait or contract requirement belongs to.

    All four fields are optional. A field left ``None`` means "not scoped on
    this axis". ``channel`` and ``project`` are accepted and folded into the
    :attr:`scope_id` ordering alongside ``recipient`` and the implicit global
    layer.
    """

    channel: str | None = None
    project: str | None = None
    recipient: str | None = None

    @property
    def scope_id(self) -> str:
        """A stable, filesystem-safe id for this exact key.

        The empty key (all fields ``None``) is the :data:`GLOBAL_SCOPE`. Any
        populated key is encoded as a sorted, ``axis=value`` join so two keys
        that differ only in field *order* can never address different files.
        Most-specific resolution is handled by :meth:`backoff_chain`, not here.
        """
        parts = [
            f"{axis}={value}"
            for axis, value in (
                ("channel", self.channel),
                ("project", self.project),
                ("recipient", self.recipient),
            )
            if value
        ]
        return GLOBAL_SCOPE if not parts else "__".join(parts)

    def is_global(self) -> bool:
        """True when no axis is set — i.e. this key *is* the global scope."""
        return self.channel is None and self.project is None and self.recipient is None

    def backoff_chain(self) -> list[ContextKey]:
        """Keys from most- to least-specific, ending at the global scope.

        The fold in :func:`workspec.compose.compose` walks this chain so a
        specific layer overrides a general one. With only a ``recipient`` set
        this yields ``[recipient_key, global_key]``; with no fields set it
        yields just ``[global_key]``. Intermediate ``channel``/``project`` rungs
        between recipient and global are not yet inserted.
        """
        chain: list[ContextKey] = []
        if not self.is_global():
            chain.append(self)
        chain.append(ContextKey())  # the global scope always anchors the chain
        return chain
