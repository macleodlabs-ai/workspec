"""The manual capability dial — the owner's per-recipient judgment of standing.

Situational leadership in one knob: the owner hand-rates each recipient into one
of three buckets — ``new``, ``developing`` (the default), or ``proven`` — and
that rating tunes how WorkSpec treats updates *to* that recipient. It is the
human's call about the *relationship*, not a fact WorkSpec mines from data.

NON-NEGOTIABLE: the bucket is ALWAYS owner-set and NEVER inferred.
WorkSpec does **not** derive capability from edit ratios, draft acceptance, check
pass-rates, or any other observed signal. Auto-inference would quietly relabel a
relationship the owner never re-rated, so it is forbidden here by construction —
there is no code path that writes a bucket except an explicit owner command.

The bucket tunes TWO knobs (the two axes, softened per recipient):

  * CHECK strictness (drives the gate, via :func:`severity_floor`):
      - ``new``        — enforce the *full* required set, including every
                         confirmed contract element; omissions are blockers.
      - ``developing`` — confirmed rules still gate, but minor structural gaps
                         soften to warnings (the default, forgiving posture).
      - ``proven``     — only the high-weight non-negotiables hard-fail, and the
                         owner's learned suppressions are honored; this person
                         has earned the benefit of the doubt.
  * DRAFT scaffolding (drives the draft, via :func:`scaffolding_directive`): a
    single directive line injected into the draft prompt — maximal hand-holding
    for ``new`` down to terse, shared-context output for ``proven``.

This module owns only the value type and the two pure bucket→knob maps. Where the
bucket is read (the fold) and where it is written (the CLI) live in
:mod:`workspec.compose` and :mod:`workspec.store` / :mod:`workspec.cli`.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from workspec.context import DEFAULT_CAPABILITY, SLStyle
from workspec.models import Severity


class Capability(BaseModel):
    """The owner's manual capability rating for one recipient, persisted per scope.

    A deliberately tiny document: just the hand-set ``bucket`` and a timestamp so
    the owner can see when they last re-rated. There is no inferred field and no
    code path that sets ``bucket`` except an explicit owner command.
    """

    bucket: SLStyle = Field(
        default=DEFAULT_CAPABILITY,
        description="Owner-set standing: new | developing | proven. NEVER inferred.",
    )
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# The single directive line injected into the draft prompt per bucket. Exactly
# one line is added so the prompt's shape is stable and the only thing that
# varies between buckets is the amount of scaffolding requested.
_SCAFFOLDING: dict[SLStyle, str] = {
    "new": (
        "SCAFFOLDING (recipient is NEW): maximize scaffolding. Spell out the "
        "reasoning, label each required element explicitly, and leave nothing "
        "implicit — assume no shared context."
    ),
    "developing": (
        "SCAFFOLDING (recipient is DEVELOPING): provide high scaffolding. Make "
        "the structure clear and surface the key reasoning, but you may assume "
        "some shared context."
    ),
    "proven": (
        "SCAFFOLDING (recipient is PROVEN): keep it terse. Assume deep shared "
        "context, skip the rationale, and lead with the essentials only."
    ),
}

# The severity floor a *minor* structural gap is reported at per bucket. ``new``
# holds gaps to blocker (nothing slides); ``developing`` softens them to warning;
# ``proven`` softens them further to note. This is the knob the checker hands the
# model so its severity policy bends with the relationship — it never changes
# *what* the spec requires, only how harshly a soft miss is graded.
_SEVERITY_FLOOR: dict[SLStyle, Severity] = {
    "new": Severity.BLOCKER,
    "developing": Severity.WARNING,
    "proven": Severity.NOTE,
}


def scaffolding_directive(style: SLStyle = DEFAULT_CAPABILITY) -> str:
    """The single draft-prompt directive line for ``style``.

    Returns one line tuning how much hand-holding the draft should carry: most
    for ``new``, least for ``proven``. Unknown values fall back to the default
    bucket so a hand-edited file can never break drafting.
    """
    return _SCAFFOLDING.get(style, _SCAFFOLDING[DEFAULT_CAPABILITY])


def severity_floor(style: SLStyle = DEFAULT_CAPABILITY) -> Severity:
    """The severity a *minor* structural gap is held to under ``style``.

    Drives check strictness without touching the spec: ``new`` keeps minor gaps
    at ``blocker`` (full enforcement, fail on omissions), ``developing`` softens
    them to ``warning``, and ``proven`` softens them to ``note`` so only the
    high-weight non-negotiables hard-fail. Unknown values fall back to the
    default bucket.
    """
    return _SEVERITY_FLOOR.get(style, _SEVERITY_FLOOR[DEFAULT_CAPABILITY])
