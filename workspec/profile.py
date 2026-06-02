"""The voice profile — WorkSpec's learned model of *how this person communicates*.

This is deliberately a structured, human-readable document, not weights or
embeddings. The person can open it, read it, edit it, and delete it. The
drafting engine injects it so generated replies sound like them; the learning
loop grows it from how the person edits the drafts WorkSpec produces.

One global profile (``~/.workspec/voice_profile.json``). Each learned trait is a
short rule with provenance and a weight, so the engine can prefer
high-confidence, edit-derived traits over weaker ones.

Signal provenance, by trust (highest first):
  * ``edit``     — the person edited a draft WorkSpec produced. The diff is
                   exactly what they'd have changed: the gold signal.
  * ``feedback`` — an explicit instruction the person gave ("be warmer", "stop
                   opening with 'I hope this finds you well'").
  * ``seed``     — a built-in/default trait not yet derived from a real edit or
                   feedback signal; the lowest-trust tier and the default for a
                   freshly constructed trait.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from workspec.learning import contradiction, decay, recurrence, semantic

Provenance = Literal["edit", "feedback", "seed"]
Category = Literal[
    "tone",
    "structure",
    "phrasing",
    "salutation",
    "signoff",
    "length",
    "formatting",
    "do_not",
    "preference",
]

# How much each provenance counts when traits conflict or are ranked.
PROVENANCE_WEIGHT = {"edit": 1.0, "feedback": 0.9, "seed": 0.7}

DEFAULT_PROFILE_DIR = Path.home() / ".workspec"
PROFILE_FILENAME = "voice_profile.json"


TraitStatus = Literal["provisional", "active", "retired"]


class VoiceTrait(BaseModel):
    """One learned rule about how the person communicates."""

    category: Category
    rule: str = Field(description="The trait, stated as an actionable instruction.")
    provenance: Provenance = "seed"
    weight: float = Field(default=0.7, ge=0.0, le=1.0)
    status: TraitStatus = Field(
        default="provisional",
        description="Lifecycle stage. Traits are born 'provisional' and graduate "
        "to 'active' once they recur; conflicting/penalized traits are 'retired'.",
    )
    observations: int = Field(
        default=1,
        description="Count of distinct learn events that produced this trait. "
        "Drives recurrence-based graduation.",
    )
    last_seen: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp of the most recent reinforcement; drives decay.",
    )
    evidence: str = Field(
        default="",
        description="Short note on where this came from (e.g. the edit that "
        "produced it). Helps the person audit the profile.",
    )
    hits: int = Field(
        default=1,
        description="How many times this trait has been reinforced by signal.",
    )
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def key(self) -> str:
        """Stable identifier (``category:rule``) used to trace applied traits."""
        return f"{self.category}:{self.rule}"


class LearnMetric(BaseModel):
    """One draft→sent similarity datapoint, recorded per learn event.

    Lets the eval surface tell whether the profile is actually helping over time.
    """

    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    edit_ratio: float = Field(
        description="difflib SequenceMatcher ratio of draft vs sent (1.0 == unedited).",
        ge=0.0,
        le=1.0,
    )


class TraitStat(BaseModel):
    """One trait's snapshot for the eval surface (stored vs effective weight)."""

    key: str
    category: Category
    rule: str
    weight: float
    effective_weight: float
    observations: int


class ProfileStats(BaseModel):
    """A read-only summary of the profile for ``workspec profile --stats``."""

    total: int = Field(description="Total trait count across all statuses.")
    counts: dict[TraitStatus, int] = Field(description="Trait counts by lifecycle status.")
    top_active: list[TraitStat] = Field(
        default_factory=list,
        description="Strongest active traits by effective (decayed) weight.",
    )
    metric_count: int = Field(default=0, description="Number of recorded learn events.")
    recent_edit_ratio: float | None = Field(
        default=None,
        description="Mean draft→sent edit ratio over recent learn events (None if no metrics).",
    )
    edit_ratio_delta: float | None = Field(
        default=None,
        description="Recent mean minus older mean; positive means drafts need less editing.",
    )


class VoiceProfile(BaseModel):
    """The whole learned profile for one person."""

    owner: str = Field(default="", description="Whose voice this is (free text).")
    traits: list[VoiceTrait] = Field(default_factory=list)
    metrics: list[LearnMetric] = Field(
        default_factory=list,
        description="Per-learn draft→sent edit-ratio history, for the eval surface.",
    )
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # --- prompt rendering ------------------------------------------------- #

    def _active_by_weight(
        self, now: datetime, min_weight: float = 0.0
    ) -> list[tuple[VoiceTrait, float]]:
        """Active traits paired with their effective (decayed) weight, strongest first.

        The single source of truth for "which traits count, and how strong are
        they" — rendering, draft-tracing, and stats all build on this, so they
        can never disagree. Excludes ``provisional``/``retired`` traits and any
        whose effective weight is below ``min_weight``.
        """
        scored = [(t, decay.effective_weight(t, now)) for t in self.traits if t.status == "active"]
        scored = [(t, w) for t, w in scored if w >= min_weight]
        scored.sort(key=lambda tw: tw[1], reverse=True)
        return scored

    def render_for_prompt(self, min_weight: float = 0.0) -> str:
        """Flatten the profile into a guidance block for the drafting model.

        Only ``active`` traits are rendered (``provisional`` traits have not yet
        recurred enough to be trusted; ``retired`` traits were superseded or
        penalized). Traits are ranked and labeled by their *effective* (decayed)
        weight so the strongest, freshest guidance leads, and ``do_not`` rules
        are surfaced prominently.
        """
        if not self.traits:
            return "(No learned voice profile yet. Draft in a clear, professional, neutral voice.)"

        usable = self._active_by_weight(datetime.now(timezone.utc), min_weight)
        if not usable:
            return "(No sufficiently confident voice traits yet. Draft neutrally.)"

        do_not = [(t, eff) for t, eff in usable if t.category == "do_not"]
        positives = [(t, eff) for t, eff in usable if t.category != "do_not"]

        lines: list[str] = []
        if positives:
            lines.append("HOW THIS PERSON WRITES:")
            for t, eff in positives:
                conf = "strong" if eff >= 0.8 else ("medium" if eff >= 0.55 else "weak")
                lines.append(f"  - [{t.category}, {conf}] {t.rule}")
        if do_not:
            lines.append("\nNEVER DO (hard constraints):")
            for t, _ in do_not:
                lines.append(f"  - {t.rule}")
        return "\n".join(lines)

    def active_trait_keys(self) -> list[str]:
        """Keys of the traits ``render_for_prompt`` would surface, strongest first.

        Used by the drafter to record which traits informed a draft (for the
        negative-signal loop).
        """
        usable = self._active_by_weight(datetime.now(timezone.utc))
        return [t.key for t, _ in usable]

    # --- eval surface ----------------------------------------------------- #

    def stats(self, *, top: int = 5, recent: int = 10) -> ProfileStats:
        """Summarize the profile for the ``profile --stats`` eval surface.

        Reports trait counts by lifecycle status, the strongest active traits by
        *effective* (decayed) weight, and the recent draft→sent edit-ratio trend
        from ``metrics`` (the mean of the last ``recent`` learn events, and how it
        compares to the older ones — a rising ratio means drafts need less editing).
        """
        now = datetime.now(timezone.utc)

        counts: dict[TraitStatus, int] = {"provisional": 0, "active": 0, "retired": 0}
        for t in self.traits:
            counts[t.status] += 1

        top_active = [
            TraitStat(
                key=t.key,
                category=t.category,
                rule=t.rule,
                weight=t.weight,
                effective_weight=eff,
                observations=t.observations,
            )
            for t, eff in self._active_by_weight(now)[: max(0, top)]
        ]

        ratios = [m.edit_ratio for m in self.metrics]
        recent_ratios = ratios[-recent:] if recent > 0 else []
        recent_mean = sum(recent_ratios) / len(recent_ratios) if recent_ratios else None
        older = ratios[: -len(recent_ratios)] if recent_ratios else ratios
        older_mean = sum(older) / len(older) if older else None
        delta = (
            recent_mean - older_mean if recent_mean is not None and older_mean is not None else None
        )

        return ProfileStats(
            total=len(self.traits),
            counts=counts,
            top_active=top_active,
            metric_count=len(ratios),
            recent_edit_ratio=recent_mean,
            edit_ratio_delta=delta,
        )

    # --- mutation --------------------------------------------------------- #

    def _find_similar(self, rule: str, category: str) -> VoiceTrait | None:
        """Crude dedup: same category + high token overlap == same trait.

        Overlap is measured with the symmetric Jaccard index
        (``|a & b| / |a | b|``) so dedup does not depend on insertion order.
        """
        rule_tokens = set(rule.lower().split())
        for t in self.traits:
            if t.category != category or t.status == "retired":
                continue  # retired traits are out of play (mirrors semantic_match)
            other_tokens = set(t.rule.lower().split())
            union = rule_tokens | other_tokens
            if not union:
                continue
            jaccard = len(rule_tokens & other_tokens) / len(union)
            if jaccard >= 0.5:
                return t
        return None

    def reinforce_or_add(
        self,
        category: Category,
        rule: str,
        provenance: Provenance,
        evidence: str = "",
    ) -> VoiceTrait:
        """Add a new trait, or strengthen an existing similar one.

        Reinforcement raises weight toward the provenance ceiling and bumps the
        hit count; a stronger provenance can also upgrade a weak trait.
        """
        base_weight = PROVENANCE_WEIGHT.get(provenance, 0.5)
        # Prefer semantic dedup (paraphrases collapse to one trait); fall back to
        # the lexical Jaccard heuristic when embeddings are unavailable.
        existing = semantic.semantic_match(self, rule, category) or self._find_similar(
            rule, category
        )
        now = datetime.now(timezone.utc).isoformat()

        if existing is not None:
            existing.hits += 1
            existing.observations += 1
            existing.last_seen = now
            # Move weight a third of the way toward the (possibly higher) ceiling,
            # but never past it: a feedback-only trait must stay <= its 0.9
            # provenance ceiling rather than creeping up to tie an 'edit' trait.
            ceiling = max(existing.weight, base_weight)
            existing.weight = min(ceiling, existing.weight + (ceiling - existing.weight) / 3 + 0.05)
            if PROVENANCE_WEIGHT.get(provenance, 0) > PROVENANCE_WEIGHT.get(existing.provenance, 0):
                existing.provenance = provenance  # upgrade source of record
            if evidence:
                existing.evidence = evidence
            existing.updated_at = now
            self.updated_at = now
            # Recurrence may graduate the trait; a contradiction may retire a
            # weaker conflicting one. Only resolve contradictions once the
            # reinforced trait is itself active — a provisional, not-yet-trusted
            # trait must not retire an active one.
            recurrence.maybe_graduate(existing)
            if existing.status == "active":
                contradiction.detect_and_resolve(self, existing)
            return existing

        trait = VoiceTrait(
            category=category,
            rule=rule,
            provenance=provenance,
            weight=base_weight,
            status="provisional",
            observations=1,
            last_seen=now,
            evidence=evidence,
        )
        # A brand-new trait is born provisional with its weight held under the
        # provisional cap, so a single lucky edit cannot mint a strong rule. It
        # graduates (and unclamps) only once it recurs enough.
        recurrence.maybe_graduate(trait)
        self.traits.append(trait)
        self.updated_at = now
        return trait


class ProfileLoadError(Exception):
    """The on-disk profile could not be read or parsed.

    Raised when the (user-editable) profile JSON is missing required fields,
    malformed, or otherwise unreadable, so callers can report a clear message
    instead of leaking a raw pydantic/JSON/OS traceback.
    """


class ProfileStore:
    """Loads/saves the single global voice profile as JSON."""

    def __init__(self, profile_dir: Path | str = DEFAULT_PROFILE_DIR):
        self.dir = Path(profile_dir)
        self.path = self.dir / PROFILE_FILENAME

    def load(self) -> VoiceProfile:
        if not self.path.exists():
            return VoiceProfile()
        try:
            return VoiceProfile.model_validate_json(self.path.read_text(encoding="utf-8"))
        except (ValidationError, ValueError, OSError) as exc:
            # The profile is user-editable and save() is best-effort: a hand-edit
            # or a truncated write can leave invalid JSON. Surface a clear domain
            # error rather than an uncaught traceback.
            raise ProfileLoadError(f"could not read voice profile at {self.path}: {exc}") from exc

    def save(self, profile: VoiceProfile) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        # Write atomically: dump to a temp file in the same directory, then
        # os.replace() it onto the target so a crash/full-disk mid-write never
        # leaves a truncated profile for load() to choke on.
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def exists(self) -> bool:
        return self.path.exists()
