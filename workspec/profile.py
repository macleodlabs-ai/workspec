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
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

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


class VoiceTrait(BaseModel):
    """One learned rule about how the person communicates."""

    category: Category
    rule: str = Field(description="The trait, stated as an actionable instruction.")
    provenance: Provenance = "seed"
    weight: float = Field(default=0.7, ge=0.0, le=1.0)
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


class VoiceProfile(BaseModel):
    """The whole learned profile for one person."""

    owner: str = Field(default="", description="Whose voice this is (free text).")
    traits: list[VoiceTrait] = Field(default_factory=list)
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # --- prompt rendering ------------------------------------------------- #

    def render_for_prompt(self, min_weight: float = 0.0) -> str:
        """Flatten the profile into a guidance block for the drafting model.

        Traits are grouped by category and ordered by weight so the strongest,
        edit-derived guidance leads. ``do_not`` rules are surfaced prominently.
        """
        if not self.traits:
            return "(No learned voice profile yet. Draft in a clear, professional, neutral voice.)"

        usable = [t for t in self.traits if t.weight >= min_weight]
        if not usable:
            return "(No sufficiently confident voice traits yet. Draft neutrally.)"

        usable.sort(key=lambda t: t.weight, reverse=True)

        do_not = [t for t in usable if t.category == "do_not"]
        positives = [t for t in usable if t.category != "do_not"]

        lines: list[str] = []
        if positives:
            lines.append("HOW THIS PERSON WRITES:")
            for t in positives:
                conf = "strong" if t.weight >= 0.8 else ("medium" if t.weight >= 0.55 else "weak")
                lines.append(f"  - [{t.category}, {conf}] {t.rule}")
        if do_not:
            lines.append("\nNEVER DO (hard constraints):")
            for t in do_not:
                lines.append(f"  - {t.rule}")
        return "\n".join(lines)

    # --- mutation --------------------------------------------------------- #

    def _find_similar(self, rule: str, category: str) -> VoiceTrait | None:
        """Crude dedup: same category + high token overlap == same trait.

        Overlap is measured with the symmetric Jaccard index
        (``|a & b| / |a | b|``) so dedup does not depend on insertion order.
        """
        rule_tokens = set(rule.lower().split())
        for t in self.traits:
            if t.category != category:
                continue
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
        existing = self._find_similar(rule, category)
        now = datetime.now(timezone.utc).isoformat()

        if existing is not None:
            existing.hits += 1
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
            return existing

        trait = VoiceTrait(
            category=category,
            rule=rule,
            provenance=provenance,
            weight=base_weight,
            evidence=evidence,
        )
        self.traits.append(trait)
        self.updated_at = now
        return trait


class ProfileStore:
    """Loads/saves the single global voice profile as JSON."""

    def __init__(self, profile_dir: Path | str = DEFAULT_PROFILE_DIR):
        self.dir = Path(profile_dir)
        self.path = self.dir / PROFILE_FILENAME

    def load(self) -> VoiceProfile:
        if not self.path.exists():
            return VoiceProfile()
        return VoiceProfile.model_validate_json(self.path.read_text(encoding="utf-8"))

    def save(self, profile: VoiceProfile) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")

    def exists(self) -> bool:
        return self.path.exists()
