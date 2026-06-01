"""Typed data models for WorkSpec.

Two distinct schemas live here:

  * ``Spec`` — the rubric/contract a human (or AI, from examples) authors. This
    is the *input* standard: what good work must and must not contain.
  * ``Verdict`` / ``Finding`` — the *output* of a lint run. The Anthropic model
    is constrained to emit exactly this shape via structured outputs, so the
    engine never parses prose.

Keeping input and output schemas separate is deliberate: the rubric is a
durable, version-controlled artifact, while a verdict is an ephemeral judgment
about one document.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Severity(str, Enum):
    """How badly a finding blocks the work from being manager-ready."""

    BLOCKER = "blocker"  # must be fixed before the work is acceptable
    WARNING = "warning"  # should be fixed; weakens the work but not fatal
    NOTE = "note"  # minor / stylistic; informational only


class AIPolicy(BaseModel):
    """Rules about how AI may be used to produce work checked by this spec."""

    ai_allowed: bool = True
    requires_source_check: bool = Field(
        default=True,
        description="Claims of fact must cite a verifiable source.",
    )
    human_accountable: bool = Field(
        default=True,
        description="A named human owns the output regardless of how it was produced.",
    )


class Spec(BaseModel):
    """A reusable work-quality contract — the rubric.

    This is the artifact a manager authors once (or derives from examples) and
    reuses across every document of a given type.
    """

    type: str = Field(description="Stable identifier, e.g. 'decision_memo'.")
    title: str = Field(description="Human-readable name shown in reports.")
    description: str = Field(
        default="",
        description="One or two sentences on what this kind of work is for.",
    )
    must_include: list[str] = Field(
        default_factory=list,
        description="Structural elements the work MUST contain.",
    )
    must_not_include: list[str] = Field(
        default_factory=list,
        description="Anti-patterns that should cause rejection.",
    )
    acceptance_tests: list[str] = Field(
        default_factory=list,
        description="Pass/fail assertions the work must satisfy, e.g. "
        "'Each claim references a data source.'",
    )
    ai_policy: AIPolicy = Field(default_factory=AIPolicy)

    def render_for_prompt(self) -> str:
        """Flatten the spec into a compact, unambiguous block for the model."""

        def bullets(items: list[str]) -> str:
            return "\n".join(f"  - {i}" for i in items) if items else "  (none)"

        return (
            f"SPEC TYPE: {self.type}\n"
            f"TITLE: {self.title}\n"
            f"DESCRIPTION: {self.description or '(none)'}\n\n"
            f"MUST INCLUDE:\n{bullets(self.must_include)}\n\n"
            f"MUST NOT INCLUDE:\n{bullets(self.must_not_include)}\n\n"
            f"ACCEPTANCE TESTS:\n{bullets(self.acceptance_tests)}\n\n"
            f"AI POLICY:\n"
            f"  - AI allowed: {self.ai_policy.ai_allowed}\n"
            f"  - Requires source check: {self.ai_policy.requires_source_check}\n"
            f"  - Human accountable: {self.ai_policy.human_accountable}\n"
        )


# --- Output schema: what the model is constrained to return ---------------- #


class Finding(BaseModel):
    """A single problem the linter found in the work."""

    severity: Severity
    rule: str = Field(
        description="Which spec rule this relates to (verbatim from the spec, "
        "or 'acceptance_test' / 'general')."
    )
    problem: str = Field(description="What is wrong, stated concretely.")
    evidence: str = Field(
        description="A short quote or paraphrase from the work that demonstrates "
        "the problem. Empty if the problem is an omission."
    )
    suggested_fix: str = Field(description="A concrete, actionable correction.")


class Verdict(BaseModel):
    """The structured result of linting one document against one spec.

    The Anthropic model is constrained to emit exactly this shape. ``passed`` is
    the single boolean a CI gate keys off of.
    """

    passed: bool = Field(description="True only if there are zero blocker-severity findings.")
    summary: str = Field(description="One blunt sentence: is this manager-ready, and if not, why.")
    findings: list[Finding] = Field(default_factory=list)
    rewrite_prompt: str | None = Field(
        default=None,
        description="A ready-to-paste prompt the author can give an AI (or "
        "follow themselves) to bring the work up to standard. Null if it passed.",
    )

    @property
    def blockers(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == Severity.BLOCKER]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == Severity.WARNING]

    @property
    def notes(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == Severity.NOTE]
