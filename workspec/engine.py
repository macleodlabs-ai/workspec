"""The WorkSpec lint engine.

Takes a Spec and a piece of work, returns a typed Verdict. The engine owns the
judgment prompt; it delegates the actual call to a provider (Anthropic or
OpenAI-compatible) and never touches an SDK directly.

Scope discipline: the system prompt judges *structural* quality — presence of
owner, evidence, decision, source, risk, next step — not whether the strategy is
correct. Over-claiming "AI can tell you if this plan is right" is how these tools
lose trust.
"""

from __future__ import annotations

from workspec._base import ProviderBackedAgent
from workspec.capability import severity_floor
from workspec.compose import _fold_contract, _resolve_capability
from workspec.context import ContextKey, SLStyle
from workspec.contract import apply_delta
from workspec.models import Severity, Spec, Verdict
from workspec.providers import VerdictProvider
from workspec.store import ContextStore

# Convenience model ids.
DEFAULT_MODEL = "claude-opus-4-8"  # sharpest judgment (Anthropic)
DEFAULT_OPENAI_MODEL = "gpt-5.5"

_SYSTEM_PROMPT = """\
You are WorkSpec, a work-quality linter. You judge whether a piece of knowledge \
work meets an explicit, author-provided standard before it reaches a busy \
manager. You are not the manager and you are not the author — you are the \
pre-flight check.

Your job is to catch STRUCTURAL quality problems:
  - missing owner, decision, deadline, evidence, source, risk, or next step
  - vague or unsupported claims ("on track", "aligned", "soon") with no backing
  - fake precision, hollow filler, summaries that contain no actual decision
  - claims of fact that cite no source when the spec requires sourcing
  - failure to satisfy the spec's explicit acceptance tests

You must NOT:
  - judge whether the underlying strategy or decision is *correct* — you cannot
    know that, and pretending to destroys trust. Judge structure and rigor.
  - invent problems to seem thorough. If the work is genuinely ready, say so.
  - rewrite the work yourself. Diagnose, then provide a rewrite *prompt*.

Severity rules:
  - blocker: violates a MUST INCLUDE / MUST NOT INCLUDE rule or fails an
    acceptance test. The work cannot pass with any blocker present.
  - warning: weakens the work but is not a hard violation.
  - note: minor or stylistic.

Set passed=true ONLY if there are zero blocker findings.
Be blunt and specific. A useful, brutal diagnosis is the whole point.
"""

_USER_TEMPLATE = """\
Lint the following work against the spec below.

=== SPEC ===
{spec}

=== WORK TO CHECK ===
{work}

=== END WORK ===

{strictness}

Return your verdict. For every finding, quote or paraphrase the specific part of \
the work that demonstrates the problem (leave evidence empty only for pure \
omissions). If the work fails, include a rewrite_prompt the author can paste \
into an AI assistant to fix it. If it passes cleanly, set rewrite_prompt to null.
"""

# Per-bucket strictness clause injected into the lint prompt. It bends only the
# severity of MINOR structural gaps; a hard MUST/MUST-NOT violation or a failed
# acceptance test is always a blocker regardless of bucket. The floor comes from
# the owner-set capability dial (Decision 4) — it is never inferred.
_STRICTNESS: dict[Severity, str] = {
    Severity.BLOCKER: (
        "STRICTNESS (recipient is NEW): enforce the FULL required set, including "
        "every learned/confirmed element. Treat any omission of a required "
        "element as a blocker — fail on missing pieces."
    ),
    Severity.WARNING: (
        "STRICTNESS (recipient is DEVELOPING): confirmed required elements still "
        "gate, but report a MINOR structural gap (a soft, non-essential omission) "
        "as at most a warning, not a blocker."
    ),
    Severity.NOTE: (
        "STRICTNESS (recipient is PROVEN): only the high-weight non-negotiables "
        "hard-fail. Honor the owner's learned suppressions and report any MINOR "
        "structural gap as at most a note. Give this person the benefit of the "
        "doubt."
    ),
}


def _strictness_clause(style: SLStyle) -> str:
    """The lint-prompt strictness clause implied by capability ``style``."""
    return _STRICTNESS[severity_floor(style)]


class WorkSpecAgent(ProviderBackedAgent):
    """Lints work against a spec using a pluggable provider.

    Construct with a provider name (``"anthropic"`` / ``"openai"``) or a ready
    ``VerdictProvider`` instance::

        WorkSpecAgent(provider="anthropic", model="claude-opus-4-8")
        WorkSpecAgent(provider="openai", base_url="http://localhost:11434/v1",
                      model="llama3.1", api_key="ollama")
    """

    def __init__(
        self,
        provider: VerdictProvider | str = "anthropic",
        model: str | None = None,
        store: ContextStore | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 4096,
    ) -> None:
        super().__init__(
            provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens,
        )
        self.store = store

    def check(self, spec: Spec, work: str, key: ContextKey | None = None) -> Verdict:
        """Lint ``work`` against ``spec`` in the context ``key`` and return a Verdict.

        ``key`` names the context whose learned contract and capability dial
        shape the check: the owner-confirmed contract is folded into the effective
        spec and the owner-set capability bucket is resolved, which tunes how
        harshly a *minor* structural gap is graded (Decision 4 — read, never
        inferred). With ``key=None`` (the no-recipient path) the spec is the base
        spec unchanged and the bucket is the default ``developing``, so the check
        behaves exactly as before.

        Unlike the draft path, the check never reads the voice profile, so it folds
        the contract and capability axes directly instead of calling
        :func:`compose`, skipping the wasted voice fold on every lint.
        """
        if not work.strip():
            raise ValueError("Work to check is empty.")
        store = self.store or ContextStore()
        key = key or ContextKey()
        spec = apply_delta(spec, _fold_contract(store, key))
        sl_style = _resolve_capability(store, key)
        user_prompt = _USER_TEMPLATE.format(
            spec=spec.render_for_prompt(),
            work=work.strip(),
            strictness=_strictness_clause(sl_style),
        )
        return self.provider.get_verdict(_SYSTEM_PROMPT, user_prompt)
