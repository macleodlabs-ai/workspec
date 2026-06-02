"""WorkSpec — an AI-backed work-quality linter that also drafts in your voice.

Two capabilities, one engine:
  * check — lint a piece of work against a quality contract (Spec) and get a
    structured pass/fail with specific fixes.
  * draft — generate a reply to an incoming message in the user's voice, against
    a contract, and learn that voice over time from how the user edits the
    drafts it produces.

Backed by Anthropic or any OpenAI-compatible endpoint. Local, structured,
no prose parsing.
"""

from workspec.capability import Capability, scaffolding_directive, severity_floor
from workspec.compose import ComposedContext, compose
from workspec.context import CapabilitySignal, ContextKey, SLStyle
from workspec.contract import (
    ContractDelta,
    ContractElement,
    ContractKind,
    apply_delta,
)
from workspec.draft import Draft, DraftAgent, LearnedTraits
from workspec.engine import WorkSpecAgent
from workspec.models import Finding, Severity, Spec, Verdict
from workspec.profile import ProfileStore, VoiceProfile, VoiceTrait
from workspec.providers import (
    AnthropicProvider,
    OpenAIProvider,
    VerdictProvider,
    build_provider,
)
from workspec.spec_loader import list_builtin_rubrics, load_spec
from workspec.store import ContextStore

__all__ = [
    "AnthropicProvider",
    "Capability",
    "CapabilitySignal",
    "ComposedContext",
    "ContextKey",
    "ContextStore",
    "ContractDelta",
    "ContractElement",
    "ContractKind",
    "Draft",
    "DraftAgent",
    "Finding",
    "LearnedTraits",
    "OpenAIProvider",
    "ProfileStore",
    "SLStyle",
    "Severity",
    "Spec",
    "Verdict",
    "VerdictProvider",
    "VoiceProfile",
    "VoiceTrait",
    "WorkSpecAgent",
    "apply_delta",
    "build_provider",
    "compose",
    "list_builtin_rubrics",
    "load_spec",
    "scaffolding_directive",
    "severity_floor",
]

__version__ = "1.2.0"
