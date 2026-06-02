"""End-to-end voice-learning scenarios across channels and people.

Drives the full ``DraftAgent`` loop (draft + learn_from_edit: provenance, dedup,
recurrence graduation, contradiction, negative signal, metrics, persistence) with
a scripted provider standing in for the LLM, so the *learning mechanics* are
exercised deterministically. Each scenario is a real-shaped conversation in one
of three channel formats — **email**, **Slack**, **WhatsApp** — to a specific
person, so the suite also covers:

* **Focus** — a draft is built around the message it is replying to (that
  submission, that person), not some other thread.
* **Context separation** — generalizable voice reaches the *shared* layer only by
  an earned cross-recipient *promotion* (Decision 6); an un-promoted
  recipient-specific trait must NOT leak into a draft for a different person, and
  per-message content from one thread never leaks anywhere.

The contextual scenarios (Part B onward) drive the per-scope
:class:`~workspec.store.ContextStore` so the backoff fold, promotion, the
confirmed-contract gate, and the manual capability dial are all exercised through
the real draft/check loop. The single-profile mechanics scenarios (Part A) keep
using the legacy :class:`~workspec.profile.ProfileStore` so the unchanged
lifecycle machinery stays covered end to end.

Semantic dedup is disabled here (it has its own suite) so results are hermetic;
a final integration-marked scenario exercises real semantic dedup.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from workspec.capability import Capability, scaffolding_directive, severity_floor
from workspec.context import ContextKey
from workspec.contract_extractor import ExtractedContract, ExtractedElement
from workspec.draft import DraftAgent, ExtractedTrait, GenerationDraft, LearnedTraits
from workspec.engine import WorkSpecAgent, _strictness_clause
from workspec.learning import decay
from workspec.learning.recurrence import GRADUATION_OBSERVATIONS, PROVISIONAL_WEIGHT_CAP
from workspec.models import Finding, Severity, Spec, Verdict
from workspec.profile import ProfileStore, VoiceProfile, VoiceTrait
from workspec.providers import VerdictProvider
from workspec.store import ContextStore

# --------------------------------------------------------------------------- #
# Channel formats & people
# --------------------------------------------------------------------------- #


def email(to: str, subject: str, body: str, signoff: str = "Best,\nAlex") -> str:
    """Render an email-shaped message."""
    first = to.split()[0]
    return f"To: {to}\nSubject: {subject}\n\nHi {first},\n\n{body}\n\n{signoff}"


def slack(handle: str, body: str) -> str:
    """Render a Slack-shaped message."""
    return f"[#team Slack] @{handle}: {body}"


def whatsapp(name: str, body: str) -> str:
    """Render a WhatsApp-shaped message."""
    return f"[WhatsApp → {name}] {body}"


DANA = "Dana Okafor"  # manager — email, formal
RAJ = "Raj"  # teammate — Slack, casual
SAM = "Sam"  # friend — WhatsApp, very casual
LIN = "Ms. Lin"  # client — email, formal
PRIYA = "Priya"  # teammate — Slack


# --------------------------------------------------------------------------- #
# Test doubles & helpers
# --------------------------------------------------------------------------- #


class ScriptedProvider(VerdictProvider):
    """Returns scripted traits/draft/contract and records every prompt — no network."""

    name = "scripted"

    def __init__(self) -> None:
        self.learned = LearnedTraits(traits=[])
        self.draft_result = GenerationDraft(draft="Thanks — confirming now.")
        self.extracted_contract = ExtractedContract(elements=[])
        self.calls: list[dict[str, Any]] = []

    def get_structured(self, system_prompt: str, user_prompt: str, schema):  # type: ignore[override]
        self.calls.append({"system": system_prompt, "user": user_prompt, "schema": schema})
        if schema is LearnedTraits:
            return self.learned
        if schema is GenerationDraft:
            return self.draft_result
        if schema is ExtractedContract:
            return self.extracted_contract
        raise AssertionError(f"unexpected schema {schema!r}")


# A required structural element used by the composed end-to-end scenario: the
# contract teaches that updates to this recipient must name a next step.
_NEXT_STEP_RULE = "State an explicit next step."


class _SpecAwareScriptedProvider(ScriptedProvider):
    """ScriptedProvider whose check fails iff the effective spec gates the next step.

    Lets the composed scenario prove that a *confirmed* contract element actually
    tightens the gate: when the rendered spec carries ``_NEXT_STEP_RULE``, work
    that omits a next step earns a blocker; otherwise the check passes.
    """

    def get_verdict(self, system_prompt: str, user_prompt: str) -> Verdict:
        self.calls.append({"system": system_prompt, "user": user_prompt, "schema": Verdict})
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


@pytest.fixture(autouse=True)
def _no_semantic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force lexical dedup so scenarios are hermetic (semantic has its own suite)."""
    monkeypatch.setattr("workspec.profile.semantic.semantic_match", lambda *a, **k: None)


def _lt(*traits: tuple[str, str]) -> LearnedTraits:
    return LearnedTraits(
        traits=[ExtractedTrait(category=c, rule=r, evidence="from edit") for c, r in traits]
    )


def _agent(tmp_path: Path) -> tuple[DraftAgent, ScriptedProvider, ProfileStore]:
    provider = ScriptedProvider()
    store = ProfileStore(tmp_path / ".workspec")
    return DraftAgent(provider=provider, profile_store=store), provider, store


def _ctx_agent(tmp_path: Path) -> tuple[DraftAgent, ScriptedProvider, ContextStore]:
    """A DraftAgent over a per-scope ContextStore (the contextual learning path)."""
    provider = ScriptedProvider()
    store = ContextStore(tmp_path / ".workspec")
    return DraftAgent(provider=provider, store=store), provider, store


def _graduate_ctx(
    agent: DraftAgent,
    provider: ScriptedProvider,
    *,
    key: ContextKey,
    rule: str,
    category: str,
    pairs: list[tuple[str, str]],
) -> None:
    """Reinforce one trait in a specific context until it graduates in that scope."""
    for draft_text, sent_text in pairs:
        provider.learned = _lt((category, rule))
        agent.learn_from_edit(draft=draft_text, sent=sent_text, key=key)


def _learn(
    agent: DraftAgent,
    provider: ScriptedProvider,
    *,
    draft: str,
    sent: str,
    traits: tuple[tuple[str, str], ...] = (),
    feedback: str = "",
    applied: list[str] | None = None,
) -> list[VoiceTrait]:
    provider.learned = _lt(*traits)
    return agent.learn_from_edit(draft=draft, sent=sent, feedback=feedback, applied_traits=applied)


def _graduate(
    agent: DraftAgent,
    provider: ScriptedProvider,
    rule: str,
    category: str,
    pairs: list[tuple[str, str]],
) -> None:
    """Reinforce one trait across several (draft, sent) contexts until it graduates."""
    for draft_text, sent_text in pairs:
        _learn(agent, provider, draft=draft_text, sent=sent_text, traits=((category, rule),))


def _trait(profile: VoiceProfile, rule_part: str) -> VoiceTrait | None:
    return next((t for t in profile.traits if rule_part.lower() in t.rule.lower()), None)


_SPEC = Spec(type="email_reply", title="Reply", must_include=["a clear answer"])


# =========================================================================== #
# Part A — learning mechanics, one scenario per channel/person
# =========================================================================== #


def test_email_to_manager_single_edit_is_provisional(tmp_path: Path) -> None:
    """EMAIL → manager Dana. One formal→casual edit is learned but stays provisional."""
    agent, provider, store = _agent(tmp_path)
    draft = email(DANA, "Q3 numbers", "Please find the figures attached for your review.")
    sent = email(DANA, "Q3 numbers", "Numbers attached — shout with questions.", "Cheers, Alex")
    _learn(
        agent,
        provider,
        draft=draft,
        sent=sent,
        traits=(("tone", "Be warm and direct, not stiff"),),
    )
    t = _trait(store.load(), "warm")
    assert t is not None and t.status == "provisional"
    assert t.weight <= PROVISIONAL_WEIGHT_CAP
    assert "warm" not in store.load().render_for_prompt()


def test_slack_to_teammate_recurrence_graduates(tmp_path: Path) -> None:
    """SLACK → teammate Raj. The same 'be concise' edit thrice graduates to active."""
    agent, provider, store = _agent(tmp_path)
    pairs = [
        (
            slack(RAJ, "hey just wanted to quickly circle back on the deploy"),
            slack(RAJ, "deploy update:"),
        ),
        (slack(RAJ, "i wanted to reach out about the flaky test"), slack(RAJ, "flaky test:")),
        (slack(RAJ, "just following up to touch base on the rollout"), slack(RAJ, "rollout:")),
    ]
    _graduate(agent, provider, "Be concise; cut filler openers", "length", pairs)
    t = _trait(store.load(), "concise")
    assert t is not None and t.status == "active"
    assert t.observations >= GRADUATION_OBSERVATIONS
    assert "concise" in store.load().render_for_prompt().lower()


def test_whatsapp_to_friend_long_edit_is_edit_provenance(tmp_path: Path) -> None:
    """WHATSAPP → friend Sam. A long verbose→short edit is the gold 'edit' signal."""
    agent, provider, store = _agent(tmp_path)
    draft = whatsapp(
        SAM,
        "Hello Sam, I hope you are doing well. I wanted to check whether you would be "
        "available to meet for dinner sometime next week, perhaps Tuesday or Wednesday?",
    )
    sent = whatsapp(SAM, "din next wk? tue/wed?")
    applied = _learn(
        agent,
        provider,
        draft=draft,
        sent=sent,
        traits=(("length", "Keep it ultra short with friends"),),
    )
    assert len(applied) == 1 and applied[0].provenance == "edit"
    assert store.load().metrics[-1].edit_ratio < 1.0


def test_email_to_client_feedback_only(tmp_path: Path) -> None:
    """EMAIL → client Ms. Lin. No diff, only explicit feedback → 'feedback' provenance."""
    agent, provider, _store = _agent(tmp_path)
    msg = email(LIN, "Proposal", "The revised proposal is attached for your consideration.")
    applied = _learn(
        agent,
        provider,
        draft=msg,
        sent=msg,
        feedback="Slightly stiff for this client — warm it up",
        traits=(("tone", "Warm the tone for clients"),),
    )
    assert len(applied) == 1 and applied[0].provenance == "feedback"


def test_slack_to_teammate_contradiction_retires_loser(tmp_path: Path) -> None:
    """SLACK → teammate Priya. Learning 'be terse' after 'be warm' retires the loser."""
    agent, provider, store = _agent(tmp_path)
    warm_pairs = [
        (slack(PRIYA, "hi"), slack(PRIYA, "hey! great to hear from you :)"))
    ] * GRADUATION_OBSERVATIONS
    _graduate(agent, provider, "Be warm and friendly", "tone", warm_pairs)
    terse_pairs = [
        (slack(PRIYA, "hey! great to hear from you :)"), slack(PRIYA, "noted."))
    ] * GRADUATION_OBSERVATIONS
    _graduate(agent, provider, "Be terse", "tone", terse_pairs)

    profile = store.load()
    warm, terse = _trait(profile, "warm"), _trait(profile, "terse")
    assert warm is not None and terse is not None
    assert {warm.status, terse.status} == {"active", "retired"}


def test_email_to_manager_negative_signal_penalizes(tmp_path: Path) -> None:
    """EMAIL → manager Dana. A bullet-points trait the person edits back into prose is penalized."""
    agent, provider, store = _agent(tmp_path)
    pairs = [
        (
            email(DANA, "Status", "Here is the status."),
            email(DANA, "Status", "Status, in bullet points:\n- shipped\n- tested"),
        )
    ] * GRADUATION_OBSERVATIONS
    _graduate(agent, provider, "Use bullet points for updates", "formatting", pairs)
    bullets = _trait(store.load(), "bullet")
    assert bullets is not None and bullets.status == "active"
    before = bullets.weight

    _learn(
        agent,
        provider,
        draft=email(DANA, "Status", "Status, in bullet points:\n- shipped\n- tested"),
        sent=email(DANA, "Status", "Status: shipped and tested, written as prose."),
        applied=[bullets.key],
    )
    after = _trait(store.load(), "bullet")
    assert after is not None and after.weight < before


def test_whatsapp_to_friend_do_not_trait_surfaces(tmp_path: Path) -> None:
    """WHATSAPP → friend Sam. A 'never open with X' trait renders in the NEVER DO block."""
    agent, provider, store = _agent(tmp_path)
    pairs = [
        (whatsapp(SAM, "I hope this message finds you well. Quick q:"), whatsapp(SAM, "quick q:"))
    ] * GRADUATION_OBSERVATIONS
    _graduate(agent, provider, "Never open with 'I hope this finds you well'", "do_not", pairs)
    rendered = store.load().render_for_prompt()
    assert "NEVER DO" in rendered and "finds you well" in rendered


def test_slack_noop_learns_nothing(tmp_path: Path) -> None:
    """SLACK → Raj. Identical draft/sent, no feedback → nothing learned or persisted."""
    agent, provider, store = _agent(tmp_path)
    msg = slack(RAJ, "sounds good, thanks!")
    assert _learn(agent, provider, draft=msg, sent=msg, traits=(("tone", "x"),)) == []
    assert not store.exists()


def test_decay_drops_stale_trait() -> None:
    """A long-stale active trait loses effective weight and falls below the render floor."""
    old = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
    fresh = VoiceTrait(
        category="tone",
        rule="Be warm",
        status="active",
        weight=0.9,
        last_seen=datetime.now(timezone.utc).isoformat(),
    )
    stale = VoiceTrait(
        category="length", rule="Be brief", status="active", weight=0.9, last_seen=old
    )
    profile = VoiceProfile(traits=[fresh, stale])
    assert decay.effective_weight(stale) < 0.1 * stale.weight
    rendered = profile.render_for_prompt(min_weight=0.3)
    assert "Be warm" in rendered and "Be brief" not in rendered


def test_email_to_client_lexical_dedup_merges_paraphrase(tmp_path: Path) -> None:
    """EMAIL → client Ms. Lin. A lexically-overlapping paraphrase merges, not duplicates."""
    agent, provider, store = _agent(tmp_path)
    draft = email(LIN, "Update", "I am writing to provide you with a comprehensive update.")
    sent = email(LIN, "Update", "Quick update below.")
    _learn(agent, provider, draft=draft, sent=sent, traits=(("length", "Keep replies short"),))
    _learn(
        agent,
        provider,
        draft=draft,
        sent=sent,
        traits=(("length", "Keep replies short and tight"), ("signoff", "Sign off with 'Cheers'")),
    )
    profile = store.load()
    assert len([t for t in profile.traits if t.category == "length"]) == 1
    assert _trait(profile, "Cheers") is not None


# =========================================================================== #
# Part B — focus & context separation across people/channels
# =========================================================================== #


def test_draft_is_focused_on_its_own_thread(tmp_path: Path) -> None:
    """Each draft is built around its own incoming message — no cross-thread bleed.

    Drafting to Dana (email, budget) and to Sam (WhatsApp, dinner) must each carry
    only that thread's submission into the model prompt.
    """
    agent, provider, _ = _agent(tmp_path)
    dana_msg = email(DANA, "Q3 budget", "Can you confirm the Q3 budget figure by Friday?")
    sam_msg = whatsapp(SAM, "still on for dinner saturday? 🍜")

    agent.draft(_SPEC, dana_msg)
    dana_prompt = provider.calls[-1]["user"]
    agent.draft(_SPEC, sam_msg)
    sam_prompt = provider.calls[-1]["user"]

    assert "Q3 budget" in dana_prompt and "dinner" not in dana_prompt  # focused on Dana's thread
    assert "dinner" in sam_prompt and "Q3 budget" not in sam_prompt  # focused on Sam's thread


def test_voice_promotes_across_recipients_to_shared_layer(tmp_path: Path) -> None:
    """A trait that graduates *independently* for three recipients is PROMOTED to global.

    Inverts the old global-generalization premise (Decision 6): generalization is
    not pooling. A recipient-scope trait stays in its own scope until the *same*
    trait has independently graduated across enough distinct recipients — only then
    is it earned into the shared (global) layer. Here the same 'Cheers' sign-off
    graduates for Dana, Raj and Sam, each in their own scope, and the cross-
    recipient recurrence promotes it to global.
    """
    agent, provider, store = _ctx_agent(tmp_path)
    rule = "Sign off with a brief 'Cheers'"
    per_recipient = {
        ContextKey(recipient="dana"): [
            (
                email(DANA, "Re: plan", f"Many thanks for your guidance on this. {i}"),
                email(DANA, "Re: plan", "Thanks! Cheers, Alex"),
            )
            for i in range(GRADUATION_OBSERVATIONS)
        ],
        ContextKey(recipient="raj"): [
            (slack(RAJ, f"appreciate the help on the bug {i}"), slack(RAJ, "thanks! cheers"))
            for i in range(GRADUATION_OBSERVATIONS)
        ],
        ContextKey(recipient="sam"): [
            (whatsapp(SAM, f"thank you so much for sorting that {i}"), whatsapp(SAM, "thanks!! 🙌"))
            for i in range(GRADUATION_OBSERVATIONS)
        ],
    }
    for key, pairs in per_recipient.items():
        _graduate_ctx(agent, provider, key=key, rule=rule, category="signoff", pairs=pairs)

    # Earned into the shared (global) layer, active, with a promotion audit note.
    promoted = _trait(store.load_voice(ContextKey()), "Cheers")
    assert promoted is not None and promoted.status == "active"
    assert "promoted" in promoted.evidence


def test_unpromoted_recipient_trait_does_not_leak_and_no_thread_bleed(tmp_path: Path) -> None:
    """An un-promoted recipient-specific trait must NOT surface for another recipient.

    Inverts the old leak premise (Decision 6): a sign-off graduated for Dana ALONE
    is below the cross-recipient promotion bar, so it stays in Dana's scope. When we
    then draft a dinner WhatsApp to Sam, that recipient-specific trait must NOT
    appear in Sam's prompt. The no-thread-CONTENT-bleed guarantees are unchanged:
    Sam's own message carries through, and none of Dana's budget content does.
    """
    agent, provider, store = _ctx_agent(tmp_path)
    dana = ContextKey(recipient="dana")
    pairs = [
        (
            email(DANA, "Q3 budget approval", f"Please approve the Q3 budget of $1.2M. {i}"),
            email(DANA, "Q3 budget approval", "Approved? Cheers, Alex"),
        )
        for i in range(GRADUATION_OBSERVATIONS)
    ]
    _graduate_ctx(
        agent, provider, key=dana, rule="Sign off with 'Cheers'", category="signoff", pairs=pairs
    )
    # Graduated in Dana's own scope, but a single recipient is below the promotion
    # bar, so it never reached the shared (global) layer.
    assert _trait(store.load_voice(dana), "Cheers") is not None
    assert _trait(store.load_voice(ContextKey()), "Cheers") is None

    agent.draft(_SPEC, whatsapp(SAM, "pizza or sushi tonight?"), key=ContextKey(recipient="sam"))
    prompt = provider.calls[-1]["user"]
    assert "Cheers" not in prompt  # un-promoted recipient trait does NOT leak to Sam
    assert "pizza or sushi" in prompt  # still focused on Sam's actual message
    assert "budget" not in prompt.lower() and "dana" not in prompt.lower()  # no thread bleed
    assert "$1.2M" not in prompt


# =========================================================================== #
# Part D — the whole thing composes: voice + contract + capability in one loop
# =========================================================================== #


def test_recipient_voice_contract_and_capability_compose_in_one_loop(tmp_path: Path) -> None:
    """One recipient's learned voice, a confirmed contract element, and a manual
    capability bucket all compose into a single check and a single draft.

    End-to-end across both agents on the contextual store, for recipient *priya*:

      1. VOICE — a 'be concise' edit graduates in priya's scope and is rendered
         into the draft prompt's voice block.
      2. CONTRACT — a 'state an explicit next step' element graduates to a proposal
         and the owner confirms it; it now gates the check (propose-first).
      3. CAPABILITY — the owner rates priya ``new`` (manual dial, never inferred),
         so the check carries the strict ``new`` clause and the draft carries the
         ``new`` scaffolding directive.

    The folded check then FAILS work that omits the confirmed next step, and the
    folded draft prompt carries priya's voice plus the ``new`` scaffolding — all
    from one ``ContextKey``.
    """
    store = ContextStore(tmp_path / ".workspec")
    provider = _SpecAwareScriptedProvider()
    agent = DraftAgent(provider=provider, store=store)
    key = ContextKey(recipient="priya")

    # 1. VOICE: graduate a concise trait in priya's own scope.
    voice_pairs = [
        (slack(PRIYA, f"just wanted to quickly circle back on item {i}"), slack(PRIYA, "update:"))
        for i in range(GRADUATION_OBSERVATIONS)
    ]
    _graduate_ctx(
        agent,
        provider,
        key=key,
        rule="Be concise; cut filler openers",
        category="length",
        pairs=voice_pairs,
    )
    assert _trait(store.load_voice(key), "concise") is not None

    # 2. CONTRACT: graduate + confirm a 'next step' element so it gates the check.
    provider.extracted_contract = ExtractedContract(
        elements=[ExtractedElement(kind="must_include", rule=_NEXT_STEP_RULE)]
    )
    for _ in range(GRADUATION_OBSERVATIONS):
        agent.learn_contract_from_edit(
            draft="Here is the status.",
            sent="Here is the status. Next, I will ship Friday.",
            key=key,
        )
    delta = store.load_contract(key)
    proposal = delta.proposals()[0]
    assert proposal.rule == _NEXT_STEP_RULE  # graduated to a proposal, not yet gating
    assert delta.confirm(proposal.key) is not None
    store.save_contract(key, delta)

    # 3. CAPABILITY: the owner manually rates priya 'new' (strictest posture).
    store.save_capability(key, Capability(bucket="new"))

    base_spec = Spec(type="status_update", title="Status", must_include=["a clear decision"])

    # CHECK: work has a decision but no next step. The confirmed contract element
    # gates it, and the 'new' bucket holds omissions to blocker -> the check fails.
    checker = WorkSpecAgent(provider=provider, store=store)
    verdict = checker.check(base_spec, "We decided to ship. Owner: me.", key=key)
    assert not verdict.passed
    assert any(f.rule == _NEXT_STEP_RULE for f in verdict.blockers)
    check_prompt = provider.calls[-1]["user"]
    assert _NEXT_STEP_RULE in check_prompt  # confirmed contract reached the spec
    assert _strictness_clause("new") in check_prompt  # 'new' capability tightened it

    # DRAFT: the same key composes priya's voice block and the 'new' scaffolding.
    drafted = agent.draft(base_spec, slack(PRIYA, "where are we on the deploy?"), key=key)
    draft_prompt = provider.calls[-1]["user"]
    assert "concise" in draft_prompt.lower()  # priya's learned voice is injected
    assert scaffolding_directive("new") in draft_prompt  # 'new' scaffolding directive
    assert "deploy" in draft_prompt  # focused on priya's actual message
    assert drafted.used_profile  # a folded profile informed the draft


# =========================================================================== #
# Part E — migration / back-compat: legacy global profile, no --recipient
# =========================================================================== #


def test_legacy_global_profile_no_recipient_behaves_like_v1(tmp_path: Path) -> None:
    """A legacy ``voice_profile.json`` + no ``--recipient`` reproduces v1.1 exactly.

    Decision 8 (identical-behavior migration): the legacy single-file profile is
    relocated into the global scope on first access, and with no recipient key
    everything resolves to global. The contextual draft over a ``ContextStore``
    then renders a prompt byte-identical to the legacy single-``ProfileStore``
    draft, and the check carries the unchanged base spec and the default
    ``developing`` strictness — i.e. nothing the owner did not opt into.
    """
    # Seed a legacy single-file profile at the base-dir root (the v1 layout).
    base = tmp_path / ".workspec"
    legacy_profile = VoiceProfile(
        traits=[
            VoiceTrait(
                category="signoff",
                rule="Sign off with 'Cheers'",
                provenance="edit",
                weight=0.9,
                status="active",
            )
        ]
    )
    ProfileStore(base).save(legacy_profile)

    spec = Spec(type="email_reply", title="Reply", must_include=["a clear answer"])
    submission = whatsapp(SAM, "pizza or sushi tonight?")

    # v1.1 path: the legacy single ProfileStore (a *copy* of the same profile, since
    # the migration relocates the original file out of the root).
    legacy_dir = tmp_path / "legacy"
    ProfileStore(legacy_dir).save(legacy_profile)
    v1_provider = ScriptedProvider()
    DraftAgent(provider=v1_provider, profile_store=ProfileStore(legacy_dir)).draft(spec, submission)

    # Contextual path: a ContextStore over the legacy base, no --recipient (key=None).
    ctx_provider = ScriptedProvider()
    ctx_agent = DraftAgent(provider=ctx_provider, store=ContextStore(base))
    drafted = ctx_agent.draft(spec, submission)

    # The legacy file was migrated into the global scope, losslessly.
    store = ContextStore(base)
    assert _trait(store.load_voice(ContextKey()), "Cheers") is not None
    assert not (base / "voice_profile.json").exists()  # relocated, not copied

    # Byte-identical draft prompt and applied traits -> identical drafting behavior.
    assert ctx_provider.calls[-1]["user"] == v1_provider.calls[-1]["user"]
    assert "Cheers" in ctx_provider.calls[-1]["user"]
    assert drafted.used_profile

    # And the check resolves to the base spec with the default 'developing'
    # strictness -> the gate is unchanged from v1.1.
    check_provider = _SpecAwareScriptedProvider()
    WorkSpecAgent(provider=check_provider, store=store).check(
        spec, "We shipped it. Owner: me. Next: monitor."
    )
    check_prompt = check_provider.calls[-1]["user"]
    assert _strictness_clause("developing") in check_prompt
    assert severity_floor("developing") is Severity.WARNING
    # No confirmed contract overlay exists, so the base spec is unchanged: the
    # rendered spec carries only the base must-include line, not any learned rule.
    assert _NEXT_STEP_RULE not in check_prompt


# =========================================================================== #
# Part C — live semantic dedup (real embeddings)
# =========================================================================== #


def _nomic_available() -> bool:
    import json
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen("http://localhost:11434/v1/models", timeout=2) as r:
            return any("nomic" in m.get("id", "") for m in json.load(r).get("data", []))
    except (urllib.error.URLError, OSError, ValueError):
        return False


@pytest.mark.integration
@pytest.mark.skipif(not _nomic_available(), reason="Ollama nomic-embed-text unreachable")
def test_semantic_dedup_live_collapses_token_disjoint_paraphrase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real embeddings collapse 'Keep it short' and 'Be concise' (no shared tokens)."""
    monkeypatch.undo()  # re-enable real semantic matching
    provider = ScriptedProvider()
    store = ProfileStore(tmp_path / ".workspec")
    agent = DraftAgent(provider=provider, profile_store=store)
    for rule in ("Keep it short", "Be concise"):
        provider.learned = _lt(("length", rule))
        agent.learn_from_edit(draft="A long winded reply.", sent="Short.")
    assert len([t for t in store.load().traits if t.category == "length"]) == 1
