"""End-to-end voice-learning scenarios across channels and people.

Drives the full ``DraftAgent`` loop (draft + learn_from_edit: provenance, dedup,
recurrence graduation, contradiction, negative signal, metrics, persistence) with
a scripted provider standing in for the LLM, so the *learning mechanics* are
exercised deterministically. Each scenario is a real-shaped conversation in one
of three channel formats — **email**, **Slack**, **WhatsApp** — to a specific
person, so the suite also covers:

* **Focus** — a draft is built around the message it is replying to (that
  submission, that person), not some other thread.
* **Context separation** — the global voice profile carries *generalizable* voice
  (tone, sign-off, length) across people and channels, while per-message content
  from one thread never leaks into a draft for a different person.

Semantic dedup is disabled here (it has its own suite) so results are hermetic;
a final integration-marked scenario exercises real semantic dedup.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from workspec.draft import DraftAgent, ExtractedTrait, GenerationDraft, LearnedTraits
from workspec.learning import decay
from workspec.learning.recurrence import GRADUATION_OBSERVATIONS, PROVISIONAL_WEIGHT_CAP
from workspec.models import Spec
from workspec.profile import ProfileStore, VoiceProfile, VoiceTrait
from workspec.providers import VerdictProvider

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
    """Returns scripted traits/draft and records every call's prompt — no network."""

    name = "scripted"

    def __init__(self) -> None:
        self.learned = LearnedTraits(traits=[])
        self.draft_result = GenerationDraft(draft="Thanks — confirming now.")
        self.calls: list[dict[str, Any]] = []

    def get_structured(self, system_prompt: str, user_prompt: str, schema):  # type: ignore[override]
        self.calls.append({"system": system_prompt, "user": user_prompt, "schema": schema})
        if schema is LearnedTraits:
            return self.learned
        if schema is GenerationDraft:
            return self.draft_result
        raise AssertionError(f"unexpected schema {schema!r}")


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


def test_voice_generalizes_across_people_and_channels(tmp_path: Path) -> None:
    """A trait reinforced across email/Slack/WhatsApp to three people graduates.

    Recurrence across *different* contexts is exactly the signal that a trait is
    generalizable voice (not a one-off for one recipient), so it graduates.
    """
    agent, provider, store = _agent(tmp_path)
    pairs = [
        (
            email(DANA, "Re: plan", "Many thanks for your guidance on this."),
            email(DANA, "Re: plan", "Thanks! Cheers, Alex"),
        ),
        (slack(RAJ, "appreciate the help on the bug"), slack(RAJ, "thanks! cheers")),
        (whatsapp(SAM, "thank you so much for sorting that"), whatsapp(SAM, "thanks!! cheers 🙌")),
    ]
    _graduate(agent, provider, "Sign off with a brief 'Cheers'", "signoff", pairs)
    t = _trait(store.load(), "Cheers")
    assert t is not None and t.status == "active"  # generalized across 3 people/channels


def test_learned_voice_applies_without_leaking_thread_content(tmp_path: Path) -> None:
    """Voice learned in one thread carries to another; that thread's content does not.

    Learn a sign-off from a budget email with Dana, then draft a dinner WhatsApp to
    Sam: the draft prompt must carry the generalizable voice trait but NOT Dana's
    budget content — the profile separates durable voice from per-message context.
    """
    agent, provider, store = _agent(tmp_path)
    pairs = [
        (
            email(DANA, "Q3 budget approval", f"Please approve the Q3 budget of $1.2M. {i}"),
            email(DANA, "Q3 budget approval", "Approved? Cheers, Alex"),
        )
        for i in range(GRADUATION_OBSERVATIONS)
    ]
    _graduate(agent, provider, "Sign off with 'Cheers'", "signoff", pairs)
    cheers = _trait(store.load(), "Cheers")
    assert cheers is not None and cheers.status == "active"

    agent.draft(_SPEC, whatsapp(SAM, "pizza or sushi tonight?"))
    prompt = provider.calls[-1]["user"]
    assert "Cheers" in prompt  # generalizable voice carries across
    assert "pizza or sushi" in prompt  # focused on Sam's actual message
    assert "budget" not in prompt.lower() and "dana" not in prompt.lower()  # no thread bleed
    assert "$1.2M" not in prompt


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
