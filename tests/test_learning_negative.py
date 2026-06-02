"""Unit tests for the negative-signal loop (``workspec.learning.negative``).

These are pure/deterministic: the penalty is a function of the token-level
difference between draft and sent, plus an injectable ``contradicts`` predicate.
No network is involved.
"""

from __future__ import annotations

from workspec.learning.negative import (
    NEGATIVE_DECREMENT,
    RETIRE_FLOOR,
    _default_contradicts,
    apply_negative_signal,
)
from workspec.profile import VoiceProfile, VoiceTrait


def _trait(
    category: str = "salutation",
    rule: str = "Open warmly with a friendly greeting.",
    weight: float = 0.8,
    observations: int = 3,
    status: str = "active",
) -> VoiceTrait:
    return VoiceTrait(
        category=category,  # type: ignore[arg-type]
        rule=rule,
        weight=weight,
        observations=observations,
        status=status,  # type: ignore[arg-type]
    )


# --- constants ------------------------------------------------------------ #


def test_constants_match_contract() -> None:
    assert NEGATIVE_DECREMENT == 0.15
    assert RETIRE_FLOOR == 0.2


# --- no-op cases ---------------------------------------------------------- #


def test_empty_applied_keys_is_noop() -> None:
    trait = _trait()
    profile = VoiceProfile(traits=[trait])
    assert apply_negative_signal(profile, [], "anything", "anything else") == []
    assert trait.weight == 0.8  # untouched


def test_key_not_in_applied_keys_is_skipped() -> None:
    trait = _trait()
    profile = VoiceProfile(traits=[trait])
    # A different key is supplied; the trait must not be penalized.
    out = apply_negative_signal(profile, ["tone:something else"], "hello friend", "hi")
    assert out == []
    assert trait.weight == 0.8


def test_unedited_text_no_penalty() -> None:
    trait = _trait()
    profile = VoiceProfile(traits=[trait])
    text = "Hi friend, hope you are well. Cheers."
    out = apply_negative_signal(profile, [trait.key], text, text)
    assert out == []
    assert trait.weight == 0.8


def test_already_retired_trait_is_skipped() -> None:
    trait = _trait(status="retired", weight=0.15)
    profile = VoiceProfile(traits=[trait])
    out = apply_negative_signal(
        profile, [trait.key], "Open warmly with a friendly greeting", "Report attached."
    )
    assert out == []
    assert trait.weight == 0.15


# --- signature-stripping reversal ----------------------------------------- #


def test_signature_stripped_penalizes() -> None:
    # The draft honored the rule (its signature words appear); the sent text
    # strips them out entirely -> reversal.
    trait = _trait(rule="Open warmly with a friendly greeting.", weight=0.8, observations=3)
    profile = VoiceProfile(traits=[trait])
    draft = "Warmly hoping this finds you well, my friendly greeting to you."
    sent = "Report attached."

    out = apply_negative_signal(profile, [trait.key], draft, sent)

    assert out == [trait]
    assert trait.weight == 0.8 - NEGATIVE_DECREMENT
    assert trait.observations == 2
    assert trait.status == "active"  # still above the retire floor


def test_signature_present_in_sent_no_penalty() -> None:
    # The distinctive words survive into the sent text -> guidance was kept.
    trait = _trait(rule="Open warmly with a friendly greeting.")
    profile = VoiceProfile(traits=[trait])
    draft = "Warmly: a friendly greeting and then the body."
    sent = "Warmly, a friendly greeting. Report attached."

    out = apply_negative_signal(profile, [trait.key], draft, sent)
    assert out == []
    assert trait.weight == 0.8


def test_signature_absent_from_draft_no_penalty() -> None:
    # The trait never actually shaped this particular draft, so removing
    # unrelated text is not evidence against it.
    trait = _trait(rule="Open warmly with a friendly greeting.")
    profile = VoiceProfile(traits=[trait])
    draft = "Please find the quarterly numbers below for review."
    sent = "Numbers below."

    out = apply_negative_signal(profile, [trait.key], draft, sent)
    assert out == []
    assert trait.weight == 0.8


# --- retirement below floor ----------------------------------------------- #


def test_penalty_below_floor_retires() -> None:
    trait = _trait(rule="Open warmly with a friendly greeting.", weight=0.3, observations=1)
    profile = VoiceProfile(traits=[trait])
    draft = "Warmly, a friendly greeting to start."
    sent = "Done."

    out = apply_negative_signal(profile, [trait.key], draft, sent)

    assert out == [trait]
    assert trait.weight == 0.3 - NEGATIVE_DECREMENT  # 0.15, below floor
    assert trait.weight < RETIRE_FLOOR
    assert trait.status == "retired"
    assert trait.observations == 0  # floored, not negative


def test_weight_floored_at_zero() -> None:
    trait = _trait(rule="Open warmly with a friendly greeting.", weight=0.1, observations=0)
    profile = VoiceProfile(traits=[trait])
    out = apply_negative_signal(profile, [trait.key], "warmly friendly greeting", "x")
    assert out == [trait]
    assert trait.weight == 0.0  # max(0.0, 0.1 - 0.15)
    assert trait.observations == 0  # already 0, stays floored
    assert trait.status == "retired"


# --- injectable contradicts predicate ------------------------------------- #


def test_injected_contradicts_drives_penalty() -> None:
    # A rule whose signature survives the edit would NOT trip the stripping
    # heuristic, but an injected predicate can still flag a reversal.
    trait = _trait(category="tone", rule="Be enthusiastic and upbeat.")
    profile = VoiceProfile(traits=[trait])
    draft = "Be enthusiastic and upbeat about the launch!"
    sent = "Be enthusiastic and upbeat about the launch."  # near-identical

    calls: list[tuple[str, str]] = []

    def always_contradicts(rule: str, change: str) -> bool:
        calls.append((rule, change))
        return True

    out = apply_negative_signal(
        profile, [trait.key], draft + " extra", sent, contradicts=always_contradicts
    )
    assert out == [trait]
    assert trait.weight == 0.8 - NEGATIVE_DECREMENT
    assert calls  # the injected predicate was consulted


def test_injected_contradicts_false_blocks_penalty() -> None:
    trait = _trait(rule="Open warmly with a friendly greeting.")
    profile = VoiceProfile(traits=[trait])
    draft = "Warmly, a friendly greeting to start."
    sent = "Done."  # signature stripped...

    # ...but the injected predicate says "no contradiction"; the stripping
    # heuristic still applies independently, so this must still penalize.
    out = apply_negative_signal(profile, [trait.key], draft, sent, contradicts=lambda _r, _c: False)
    assert out == [trait]  # stripping path is independent of contradicts


# --- default contradicts heuristic ---------------------------------------- #


def test_default_negation_cue_contradiction() -> None:
    # Rule says to include a sign-off; the removed text carries a negation cue
    # against the rule's signature word -> default heuristic flags it.
    trait = _trait(category="signoff", rule="Include a cheerful signoff line.")
    profile = VoiceProfile(traits=[trait])
    # draft has "signoff cheerful"; sent removes them and adds a negation token.
    draft = "Body text. cheerful signoff included."
    sent = "Body text. no signoff."

    out = apply_negative_signal(profile, [trait.key], draft, sent)
    assert out == [trait]


def test_default_antonym_contradiction() -> None:
    # Rule signature word "long"; the removed change introduces antonym "short".
    trait = _trait(category="length", rule="Write a long, detailed explanation.")
    profile = VoiceProfile(traits=[trait])
    draft = "Write a long detailed explanation here please."
    # "long" and "detailed" removed; "short" present in removed set.
    sent = "Write a short note."

    out = apply_negative_signal(profile, [trait.key], draft, sent)
    assert out == [trait]


# --- multiple traits ------------------------------------------------------ #


def test_multiple_traits_only_reversed_penalized() -> None:
    reversed_trait = _trait(
        category="salutation", rule="Open warmly with a friendly greeting.", weight=0.8
    )
    kept_trait = _trait(category="signoff", rule="Sign off with cheers.", weight=0.8)
    profile = VoiceProfile(traits=[reversed_trait, kept_trait])

    draft = "Warmly, a friendly greeting. Body. Cheers."
    sent = "Body. Cheers."  # salutation stripped, signoff kept

    out = apply_negative_signal(profile, [reversed_trait.key, kept_trait.key], draft, sent)

    assert out == [reversed_trait]
    assert reversed_trait.weight == 0.8 - NEGATIVE_DECREMENT
    assert kept_trait.weight == 0.8  # untouched


def test_returns_traits_in_profile_order() -> None:
    t1 = _trait(category="tone", rule="Be warm and personal.", weight=0.8)
    t2 = _trait(category="phrasing", rule="Use vivid descriptive imagery.", weight=0.8)
    profile = VoiceProfile(traits=[t1, t2])

    draft = "Be warm and personal. Use vivid descriptive imagery throughout."
    sent = "Plain status update."  # both stripped

    out = apply_negative_signal(profile, [t1.key, t2.key], draft, sent)
    assert out == [t1, t2]  # profile order preserved


# --- determinism ---------------------------------------------------------- #


def test_deterministic_repeatable_on_fresh_inputs() -> None:
    draft = "Warmly, a friendly greeting to open."
    sent = "Hi."

    def run() -> tuple[float, int, str]:
        trait = _trait(rule="Open warmly with a friendly greeting.")
        profile = VoiceProfile(traits=[trait])
        apply_negative_signal(profile, [trait.key], draft, sent)
        return trait.weight, trait.observations, trait.status

    assert run() == run()


# --- empty-signature guards ----------------------------------------------- #


def test_signatureless_rule_is_never_penalized() -> None:
    """A rule made only of stopwords/negations has no distinctive signature, so
    ``_was_reversed`` short-circuits and the trait is left untouched."""
    trait = _trait(rule="do not use the", weight=0.8)
    profile = VoiceProfile(traits=[trait])
    affected = apply_negative_signal(
        profile, [trait.key], draft="anything here", sent="nothing here"
    )
    assert affected == []
    assert trait.weight == 0.8
    assert trait.status == "active"


def test_default_contradicts_signatureless_rule_returns_false() -> None:
    """The default contradicts predicate returns False for a rule whose tokens
    are all stopwords/negations (no distinctive signature to contradict)."""
    assert _default_contradicts("do not use the", "no warm greeting here") is False
    # A real signature with an antonym cue in the change does fire, for contrast.
    assert _default_contradicts("write a long reply", "kept it short") is True


def test_self_antonym_rule_does_not_self_trigger() -> None:
    """A rule carrying BOTH ends of an antonym pair must not penalize itself."""
    from workspec.learning import negative

    # rule_sig has include AND exclude; a change that merely keeps 'include' must
    # not be read as a reversal (the pair is skipped).
    assert (
        negative._default_contradicts("include or exclude details", "include the details") is False
    )
