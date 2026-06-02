"""Cross-recipient promotion — a recurring trait is *earned* into the shared layer.

Generalization is not pooling (Decision 6): a trait learned for one recipient
must never leak to a different recipient. But when the *same* trait has
independently graduated to ``active`` across enough distinct recipient scopes,
that is evidence it is a property of how this person writes in general, not of
one relationship — so it is promoted to the shared global layer.

Promotion is deliberately conservative:

  * Only ``active`` recipient-scope traits count. A provisional, not-yet-trusted
    trait in a single relationship is not evidence of anything general.
  * Distinctness is by *recipient scope*, not by observation: three independent
    relationships graduating "sign off with 'Cheers'" promotes; three edits to
    one recipient does not.
  * A trait already held by the global layer is not re-counted into a duplicate;
    promotion reinforces the existing shared trait instead (it simply gets
    stronger), reusing the profile's own paraphrase-aware dedup.

The trait-identity test is delegated to :meth:`VoiceProfile.find_match`, so a
promotion groups recipient traits exactly the way reinforcement collapses
paraphrases onto one trait — semantic dedup first, lexical Jaccard fallback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from workspec.context import ContextKey
from workspec.profile import VoiceProfile

if TYPE_CHECKING:
    from workspec.profile import VoiceTrait
    from workspec.store import ContextStore

#: Distinct recipient scopes that must each have independently graduated a
#: matching trait before it is promoted to the shared global layer.
PROMOTION_DISTINCT_RECIPIENTS = 3


def count_graduated_recipients(
    profiles: dict[str, VoiceProfile],
    trait: VoiceTrait,
) -> int:
    """Count distinct recipient profiles that hold an *active* match for ``trait``.

    ``profiles`` maps a recipient id to that recipient's voice profile. A profile
    counts when it contains a non-retired, ``active`` trait equivalent to
    ``trait`` (same category, paraphrase-aware match). Pure and store-free so it
    can be unit-tested without touching disk.
    """
    count = 0
    for profile in profiles.values():
        match = profile.find_match(trait.rule, trait.category)
        if match is not None and match.status == "active":
            count += 1
    return count


def maybe_promote(
    store: ContextStore,
    trait: VoiceTrait,
    *,
    distinct_recipients: int = PROMOTION_DISTINCT_RECIPIENTS,
) -> VoiceTrait | None:
    """Promote ``trait`` to the global layer once enough recipients have earned it.

    Scans every recipient-scope voice profile under ``store`` and, if at least
    ``distinct_recipients`` of them independently hold an ``active`` match for
    ``trait``, reinforces it into the global profile and returns the resulting
    shared trait. Returns ``None`` when the bar is not met (no global write).

    The promoted trait lands in the global layer already ``active`` — it has met a
    *stronger* bar than ordinary in-scope graduation (independent graduation in
    several relationships), so it should not have to re-graduate globally. It
    carries an evidence note recording that it was promoted, so the owner can
    audit *why* a shared trait exists. Grafting reuses the global profile's
    paraphrase-aware dedup, so promoting an already-shared trait strengthens it
    rather than duplicating it.
    """
    if not trait.rule.strip():
        return None

    recipient_profiles = _recipient_profiles(store)
    if count_graduated_recipients(recipient_profiles, trait) < distinct_recipients:
        return None

    incoming = trait.model_copy(deep=True)
    incoming.status = "active"
    incoming.evidence = f"promoted from {distinct_recipients}+ recipients"
    global_profile = store.load_voice(ContextKey())
    promoted = global_profile.graft_trait(incoming)
    store.save_voice(ContextKey(), global_profile)
    return promoted


def _recipient_profiles(store: ContextStore) -> dict[str, VoiceProfile]:
    """Load every recipient-scope voice profile under ``store``, keyed by recipient.

    Promotion counts only single-axis recipient scopes, so this reads the
    ``recipient=<id>.json`` files; the global scope and composite
    (channel/project) scopes are excluded. Returns an empty map when nothing
    has been learned per-recipient yet.
    """
    profiles: dict[str, VoiceProfile] = {}
    voice_dir = store.voice_dir
    if not voice_dir.exists():
        return profiles
    prefix = "recipient="
    for path in sorted(voice_dir.glob("*.json")):
        scope_id = path.stem
        # Only single-axis recipient scopes are counted; the global scope and
        # any composite (channel/project) scopes are excluded.
        if not scope_id.startswith(prefix) or "__" in scope_id:
            continue
        recipient = scope_id[len(prefix) :]
        profiles[recipient] = store.load_voice(ContextKey(recipient=recipient))
    return profiles
