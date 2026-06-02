# Voice Learning v2 — design & plan

Reworks the voice-learning loop from "one edit → one strong rule" into a sound,
self-correcting model. Built on the v1 profile (`VoiceTrait`/`VoiceProfile` in
`workspec/profile.py`).

> No migration needed: voice learning shipped off-by-default and no
> `voice_profile.json` files exist yet, so there is **no back-compat layer** —
> the new fields are simply part of the model.

## Goals (the six soundness gaps)

1. **Recurrence gating** — a trait earns strength by recurring across distinct
   edits, not from a single sample. New traits start `provisional` (weight
   capped low) and graduate to `active` only after N independent observations.
2. **Recency decay** — a trait's *effective* weight fades with time since it was
   last seen, so stale style drifts out unless re-reinforced. Non-destructive.
3. **Contradiction resolution** — conflicting same-category traits ("be warm" vs
   "be terse") are detected; the weaker/older one is retired.
4. **Semantic dedup** — paraphrases ("keep it short" / "be concise") collapse to
   one reinforced trait via embeddings, with a lexical fallback.
5. **Negative signal** — when a trait that influenced a draft is edited back out,
   it is penalized (weight/observations down, retire below a floor). Closes the loop.
6. **Evaluation** — track draft→sent similarity over time so we can tell whether
   the profile is actually helping.

## Data model (foundation — `workspec/profile.py`)

`VoiceTrait` gains:
- `status: Literal["provisional", "active", "retired"] = "provisional"`
  (traits are born provisional; `reinforce_or_add` sets it explicitly).
- `observations: int = 1` — count of distinct learn events that produced it.
- `last_seen: str` (ISO, default now) — drives decay.
- `key` property → stable `f"{category}:{rule}"` used to trace applied traits.

`VoiceProfile` gains:
- `metrics: list[LearnMetric] = []` where `LearnMetric = {timestamp, edit_ratio}`
  (appended each `learn_from_edit`, for the eval surface).

Constants live with their feature module (below).

## Seams (the foundation wires these calls; Phase-2 agents implement the bodies)

New package `workspec/learning/` — one module per feature, each independently owned:

```python
# recurrence.py
GRADUATION_OBSERVATIONS = 3
PROVISIONAL_WEIGHT_CAP = 0.5
def maybe_graduate(trait: VoiceTrait) -> None
    """provisional → active once observations >= GRADUATION_OBSERVATIONS;
    while provisional, clamp weight <= PROVISIONAL_WEIGHT_CAP. Mutates trait."""

# decay.py
DECAY_HALFLIFE_DAYS = 90.0
def effective_weight(trait: VoiceTrait, now: datetime | None = None) -> float
    """Non-destructive: stored weight * 0.5 ** (age_days / DECAY_HALFLIFE_DAYS),
    age from trait.last_seen. Used for ranking/labeling in render_for_prompt."""

# contradiction.py
def detect_and_resolve(profile, new_trait, *, contradicts=None) -> list[VoiceTrait]
    """Among active same-category traits, find ones that contradict new_trait
    (using the injectable `contradicts(a_rule, b_rule)->bool`, default heuristic:
    negation/antonym cues). Retire the lower (effective_weight, observations,
    recency) side by setting status='retired'. Returns retired traits."""

# semantic.py
SIMILARITY_THRESHOLD = 0.82
def semantic_match(profile, rule, category, *, threshold=SIMILARITY_THRESHOLD) -> VoiceTrait | None
    """Return an existing same-category, non-retired trait whose rule is
    semantically equivalent (embedding cosine >= threshold), else None.
    Uses Ollama `nomic-embed-text` at $OLLAMA_BASE_URL (default localhost:11434);
    returns None when embeddings are unreachable so the caller falls back to the
    lexical `_find_similar`. Never raises on a missing server."""

# negative.py
NEGATIVE_DECREMENT = 0.15
RETIRE_FLOOR = 0.2
def apply_negative_signal(profile, applied_keys, draft, sent, *, contradicts=None) -> list[VoiceTrait]
    """For each trait whose .key is in applied_keys but whose guidance was reversed
    in `sent` vs `draft` (heuristic / contradicts), decrement weight by
    NEGATIVE_DECREMENT and observations by 1; retire if weight < RETIRE_FLOOR.
    Returns affected traits."""
```

## Wiring (foundation does this; Phase-2 does NOT touch profile.py/draft.py)

- `VoiceProfile.reinforce_or_add`: try `semantic.semantic_match` first, else lexical
  `_find_similar`. New traits created with `status="provisional"`, `observations=1`.
  On reinforce: `observations += 1`, refresh `last_seen`, then
  `recurrence.maybe_graduate(trait)` and `contradiction.detect_and_resolve(profile, trait)`.
- `VoiceProfile.render_for_prompt`: rank/label by `decay.effective_weight`; include
  only `active` traits (exclude `provisional`/`retired`); keep the do_not split.
- `DraftAgent.draft`: record which trait keys informed the draft → new
  `Draft.applied_traits: list[str]`; return them.
- `DraftAgent.learn_from_edit`: accept `applied_traits: list[str] | None = None`;
  after reinforcement, call `negative.apply_negative_signal`; append a `LearnMetric`
  (difflib ratio of draft vs sent) to `profile.metrics`.

## Eval surface (integration)

- `workspec profile --stats`: counts by status, top active traits by effective
  weight, and the recent draft→sent edit-ratio trend from `profile.metrics`.

## Phasing

- **Phase 1 — Foundation** (1 agent): data model + `learning/` package with stub
  bodies + all wiring in profile.py/draft.py/cli.py + this doc's constants. Leaves
  a green-importing tree (stubs are no-ops).
- **Phase 2 — Features** (5 parallel agents, one module each + its test file):
  recurrence, decay, contradiction, semantic, negative.
- **Phase 3 — Integration** (1 agent): coherent ordering, `profile --stats`,
  full ruff + ty + pytest (+ live semantic via Ollama embeddings), coverage,
  fix breakage.

## Non-negotiables

- No back-compat layer — there are no existing profiles to migrate.
- No hard dependency on a running Ollama: semantic/eval degrade gracefully.
- 100% style discipline: full type hints, ruff + ty clean, tests for each feature.
