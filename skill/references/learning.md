# Reference: learning the user's voice

## The signal trust model

WorkSpec learns voice traits from two signals, weighted by reliability. Always
prefer the higher-trust one.

| Signal | How | Trust | Command |
|--------|-----|-------|---------|
| edit | user edited a draft WorkSpec produced | highest (1.0) | `learn-from-edit` |
| feedback | explicit instruction ("too formal") | high (0.9) | `learn-from-edit --feedback` |

The **edit** signal is the gold standard: the diff between the draft and what
the user actually sent is exactly what they would have changed. Lead with it.

## What gets learned (and what doesn't)

The learner extracts only traits that **generalize to future messages**:

- tone (warmer/cooler, more/less formal)
- structure (greeting style, sign-off, paragraphing, bullets)
- recurring phrasing the user adds or removes
- length / register preferences
- consistent deletions → `do_not` traits ("never open with 'I hope this finds you well'")

It deliberately ignores one-off content edits (a specific date, name, or number)
because those don't transfer to the next message.

## How traits accumulate

Each trait carries a `category`, a `rule` (actionable instruction), `provenance`,
a `weight` (0–1), an `evidence` note, and a `hits` count. Reinforcing a similar
trait raises its weight and bumps `hits`. When drafting, stronger traits lead
and `do_not` rules are surfaced as hard constraints.

## Privacy

The profile lives locally at `.workspec/voice_profile.json` (override with
`--profile-dir`). It is human-readable, the user can inspect it with
`workspec profile`, and delete it with `workspec profile --reset`. Never copy it
off the user's machine or include it in anything sent to a third party other
than the drafting model call itself.
