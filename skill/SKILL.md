---
name: workspec
description: >-
  Draft replies to incoming messages (email, Slack, tickets, forms) in the
  user's own voice, and check inbound or outbound work against a quality
  contract before it is sent. Use whenever the user asks to reply to, respond
  to, draft, or triage a message on a comms channel, or to review whether a
  piece of work is ready to send. In learning mode, improves at matching the
  user's voice over time from how they edit the drafts it produces.
---

# WorkSpec

WorkSpec is a local command-line engine. This skill teaches you to drive it. You
provide the channel (you already read the user's email / Slack / tickets);
WorkSpec provides two things:

1. **draft** — generate a reply to an incoming message, written in the user's
   voice, against a contract of what a good reply contains.
2. **check** — lint any work (inbound or your own outbound draft) against a
   quality rubric and get a structured pass/fail with specific fixes.

WorkSpec never sends anything and has no channel access. You hand it text and it
returns text. **Send policy is yours / the user's — always let the user review a
draft before it goes out unless they have explicitly told you otherwise.**

## Setup check

Before first use, confirm the engine is available:

```bash
workspec rubrics
```

If that fails, the package is not installed. Tell the user to install it
(`uv pip install -e .` in the WorkSpec directory) and set an API key
(`ANTHROPIC_API_KEY`, or `OPENAI_API_KEY` with `--provider openai`). Do not try
to reproduce WorkSpec's behavior yourself — the whole point is the engine.

## Core workflow: reply to a message

When the user wants a reply drafted to an incoming message:

1. Save the incoming message text to a file (e.g. `/tmp/submission.txt`).
2. Generate the draft:

   ```bash
   workspec draft /tmp/submission.txt --rubric email_reply --json
   ```

   Use `--json` so you get the draft plus `open_questions` and `rationale`
   structured. The `--rubric email_reply` is the default reply contract; swap in
   another rubric or `--spec PATH` if the user has a custom one.

3. Present the draft to the user. **Surface every `open_questions` item** — these
   are things WorkSpec could not verify (an unconfirmed date, a commitment only
   the user can make). Do not paper over them.
4. The user reviews, edits, and sends (you do not send). If the message warrants
   it, you may first lint your own draft — see "Check a draft before sending".

A one-off steer is supported: `--instruction "keep it to three sentences"`.

## Check work against a contract

To judge whether a piece of work (a memo, an update, your own draft) is ready:

```bash
workspec check /tmp/work.md --rubric decision_memo --json
```

Exit code is `0` if it passes, `1` if it has blockers — so you can gate on it.
The JSON has `passed`, `summary`, `findings` (each with `severity`, `problem`,
`suggested_fix`), and a `rewrite_prompt`. Relay the blockers and the rewrite
prompt to the user plainly. See `references/checking.md` for the rubric list.

## Learning mode (improving the user's voice over time)

**Learning is OFF by default.** Only do the following if the user has explicitly
asked WorkSpec to learn their voice / improve over time.

The reliable signal is *how the user edits a draft you produced*. When learning
mode is on and the user edits a WorkSpec draft before sending:

1. Save your original draft (e.g. `/tmp/draft.txt`) and the version the user
   actually sent (e.g. `/tmp/sent.txt`).
2. Feed the difference back:

   ```bash
   workspec learn-from-edit --draft /tmp/draft.txt --sent /tmp/sent.txt
   ```

   If the user also gave a spoken instruction ("too formal", "drop the
   greeting"), pass it with `--feedback "too formal"`.

This distils durable voice traits (tone, sign-off, length, things to never do)
into a local profile that future drafts use automatically. It ignores one-off
content changes (a specific date or name) and only keeps what generalizes.

To show the user what has been learned, or to wipe it:

```bash
workspec profile           # view learned traits
workspec profile --reset   # delete the profile entirely
```

The profile is local, human-readable, and the user's to inspect or delete at any
time. See `references/learning.md` for how signals are weighted and decayed.

## Non-learning mode

If the user has not enabled learning, simply use `draft` and `check` and never
call `learn-from-edit` or anything that writes the profile. Drafts still use
whatever profile already exists (read-only); nothing new is recorded.

## Important boundaries

- **You do not send.** Return drafts for the user; respect their send policy.
- **Do not invent commitments.** If a draft contains a `[CONFIRM: ...]`
  placeholder or an open question, that is WorkSpec flagging something only the
  user can decide. Carry it to the user; never fill it in yourself.
- **The profile is sensitive.** It is a model of how the user writes. Keep it
  local, never paste it elsewhere, and delete it on request.
- **Don't reimplement the engine.** If `workspec` is unavailable, say so rather
  than free-handing drafts or quality checks — consistency comes from the engine.

## Reference files

- `references/checking.md` — full rubric list, custom contracts, exit codes.
- `references/learning.md` — signal trust model, profile internals.
- `references/cli.md` — complete command and flag reference.
