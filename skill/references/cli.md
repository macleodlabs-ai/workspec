# Reference: WorkSpec CLI

All commands accept `--provider {anthropic,openai}`, `--model ID`, and
`--base-url URL` (the last for any OpenAI-compatible endpoint: Azure, OpenRouter,
vLLM, Ollama, etc.). Anthropic is the default.

## draft — generate a reply in the user's voice

```bash
workspec draft SUBMISSION_FILE (--rubric NAME | --spec PATH) [options]
  --instruction TEXT   one-off steer, e.g. "keep it short"
  --profile-dir DIR    voice profile location (default .workspec)
  --json               structured output: draft, rationale, open_questions, used_profile
```

Prefer `--json` when driving programmatically so you can surface `open_questions`
to the user.

## check — lint work against a contract

```bash
workspec check WORK_FILE (--rubric NAME | --spec PATH) [options]
  --json               raw verdict as JSON
```

Exit: `0` pass · `1` blockers · `2` error.

## learn-from-edit — voice learning (learning mode only)

```bash
workspec learn-from-edit --draft DRAFT_FILE --sent SENT_FILE [options]
  --feedback TEXT      explicit note, e.g. "too formal"
  --dry-run            extract traits without writing the profile
  --profile-dir DIR
```

## profile — inspect or reset the learned voice

```bash
workspec profile                 # view traits
workspec profile --reset         # delete the profile
workspec profile --profile-dir DIR
```

## rubrics — list built-in contracts

```bash
workspec rubrics
```
