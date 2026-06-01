# Reference: checking work

## Built-in rubrics

Run `workspec rubrics` for the live list. Shipped rubrics (bundled in the package):

- `email_reply` — reply contract (default for `draft`).
- `decision_memo` — a memo asking a named person to choose between options.
- `ai_delegation_brief` — a work contract for handing a task to an AI agent.

A `status_update` contract ships under `examples/status_update.yaml` and is loaded
by path (`--spec examples/status_update.yaml`), not as a built-in name.

## Custom contracts

A contract is a YAML file with `must_include`, `must_not_include`,
`acceptance_tests`, and an `ai_policy` block. Pass one with `--spec PATH`
instead of `--rubric NAME`. Authoring tip for the user: derive it from 3 good +
3 weak examples rather than writing from scratch.

## Reading a verdict (JSON)

```json
{
  "passed": false,
  "summary": "one blunt sentence",
  "findings": [
    {"severity": "blocker", "rule": "...", "problem": "...",
     "evidence": "...", "suggested_fix": "..."}
  ],
  "rewrite_prompt": "paste-ready fix prompt, or null if it passed"
}
```

- `severity` is one of `blocker`, `warning`, `note`. The work fails if any
  `blocker` is present.
- Relay `summary` + each blocker's `problem`/`suggested_fix` to the user. Offer
  the `rewrite_prompt` if they want to fix it with AI.

## Exit codes

`0` passed · `1` blockers present · `2` usage/runtime error. Gate on these in
scripts: `workspec check f.md --rubric prd && send_it`.
