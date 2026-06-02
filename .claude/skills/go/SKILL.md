---
name: go
description: Final completion/sanity check — audit the current code with a team of subagents to prove it is really finished (no stubs/TODOs/no-ops), plus correctness bugs and leanness issues, then fix them and verify. Use when the user types /go, or asks to sanity-check completion, audit-and-fix, review-and-fix, clean up, or harden the current branch/diff.
---

# /go — team completion check, audit & fix

Run this as a **sanity completion check**: prove the work is really done — not stubbed, half-wired, or TODO'd — and free of correctness bugs and needless cruft. Orchestrate a multi-agent workflow that audits the code under review for (1) **completeness** (no stubs/placeholders/no-ops masquerading as done), (2) **correctness** defects, and (3) **leanness/cleanliness** issues, applies the fixes, and verifies against the project's own gate (lint + type-check + tests). Then independently verify and report.

## How to run it

1. **Launch the bundled workflow** (it runs in the background; you're notified on completion):

   ```
   Workflow({
     scriptPath: "<REPO_ROOT>/.claude/skills/go/audit-workflow.js",
     args: { focus: "<anything the user typed after /go, else empty>" }
   })
   ```

   Resolve `<REPO_ROOT>` with the current working directory (the repo you're in). Pass any words the user added after `/go` as `args.focus` to narrow scope (e.g. `/go the providers layer`).

2. **When it completes, independently verify** — don't just trust the agents. Re-run the project's full gate yourself (discover it: `ruff check . && ruff format --check .`, `ty check` or `mypy`, `pytest` with coverage if configured — prefer the project's `.venv` binaries) and confirm green.

3. **Report**: the confirmed correctness bugs fixed, the leanness fixes applied, anything deliberately left (with rationale), and the verification results.

4. **Do NOT commit or push** unless the user explicitly asks. Leave the working tree with the fixes applied and summarize.

## What the workflow does (phases)

- **Scope** — a scout agent uses git to list the files under review (commits on the current branch vs the default branch, plus uncommitted working changes); falls back to the package source if the tree is clean.
- **Audit (parallel)** — correctness reviewers (logic bugs, edge cases & error handling, types & contracts) and leanness reviewers (dead code & duplication, over-abstraction & complexity, stale comments & cruft). Each returns structured findings with severity, confidence, file, and a proposed fix.
- **Verify findings (adversarial)** — every correctness finding is independently re-checked by a skeptic; false positives are dropped before anything is touched.
- **Fix (file-owned, parallel)** — confirmed correctness fixes + leanness fixes are grouped **by file** so no two agents edit the same file; each agent applies minimal, behavior-preserving changes to its files only.
- **Gate** — one integrator discovers and runs the project's lint/type/test, fixes any breakage, and reports status + coverage.

## Principles

- **Safe by default**: apply high-confidence correctness fixes and behavior-preserving leanness cleanups. Anything risky or behavior-changing is *reported, not force-applied*.
- **No silent scope changes**: if a fix would change behavior or public API, surface it instead.
- **Idempotent**: if there's nothing to fix, the workflow says so rather than inventing work.
- This skill only orchestrates and fixes; **committing/pushing is the user's call.**
