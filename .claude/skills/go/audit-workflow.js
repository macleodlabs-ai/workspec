export const meta = {
  name: 'go-audit-fix',
  description: 'Team audit (correctness + leanness) of the code under review, then fix and verify',
  phases: [
    { title: 'Scope', detail: 'find the files under review (git)' },
    { title: 'Audit', detail: 'parallel correctness + leanness reviewers' },
    { title: 'Verify', detail: 'adversarially confirm correctness findings' },
    { title: 'Fix', detail: 'apply fixes, grouped by file (no two agents share a file)' },
    { title: 'Gate', detail: 'run lint + type-check + tests, fix breakage' },
  ],
}

const focus = (args && args.focus) ? String(args.focus) : ''
const focusLine = focus ? `\nUser focus/scope hint: ${focus}` : ''

// --------------------------------------------------------------------------- //
// Phase 1 — Scope: discover the files under review
// --------------------------------------------------------------------------- //

phase('Scope')

const SCOPE_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    base: { type: 'string', description: 'Base branch compared against, or "" if none' },
    files: { type: 'array', items: { type: 'string' }, description: 'Relative paths under review' },
    summary: { type: 'string' },
  },
  required: ['base', 'files', 'summary'],
}

const scope = await agent(
  `You are scoping a code audit in the current git repo (cwd is the repo root).${focusLine}

Determine the set of source files "under review":
1. Find the default branch (try: \`git symbolic-ref refs/remotes/origin/HEAD\`, else 'main'/'master').
2. List files changed on the current branch vs the merge-base with that default branch:
   \`git diff --name-only $(git merge-base HEAD <base>)...HEAD\`
3. Add uncommitted working-tree changes: \`git status --porcelain\` (staged + unstaged).
4. Union them; keep only existing source files (code + tests). Drop deleted files, lockfiles,
   images, and generated artifacts.
5. If that union is EMPTY (clean tree, no branch divergence), fall back to the project's primary
   source package (the main importable package dir and its tests) so the audit still has a target.
6. If a focus hint was given, narrow to files matching it.

Return the base branch, the deduped relative file list, and a one-line summary. Do not edit anything.`,
  { label: 'scope', phase: 'Scope', schema: SCOPE_SCHEMA },
)

const files = (scope && scope.files) ? scope.files.filter(Boolean) : []
if (!files.length) {
  log('Nothing to audit — no changed files and no fallback package found.')
  return { scope, audited: 0, findings: [], message: 'nothing to audit' }
}
log(`Auditing ${files.length} file(s) vs ${scope.base || '(no base)'}`)
const fileList = files.join('\n')

// --------------------------------------------------------------------------- //
// Phase 2 — Audit: parallel correctness + leanness reviewers
// --------------------------------------------------------------------------- //

phase('Audit')

const FINDINGS_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    dimension: { type: 'string' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          kind: { type: 'string', enum: ['correctness', 'leanness'] },
          severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
          confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
          file: { type: 'string', description: 'Primary file the fix touches' },
          location: { type: 'string', description: 'function / line area' },
          problem: { type: 'string' },
          fix: { type: 'string', description: 'Concrete, minimal fix' },
          behavior_changing: { type: 'boolean' },
        },
        required: ['kind', 'severity', 'confidence', 'file', 'location', 'problem', 'fix', 'behavior_changing'],
      },
    },
  },
  required: ['dimension', 'findings'],
}

const base = `You are auditing these files in the current repo (read them; cite file:line). Be specific and
only report what the code supports.${focusLine}

FILES UNDER REVIEW:
${fileList}
`

const DIMENSIONS = [
  { key: 'completeness', kind: 'correctness', brief: `COMPLETENESS — unfinished work masquerading as done. Flag: stub/no-op bodies (a lone \`pass\`, \`...\`, or \`return None\`/\`return []\` where real logic is implied by the name/docstring); NotImplementedError; TODO/FIXME/XXX/HACK markers; placeholder or hardcoded fake return values; functions whose body does NOT deliver what their name + docstring promise; "phase N"/"will be filled in"/"stub" scaffolding; features that are wired/called but not actually implemented; tests that assert nothing, are skipped/xfail without cause, or only check trivia. This is the primary lens: prove things are really finished, not stubbed. kind="correctness".` },
  { key: 'logic', kind: 'correctness', brief: `CORRECTNESS — logic bugs: wrong conditions, off-by-one, inverted/again checks, incorrect state updates, wrong return values, mishandled None/empty, ordering bugs, races. Each finding kind="correctness".` },
  { key: 'edges', kind: 'correctness', brief: `CORRECTNESS — edge cases & error handling: unhandled exceptions, broad excepts that swallow, missing input validation, resource leaks (unclosed files/clients), retry/timeout gaps, boundary inputs. kind="correctness".` },
  { key: 'types', kind: 'correctness', brief: `CORRECTNESS — types & contracts: signature/return mismatches, Any leaks, Optional misuse, mutable defaults, model/schema violations, callers not matching a function's contract. kind="correctness".` },
  { key: 'dup', kind: 'leanness', brief: `LEANNESS — dead code & duplication: unused functions/params/imports/exports, duplicated logic or constants, copy-paste that should be shared, redundant branches. kind="leanness".` },
  { key: 'complexity', kind: 'leanness', brief: `LEANNESS — over-abstraction & complexity: needless indirection/abstraction, over-built features for the purpose, functions that can be simpler/shorter, repeated recomputation, awkward control flow. kind="leanness".` },
  { key: 'cruft', kind: 'leanness', brief: `LEANNESS — stale comments & cruft: outdated/misleading comments, leftover scaffolding/TODO/"phase N"/stub notes, docstrings that no longer match the code, commented-out code. kind="leanness".` },
]

const reviews = await parallel(
  DIMENSIONS.map((d) => () =>
    agent(`${base}\nFOCUS: ${d.brief}\n\nReport each issue with a concrete minimal fix, the primary file it touches, a confidence, and whether the fix changes behavior. Prefer precision over volume.`,
      { label: `audit:${d.key}`, phase: 'Audit', schema: FINDINGS_SCHEMA }),
  ),
)

const all = []
for (const r of reviews.filter(Boolean)) {
  for (const f of (r.findings || [])) all.push(f)
}
log(`Raw findings: ${all.length}`)

if (!all.length) {
  return { scope, audited: files.length, findings: [], message: 'no issues found' }
}

// --------------------------------------------------------------------------- //
// Phase 3 — Verify: adversarially confirm correctness findings
// --------------------------------------------------------------------------- //

phase('Verify')

const VERDICT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    real: { type: 'boolean' },
    why: { type: 'string' },
  },
  required: ['real', 'why'],
}

const correctness = all.filter((f) => f.kind === 'correctness')
const leanness = all.filter((f) => f.kind === 'leanness')

const verdicts = await parallel(
  correctness.map((f) => () =>
    agent(`Adversarially verify this claimed CORRECTNESS bug by reading the actual code in the repo. Default to
real=false unless you can confirm it from the code.

File: ${f.file}
Where: ${f.location}
Claim: ${f.problem}
Proposed fix: ${f.fix}

Is this a genuine bug that the proposed fix correctly addresses?`,
      { label: `verify:${f.file}`, phase: 'Verify', schema: VERDICT_SCHEMA }),
  ),
)

const confirmed = correctness.filter((_f, i) => verdicts[i] && verdicts[i].real)
log(`Confirmed correctness: ${confirmed.length}/${correctness.length}; leanness: ${leanness.length}`)

// Auto-apply ONLY safe, non-behavior-changing fixes — this is a sanity check,
// not an autonomous redesigner. Behavior-changing items (even confirmed bugs)
// are REPORTED for human review, honoring the skill's stated contract.
const toFix = [
  ...confirmed.filter((f) => !f.behavior_changing),
  ...leanness.filter((f) => !f.behavior_changing),
]
const reportOnly = [
  ...correctness.filter((_f, i) => !(verdicts[i] && verdicts[i].real)).map((f) => ({ ...f, reason: 'unconfirmed' })),
  ...confirmed.filter((f) => f.behavior_changing).map((f) => ({ ...f, reason: 'confirmed bug, but behavior-changing — review before applying' })),
  ...leanness.filter((f) => f.behavior_changing).map((f) => ({ ...f, reason: 'behavior-changing — needs review' })),
]

if (!toFix.length) {
  return { scope, audited: files.length, confirmed, leanness, reportOnly, fixed: [], message: 'nothing safe to auto-fix' }
}

// --------------------------------------------------------------------------- //
// Phase 4 — Fix: group by file so no two agents edit the same file
// --------------------------------------------------------------------------- //

phase('Fix')

const byFile = {}
for (const f of toFix) {
  (byFile[f.file] = byFile[f.file] || []).push(f)
}
const groups = Object.keys(byFile).map((file) => ({ file, items: byFile[file] }))

const fixed = await parallel(
  groups.map((g) => () =>
    agent(`Apply these audited fixes to ONE file only: ${g.file}. Edit no other file (other agents own the rest).
Read the file first. Make MINIMAL, behavior-preserving edits for leanness items, and the stated fix for
confirmed correctness items. Keep the project's style (type hints, docstring tone). Do NOT run the test suite
or repo-wide formatters; you may run a scoped formatter/linter on just this file. Do NOT run git.

FIXES TO APPLY in ${g.file}:
${g.items.map((it, i) => `${i + 1}. [${it.kind}/${it.severity}] ${it.location} — ${it.problem}\n   FIX: ${it.fix}`).join('\n')}

Return a one-line confirmation of what you changed.`,
      { label: `fix:${g.file}`, phase: 'Fix' }),
  ),
)

// --------------------------------------------------------------------------- //
// Phase 5 — Gate: run the project's lint + type-check + tests, fix breakage
// --------------------------------------------------------------------------- //

phase('Gate')

const gate = await agent(
  `The audit fixes are applied across these files:
${groups.map((g) => g.file).join('\n')}

Make the whole project pass its own quality gate. Discover the commands (prefer the project's virtualenv
binaries, e.g. .venv/bin/...):
- Formatter/linter (e.g. \`ruff check .\` + \`ruff format --check .\`, or flake8/black).
- Type checker (e.g. \`ty check\`, or \`mypy\`) if configured.
- Tests (e.g. \`pytest\`), with coverage if the project measures it.

Run them. FIX any breakage the audit fixes introduced (you may edit any file now). If a live/integration test
needs a service that's unavailable, it should already skip — do not delete tests. Re-run until green.

Return a concise PLAIN-TEXT report: the exact commands run, final pass/fail of each, test count, coverage %
if available, and anything you had to change to get green. Do NOT run git.`,
  { label: 'gate', phase: 'Gate' },
)

return { scope, audited: files.length, confirmed, leanness, reportOnly, fixed, gate }
