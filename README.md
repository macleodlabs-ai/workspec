# WorkSpec

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with ty](https://img.shields.io/badge/types-ty-261230.svg)](https://github.com/astral-sh/ty)
[![Tested with pytest](https://img.shields.io/badge/tested%20with-pytest-0a9edc.svg)](https://docs.pytest.org/)
[![Providers: Anthropic · OpenAI · Ollama](https://img.shields.io/badge/providers-Anthropic%20%C2%B7%20OpenAI%20%C2%B7%20Ollama-8a2be2.svg)](#providers)

![](assets/workspec.png)

Lint work against a quality contract, and draft replies in your voice. One small
local engine, two capabilities, backed by Anthropic or any OpenAI-compatible model.

> Built by **[MacLeod Labs](https://github.com/macleodlabs-ai)**.

- **check** — judge whether a piece of work meets an explicit contract, and get a structured pass/fail with specific fixes.
- **draft** — generate a reply to an incoming message in the user's voice, against a contract, and learn that voice over time from how the user edits the drafts it produces.

The model is constrained to typed schemas (structured outputs), so results are always validated objects — never prose to parse. WorkSpec has no channel access and never sends messages on your behalf; it does send the work/submission text (and, for drafting, your voice profile) to the model backend you configure — including any custom `--base-url`.

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [Checking work](#checking-work)
- [Drafting in the user's voice](#drafting-in-the-users-voice)
- [Learning the voice](#learning-the-voice)
- [The agent skill](#the-agent-skill)
- [Providers](#providers)
- [Built-in contracts](#built-in-contracts)
- [Architecture](#architecture)
- [Development](#development)
- [License](#license)

## Install

WorkSpec uses [uv](https://docs.astral.sh/uv/) — it's the default for every command here (it's many times faster than pip and manages the virtual env for you).

```bash
uv venv                              # create .venv
uv pip install -e .                  # both backends (Anthropic + OpenAI) ship by default

export ANTHROPIC_API_KEY=sk-ant-...  # default provider
export OPENAI_API_KEY=sk-...         # for --provider openai
```

> Don't have uv? Install it with `curl -LsSf https://astral.sh/uv/install.sh | sh` (or `pip install uv`). Plain `pip install -e .` still works if you prefer.

Or put keys in a `.env` file at the repo root (loaded automatically; does not override variables already exported in your shell).

```bash
uv run examples/demo.py   # lints good/bad status_update examples
```

## Quick start

```bash
workspec rubrics                                   # list built-in contracts
workspec check report.md --rubric decision_memo    # lint a piece of work
workspec draft incoming.txt --rubric email_reply   # draft a reply in your voice
```

## Checking work

```bash
workspec check WORK_FILE (--rubric NAME | --spec PATH) [--json]
```

Judges **structure and rigor** — missing owner, decision, evidence, source, risk, or next step; vague or unsupported claims; hollow filler. It deliberately does **not** judge whether the strategy is *correct*; over-claiming that is how these tools lose trust.

Exit code is `0` when the work passes and `1` when it has blockers, so it drops into CI or a git hook as a gate. The verdict (JSON via `--json`) carries `passed`, `summary`, `findings` (each with `severity`, `problem`, `suggested_fix`), and a `rewrite_prompt`.

## Drafting in the user's voice

The same contract that *judges* inbound work can *generate* an outbound reply.

```bash
workspec draft incoming.txt --rubric email_reply --json
```

The draft is written against a learned **voice profile** — a local, human-readable model of how the user writes. It never invents commitments: anything it cannot verify becomes a `[CONFIRM: ...]` placeholder and an `open_questions` item for the user to resolve before sending. A one-off steer is supported with `--instruction "keep it to three sentences"`.

## Learning the voice

Off by default. When enabled, WorkSpec learns from how the user edits the drafts it produced — the reliable signal:

```bash
workspec learn-from-edit --draft draft.txt --sent sent.txt --feedback "too formal"
workspec profile          # see what's been learned
workspec profile --reset  # wipe it
```

It distils only *generalizable* traits (tone, sign-off, length, things to never do) and ignores one-off content edits. Two signals, by trust: edits (highest) and explicit feedback. The profile is local at `.workspec/voice_profile.json` and the user's to inspect or delete.

## The agent skill

`skill/` is a portable [Agent Skill](https://agentskills.io) (`SKILL.md` + reference files) that teaches any host agent sitting on a comms channel — email, Slack, tickets — how to drive WorkSpec: draft a reply, surface open questions, let the user send, and (in learning mode) feed edits back. The skill contains no channel code or credentials; the host owns the channel and the send policy. Drop `skill/` into a Claude Code / Cowork skills directory.

## Providers

Both backends ship by default — no extra install. Select one with `--provider`. The `openai` backend covers any OpenAI-compatible endpoint via `--base-url`.


| Provider    | `--base-url` | Default model     | Notes                                             |
| ----------- | ------------ | ----------------- | ------------------------------------------------- |
| `anthropic` | not used     | `claude-opus-4-8` | uses `messages.parse` structured outputs          |
| `openai`    | optional     | `gpt-5.5`         | set `--base-url` for Azure/OpenRouter/vLLM/Ollama |


### Choosing the model

The model and provider resolve in this order — **flag → environment variable → built-in default**:

```bash
# Per-command (highest priority)
workspec check memo.md --rubric decision_memo --model claude-haiku-4-5
workspec draft incoming.txt --provider openai --model gpt-5.5

# Set a default once (env var) — applies to every command, no flag needed
export WORKSPEC_MODEL=claude-haiku-4-5
export WORKSPEC_PROVIDER=anthropic     # optional; defaults to anthropic
workspec check memo.md --rubric decision_memo   # uses claude-haiku-4-5
```

Both env vars are also read from a repo-root `.env` file (without overriding what's already in your shell). With nothing set, the default is `--provider anthropic` and `claude-opus-4-8` (sharpest judgment); `gpt-5.5` for `--provider openai`.

```bash
# Local Ollama
workspec check memo.md --rubric decision_memo \
  --provider openai --base-url http://localhost:11434/v1 --model llama3.1
```

## Built-in contracts

Built-in contracts are plain YAML files that ship inside the package (run `workspec rubrics` to list them) — first-class, editable data. List them with `workspec rubrics`.

| Name                  | For                                                     |
| --------------------- | ------------------------------------------------------- |
| `email_reply`         | reply contract (default for `draft`).                   |
| `decision_memo`       | a memo asking a named person to choose between options. |
| `ai_delegation_brief` | a work contract for handing a task to an AI agent.      |

A contract is plain YAML (`must_include`, `must_not_include`, `acceptance_tests`, `ai_policy`).

**Use a contract from any file.** Pass any YAML on disk with `--spec` — absolute, relative, or `~/...` paths all work, and the `.yaml`/`.yml` extension is optional:

```bash
workspec check report.md --spec ./contracts/board_memo.yaml
workspec check report.md --spec ~/team/standards/decision_memo   # extension optional
```

`examples/status_update.yaml` is one such file — the demo loads it by path rather than as a built-in.

## Architecture

Single-purpose modules:

```text
workspec/
  models.py        Spec (contract) + Verdict/Finding (typed check output)
  providers.py     Anthropic and OpenAI-compatible backends behind one interface
  engine.py        The lint engine: contract + work -> Verdict
  profile.py       Voice profile: learned, versioned, human-readable traits
  draft.py         Voice-aware reply generation + learn-from-edit
  spec_loader.py   Load built-in contracts or a YAML file from anywhere on disk
  render.py        Rich terminal rendering for verdicts
  cli.py           argparse entrypoint (rubrics, check, draft, learn-from-edit, profile)
  rubrics/*.yaml   Built-in contracts — ship inside the package (run `workspec rubrics`)
examples/          Runnable demo + sample work and its status_update.yaml contract
skill/             Portable Agent Skill wrapping the CLI for host agents
```

The provider layer is the only code that touches an SDK: Anthropic via `messages.parse(output_format=...)`, OpenAI via `chat.completions.parse(response_format=...)`. Both enforce the Pydantic schema natively, so there is no prose parsing anywhere.

## Development

Dev tooling is declared as a [PEP 735](https://peps.python.org/pep-0735/) dependency group and installed with uv:

```bash
uv pip install -e .                       # runtime (both backends included)
uv pip install ruff ty pytest pytest-cov  # lint, type-check, test tools
```

Type checking uses [`ty`](https://docs.astral.sh/ty/) — Astral's Rust type checker (same team as uv and ruff), many times faster than mypy. Lint and type-check (both must pass clean before a PR):

```bash
uv run ruff check .        # lint
uv run ruff format .       # auto-format
uv run ty check            # type-check (config in pyproject.toml)
```

Ruff and ty are configured under `[tool.ruff]` / `[tool.ty]` in `pyproject.toml`.

### Tests

Run from the **project root** (pytest only honors `testpaths` when invoked there — from a subdirectory it finds nothing):

```bash
uv run pytest          # from the repo root
```

The suite has two layers:

- **Unit tests** — fast, no network. Every module is covered with a fake provider (`uv run pytest -m "not integration"`).
- **Live integration tests** (`tests/test_integration_backends.py`) — run a real `check`, `draft`, and `learn-from-edit` against **each backend that's reachable**: a local [Ollama](https://ollama.com) server, Anthropic (when `ANTHROPIC_API_KEY` is set), and OpenAI (when `OPENAI_API_KEY` is set). They assert the structured-output contract and **auto-skip** any backend that's unavailable, so the suite is safe to run anywhere.

```bash
uv run pytest                       # everything available
uv run pytest -m "not integration"  # unit tests only (no network)
uv run pytest -m integration -v     # just the live-backend tests
uv run pytest --cov=workspec --cov-report=term-missing   # coverage

# To exercise the local path with no cloud keys:
ollama serve && ollama pull llama3.2
uv run pytest -m integration        # WORKSPEC_OLLAMA_MODEL=... to force a model
```

The unit layer holds the package at **100% line coverage on the hermetic suite** (`pytest -m "not integration"`); the integration layer additionally exercises the live structured-output path against real models.

## License

[MIT](LICENSE) © [MacLeod Labs](https://github.com/macleodlabs-ai). Use it, fork it, ship it.