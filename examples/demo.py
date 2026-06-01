#!/usr/bin/env python3
"""End-to-end demo of WorkSpec.

Lints two example status updates — one workslop, one solid — against the
``status_update.yaml`` contract that ships beside this demo, and renders both
verdicts. The contract lives here (not in the built-in ``rubrics/``) because it
is part of the example, loaded by path the same way you'd load your own.

Default (Anthropic):
    export ANTHROPIC_API_KEY=sk-...
    python examples/demo.py

OpenAI / OpenAI-compatible:
    export OPENAI_API_KEY=sk-...
    python examples/demo.py --provider openai
    python examples/demo.py --provider openai \
        --base-url http://localhost:11434/v1 --model llama3.1
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.rule import Rule

from workspec import WorkSpecAgent, load_spec
from workspec.render import render_verdict

console = Console()
EXAMPLES = Path(__file__).parent


def run(args: argparse.Namespace) -> None:
    spec = load_spec(str(EXAMPLES / "status_update.yaml"))
    agent = WorkSpecAgent(provider=args.provider, model=args.model, base_url=args.base_url)
    for name in ("bad_status_update.md", "good_status_update.md"):
        path = EXAMPLES / name
        console.print(Rule(f"Linting {name}  ·  {args.provider}"))
        verdict = agent.check(spec, path.read_text(encoding="utf-8"))
        render_verdict(verdict, spec, name, console=console)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WorkSpec demo")
    parser.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    parser.add_argument("--model", help="Model id (provider default if omitted).")
    parser.add_argument("--base-url", help="OpenAI-compatible endpoint URL.")
    run(parser.parse_args())
