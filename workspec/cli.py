"""WorkSpec command-line interface.

Commands
--------
    workspec rubrics
        List built-in contracts.

    workspec check WORK_FILE (--rubric NAME | --spec PATH) [options]
        Lint a piece of work against a contract.

    workspec draft SUBMISSION_FILE (--rubric NAME | --spec PATH) [options]
        Draft a reply to a message, in the user's voice.

    workspec learn-from-edit --draft FILE --sent FILE [--feedback ...]
        Learn voice traits from how the user edited a draft before sending.

    workspec profile [--reset | --stats]
        View, summarize, or delete the learned voice profile.

Exit codes
----------
    0   passed (or command succeeded)
    1   work failed a check (blockers present) — usable as a CI gate
    2   usage / runtime error
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path

from rich.console import Console

from workspec.draft import DraftAgent
from workspec.engine import (
    DEFAULT_MODEL,
    DEFAULT_OPENAI_MODEL,
    WorkSpecAgent,
)
from workspec.env import load_dotenv
from workspec.profile import DEFAULT_PROFILE_DIR, ProfileLoadError, ProfileStore, VoiceProfile
from workspec.render import render_verdict
from workspec.spec_loader import list_builtin_rubrics, load_spec

DEFAULT_PROVIDER = "anthropic"
_PROVIDERS = ("anthropic", "openai")

console = Console()
err = Console(stderr=True)


# --- shared arg helpers -------------------------------------------------- #


def _add_provider_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--provider",
        choices=list(_PROVIDERS),
        default=None,
        help="LLM backend. 'openai' covers any OpenAI-compatible endpoint via "
        "--base-url. Default: $WORKSPEC_PROVIDER or 'anthropic'.",
    )
    p.add_argument(
        "--model",
        help="Model id. Default: $WORKSPEC_MODEL, else the provider's built-in default.",
    )
    p.add_argument("--base-url", help="OpenAI-compatible endpoint URL.")


def _resolve_provider(args: argparse.Namespace) -> str:
    """Pick the backend: --provider flag > $WORKSPEC_PROVIDER > built-in default."""
    provider = (args.provider or os.environ.get("WORKSPEC_PROVIDER") or DEFAULT_PROVIDER).lower()
    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown provider '{provider}' (from $WORKSPEC_PROVIDER). "
            f"Use one of: {', '.join(_PROVIDERS)}."
        )
    return provider


def _add_profile_dir(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--profile-dir",
        default=str(DEFAULT_PROFILE_DIR),
        help=f"Voice profile directory (default: {DEFAULT_PROFILE_DIR}).",
    )


def _profile_store(args: argparse.Namespace) -> ProfileStore:
    """Build the ProfileStore for the active profile dir (defaulted by the subparser)."""
    return ProfileStore(getattr(args, "profile_dir", DEFAULT_PROFILE_DIR))


def _resolve_model(args: argparse.Namespace, provider: str) -> str | None:
    """Pick the model: --model flag > $WORKSPEC_MODEL > ``provider``'s default."""
    model: str | None = args.model or os.environ.get("WORKSPEC_MODEL")
    if model:
        return model
    return DEFAULT_MODEL if provider == "anthropic" else DEFAULT_OPENAI_MODEL


# --- rubrics ------------------------------------------------------------- #


def _cmd_rubrics(_: argparse.Namespace) -> int:
    rubrics = list_builtin_rubrics()
    if not rubrics:
        console.print("[yellow]No built-in rubrics found.[/]")
        return 0
    console.print("[bold]Built-in contracts:[/]")
    for name in rubrics:
        spec = load_spec(name)
        console.print(f"  [cyan]{name}[/]  — {spec.title}")
    console.print(
        f"\nCheck:  [bold]workspec check WORK_FILE --rubric {next(iter(rubrics))}[/]"
        f"\nDraft:  [bold]workspec draft MSG_FILE --rubric email_reply[/]"
    )
    return 0


# --- check --------------------------------------------------------------- #


def _cmd_check(args: argparse.Namespace) -> int:
    work_path = Path(args.work_file)
    if not work_path.exists():
        err.print(f"[red]Work file not found:[/] {work_path}")
        return 2
    source = args.spec or args.rubric
    if not source:
        err.print("[red]Provide a contract via --rubric NAME or --spec PATH.[/]")
        return 2
    try:
        spec = load_spec(source)
    except Exception as exc:
        err.print(f"[red]Could not load contract:[/] {exc}")
        return 2

    try:
        provider = _resolve_provider(args)
        agent = WorkSpecAgent(
            provider=provider, model=_resolve_model(args, provider), base_url=args.base_url
        )
    except (RuntimeError, ValueError) as exc:
        err.print(f"[red]{exc}[/]")
        return 2

    try:
        text = work_path.read_text(encoding="utf-8")
        cm = (
            contextlib.nullcontext()
            if args.json
            else console.status(f"Linting against [cyan]{spec.title}[/]…")
        )
        with cm:
            verdict = agent.check(spec, text)
    except Exception as exc:
        err.print(f"[red]Check failed:[/] {type(exc).__name__}: {exc}")
        return 2

    if args.json:
        print(verdict.model_dump_json(indent=2))
    else:
        render_verdict(verdict, spec, work_path.name, console=console)
    return 0 if verdict.passed else 1


# --- draft --------------------------------------------------------------- #


def _draft_agent(args: argparse.Namespace) -> DraftAgent:
    provider = _resolve_provider(args)
    return DraftAgent(
        provider=provider,
        model=_resolve_model(args, provider),
        base_url=args.base_url,
        profile_store=_profile_store(args),
    )


def _cmd_draft(args: argparse.Namespace) -> int:
    sub_path = Path(args.submission)
    if not sub_path.exists():
        err.print(f"[red]Submission file not found:[/] {sub_path}")
        return 2
    # --rubric defaults to email_reply on the subparser, so source is always set
    # here; an explicit --spec still wins.
    source = args.spec or args.rubric
    try:
        spec = load_spec(source)
    except Exception as exc:
        err.print(f"[red]Could not load contract:[/] {exc}")
        return 2
    try:
        agent = _draft_agent(args)
    except (RuntimeError, ValueError) as exc:
        err.print(f"[red]{exc}[/]")
        return 2
    try:
        result = agent.draft(
            spec, sub_path.read_text(encoding="utf-8"), instruction=args.instruction or ""
        )
    except Exception as exc:
        err.print(f"[red]Drafting failed:[/] {type(exc).__name__}: {exc}")
        return 2

    # Hand-off for the negative-signal loop: persist the trait keys that informed
    # this draft so a later `learn-from-edit --applied-traits` can penalize the
    # ones the user edited back out. The sidecar lives next to the submission.
    sidecar = sub_path.with_suffix(sub_path.suffix + ".traits")
    if result.applied_traits:
        sidecar.write_text("\n".join(result.applied_traits) + "\n", encoding="utf-8")

    if args.json:
        print(result.model_dump_json(indent=2))
        return 0
    console.print(result.draft)
    if result.open_questions:
        console.print("\n[yellow]Before sending, check:[/]")
        for q in result.open_questions:
            console.print(f"  [yellow]•[/] {q}")
    if result.rationale:
        tag = "voice profile applied" if result.used_profile else "no profile yet"
        console.print(f"\n[dim]approach: {result.rationale} · {tag}[/]")
    if result.applied_traits:
        console.print(
            f"[dim]applied traits written to {sidecar} "
            f"(pass to `learn-from-edit --applied-traits`).[/]"
        )
    return 0


# --- learn-from-edit ----------------------------------------------------- #


def _resolve_applied_traits(values: list[str] | None) -> list[str]:
    """Resolve ``--applied-traits`` into trait keys.

    Each value is either a path to a sidecar file (one ``category:rule`` key per
    line, as written by ``workspec draft``) or a literal key. File contents are
    expanded; literals pass through. Blank lines are dropped.
    """
    if not values:
        return []
    keys: list[str] = []
    for value in values:
        path = Path(value)
        if path.is_file():
            keys.extend(
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        elif value.strip():
            keys.append(value.strip())
    return keys


def _cmd_learn_edit(args: argparse.Namespace) -> int:
    draft_path, sent_path = Path(args.draft), Path(args.sent)
    for p in (draft_path, sent_path):
        if not p.exists():
            err.print(f"[red]File not found:[/] {p}")
            return 2
    try:
        agent = _draft_agent(args)
    except (RuntimeError, ValueError) as exc:
        err.print(f"[red]{exc}[/]")
        return 2
    try:
        applied = agent.learn_from_edit(
            draft=draft_path.read_text(encoding="utf-8"),
            sent=sent_path.read_text(encoding="utf-8"),
            feedback=args.feedback or "",
            apply=not args.dry_run,
            applied_traits=_resolve_applied_traits(args.applied_traits),
        )
    except Exception as exc:
        err.print(f"[red]Learning failed:[/] {type(exc).__name__}: {exc}")
        return 2

    if not applied:
        console.print("[dim]No generalizable voice traits found in this edit.[/]")
        return 0
    verb = "would learn" if args.dry_run else "learned"
    console.print(f"[green]{verb} {len(applied)} voice trait(s):[/]")
    for t in applied:
        console.print(f"  [cyan]{t.category}[/]: {t.rule} [dim](w={t.weight:.2f})[/]")
    if not args.dry_run:
        console.print(f"[dim]profile updated: {_profile_store(args).path}[/]")
    return 0


# --- profile ------------------------------------------------------------- #


def _print_profile_stats(profile: VoiceProfile, store: ProfileStore) -> int:
    """Render the ``profile --stats`` eval surface: status counts, top active
    traits by effective weight, and the recent draft→sent edit-ratio trend."""
    stats = profile.stats()
    console.print(f"[bold]Voice profile stats[/] [dim]({store.path})[/]")
    console.print(
        f"[dim]{stats.total} trait(s): "
        f"[green]{stats.counts['active']} active[/], "
        f"{stats.counts['provisional']} provisional, "
        f"{stats.counts['retired']} retired[/]\n"
    )

    if stats.top_active:
        console.print("[bold]Top active traits[/] [dim](by effective weight)[/]")
        for ts in stats.top_active:
            console.print(
                f"  [cyan]{ts.category}[/] [dim]eff={ts.effective_weight:.2f} "
                f"(w={ts.weight:.2f}, ×{ts.observations} obs)[/]\n    {ts.rule}"  # noqa: RUF001
            )
    else:
        console.print("[dim]No active traits yet (still provisional / retired).[/]")

    console.print()
    if stats.recent_edit_ratio is None:
        console.print("[dim]No edit-ratio metrics recorded yet.[/]")
    else:
        msg = (
            f"[bold]Edit-ratio trend[/] [dim](over {stats.metric_count} learn event(s))[/]\n"
            f"  recent mean: {stats.recent_edit_ratio:.2f} "
            "[dim](1.0 == sent unedited)[/]"
        )
        if stats.edit_ratio_delta is not None:
            delta = stats.edit_ratio_delta
            if delta > 0:
                msg += f"\n  [green]↑ {delta:+.2f} vs earlier[/]"
            elif delta < 0:
                msg += f"\n  [yellow]↓ {delta:+.2f} vs earlier[/]"
            else:
                msg += "\n  [dim]no change vs earlier[/]"
        console.print(msg)
    return 0


def _cmd_profile(args: argparse.Namespace) -> int:
    store = ProfileStore(getattr(args, "profile_dir", DEFAULT_PROFILE_DIR))
    if args.reset:
        if store.exists():
            store.path.unlink()
            console.print(f"[yellow]Voice profile deleted:[/] {store.path}")
        else:
            console.print("[dim]No profile to delete.[/]")
        return 0
    try:
        profile = store.load()
    except ProfileLoadError as exc:
        err.print(f"[red]{exc}[/]")
        return 2
    if getattr(args, "stats", False):
        return _print_profile_stats(profile, store)
    if not profile.traits:
        console.print(
            "[yellow]No voice profile yet.[/] [dim]It builds up via `workspec learn-from-edit`.[/]"
        )
        return 0
    console.print(f"[bold]Voice profile[/] [dim]({store.path})[/]")
    console.print(f"[dim]{len(profile.traits)} trait(s), updated {profile.updated_at}[/]\n")
    for t in sorted(profile.traits, key=lambda x: x.weight, reverse=True):
        console.print(
            f"  [cyan]{t.category}[/] [dim]w={t.weight:.2f} "
            f"{t.provenance} ×{t.hits}[/]\n    {t.rule}"  # noqa: RUF001
        )
    return 0


# --- parser -------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="workspec",
        description="Lint work against a quality contract, and draft replies in your voice.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_rubrics = sub.add_parser("rubrics", help="List built-in contracts.")
    p_rubrics.set_defaults(func=_cmd_rubrics)

    p_check = sub.add_parser("check", help="Lint a work file against a contract.")
    p_check.add_argument("work_file", help="Path to the work to check.")
    p_check.add_argument("--rubric", help="Built-in contract name (see `workspec rubrics`).")
    p_check.add_argument(
        "--spec", help="Path to any contract YAML file (absolute, relative, or ~/...)."
    )
    _add_provider_args(p_check)
    p_check.add_argument(
        "--json", action="store_true", help="Emit the verdict as JSON (for CI / scripting)."
    )
    p_check.set_defaults(func=_cmd_check)

    p_draft = sub.add_parser("draft", help="Draft a reply to a message, in the user's voice.")
    p_draft.add_argument("submission", help="Path to the incoming message to reply to.")
    p_draft.add_argument(
        "--rubric",
        default="email_reply",
        help="Reply contract: built-in name (default email_reply).",
    )
    p_draft.add_argument(
        "--spec", help="Reply contract: path to any spec YAML (absolute, relative, or ~/...)."
    )
    p_draft.add_argument("--instruction", help="One-off steer, e.g. 'keep it short'.")
    _add_provider_args(p_draft)
    _add_profile_dir(p_draft)
    p_draft.add_argument(
        "--json",
        action="store_true",
        help="Emit the structured draft (with open questions) as JSON.",
    )
    p_draft.set_defaults(func=_cmd_draft)

    p_le = sub.add_parser(
        "learn-from-edit",
        help="Learn voice traits from how the user edited a draft before sending.",
    )
    p_le.add_argument("--draft", required=True, help="Path to the draft WorkSpec produced.")
    p_le.add_argument("--sent", required=True, help="Path to what the user actually sent.")
    p_le.add_argument("--feedback", help="Optional explicit note, e.g. 'too formal'.")
    p_le.add_argument(
        "--applied-traits",
        nargs="*",
        metavar="FILE_OR_KEY",
        help="Trait keys that informed the draft (drives the negative-signal loop). "
        "Each value is a sidecar file written by `workspec draft` (one key per "
        "line) or a literal category:rule key.",
    )
    p_le.add_argument(
        "--dry-run", action="store_true", help="Extract traits without writing them to the profile."
    )
    _add_provider_args(p_le)
    _add_profile_dir(p_le)
    p_le.set_defaults(func=_cmd_learn_edit)

    p_prof = sub.add_parser("profile", help="View or reset the learned voice profile.")
    p_prof.add_argument("--reset", action="store_true", help="Delete the voice profile.")
    p_prof.add_argument(
        "--stats",
        action="store_true",
        help="Show trait counts by status, top active traits, and the edit-ratio trend.",
    )
    _add_profile_dir(p_prof)
    p_prof.set_defaults(func=_cmd_profile)

    return parser


def main(argv: list[str] | None = None) -> int:
    # Load repo .env early so WORKSPEC_MODEL / WORKSPEC_PROVIDER (and API keys) are
    # visible to resolution below. Never overrides vars already set in the shell.
    load_dotenv()
    args = build_parser().parse_args(argv)
    exit_code: int = args.func(args)
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
