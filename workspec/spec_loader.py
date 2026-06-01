"""Loading specs — from built-in rubrics or user-authored YAML files.

Built-in rubrics ship as YAML inside the package (``workspec/rubrics/``) so they
are present in the installed wheel, not just the source tree. They remain
first-class, editable data — contracts a team can read, diff, and own — and users
point at their own ``.yaml`` files the same way.

The directory is resolved at runtime, in order:
  1. ``$WORKSPEC_RUBRICS_DIR`` if set (explicit override).
  2. The packaged ``workspec/rubrics/`` shipped with the install.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from workspec.models import Spec


def _resolve_rubric_dir() -> Path:
    """Find the built-in rubrics directory: env override, else the packaged copy."""
    # Rubrics ship inside the package and are the primary location.
    packaged = Path(__file__).resolve().parent / "rubrics"
    override = os.environ.get("WORKSPEC_RUBRICS_DIR")
    candidates = [Path(override), packaged] if override else [packaged]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    # Nothing found: return the packaged location for a clear error later.
    return packaged  # pragma: no cover - defensive fallback


_RUBRIC_DIR = _resolve_rubric_dir()


def list_builtin_rubrics() -> dict[str, Path]:
    """Map built-in rubric name -> file path (name is the filename stem)."""
    if not _RUBRIC_DIR.is_dir():
        return {}
    return {p.stem: p for p in sorted(_RUBRIC_DIR.glob("*.yaml"))}


def _as_existing_path(source: str) -> Path | None:
    """Return an existing YAML file for ``source``, or None if it isn't a path.

    Accepts absolute, relative, and ``~``-prefixed paths. If ``source`` has no
    extension, also tries ``.yaml`` / ``.yml`` so ``--spec contracts/memo``
    works alongside ``--spec contracts/memo.yaml``.
    """
    expanded = Path(source).expanduser()
    candidates = [expanded]
    if expanded.suffix == "":
        candidates += [expanded.with_suffix(".yaml"), expanded.with_suffix(".yml")]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_spec(source: str) -> Spec:
    """Load a spec from a built-in rubric name OR a path to any YAML file.

    A rubric can live anywhere on disk — pass an absolute path, a relative path,
    or ``~/...``. Resolution order:

      1. If ``source`` is an existing file (optionally without a ``.yaml``/``.yml``
         extension), load that file.
      2. Else, if ``source`` matches a built-in rubric name, load that.
      3. Else, error with the list of built-ins.
    """
    path = _as_existing_path(source)
    if path is None:
        builtins = list_builtin_rubrics()
        if source in builtins:
            path = builtins[source]
        else:
            available = ", ".join(builtins) or "(none)"
            raise FileNotFoundError(
                f"No file at '{source}' and no built-in rubric by that name. "
                f"Pass a path to any .yaml contract, or use a built-in: {available}"
            )

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return Spec.model_validate(data)


def load_spec_from_yaml_str(text: str) -> Spec:
    """Parse a spec directly from a YAML string (handy for tests/derived specs)."""
    return Spec.model_validate(yaml.safe_load(text))
