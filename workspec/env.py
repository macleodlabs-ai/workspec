# Load KEY=VALUE pairs from a .env file into os.environ (no override of existing vars).
# Called from providers before resolving API keys so repo-root .env works for CLI and demo.

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(start: Path | None = None) -> Path | None:
    """Load the nearest ``.env`` walking up from ``start`` or the current working directory.

    Existing environment variables are never overwritten. Returns the path loaded, or
    ``None`` if no ``.env`` file was found.
    """
    cur = (start or Path.cwd()).resolve()
    for directory in (cur, *cur.parents):
        env_file = directory / ".env"
        if env_file.is_file():
            _apply_env_file(env_file)
            return env_file
        if directory.parent == directory:
            break
    return None


def _apply_env_file(path: Path) -> None:
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ[key] = value
