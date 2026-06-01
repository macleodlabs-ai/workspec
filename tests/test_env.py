"""Unit tests for the tiny .env loader."""

from __future__ import annotations

from pathlib import Path

from workspec.env import load_dotenv


def test_loads_pairs_without_override(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "# a comment",
                "",
                "FOO=bar",
                'QUOTED="quoted value"',
                "SINGLE='single'",
                "export EXPORTED=exp",
                "ALREADY=from_file",
                "NO_VALUE_LINE_WITHOUT_EQUALS",
            ]
        ),
        encoding="utf-8",
    )
    # Touch every var via monkeypatch so load_dotenv's writes to os.environ are
    # restored at teardown and don't leak into other tests.
    for key in ("FOO", "QUOTED", "SINGLE", "EXPORTED"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ALREADY", "from_shell")  # must NOT be overwritten

    loaded = load_dotenv(start=tmp_path)

    assert loaded == tmp_path / ".env"
    import os

    assert os.environ["FOO"] == "bar"
    assert os.environ["QUOTED"] == "quoted value"
    assert os.environ["SINGLE"] == "single"
    assert os.environ["EXPORTED"] == "exp"
    assert os.environ["ALREADY"] == "from_shell"  # existing var wins


def test_walks_up_to_find_env(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / ".env").write_text("DEEP=yes\n", encoding="utf-8")
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)
    monkeypatch.delenv("DEEP", raising=False)

    loaded = load_dotenv(start=nested)

    assert loaded == tmp_path / ".env"
    import os

    assert os.environ["DEEP"] == "yes"


def test_returns_none_when_absent(tmp_path: Path) -> None:
    assert load_dotenv(start=tmp_path) is None
