"""Unit tests for spec/rubric loading and resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from workspec import spec_loader
from workspec.models import Spec
from workspec.spec_loader import (
    list_builtin_rubrics,
    load_spec,
    load_spec_from_yaml_str,
)

_YAML = """
type: sample
title: Sample Contract
description: For tests.
must_include:
  - an owner
acceptance_tests:
  - It has an owner.
"""


def test_list_builtin_rubrics_returns_known_contracts() -> None:
    names = list_builtin_rubrics()
    assert "email_reply" in names
    assert "decision_memo" in names
    assert "ai_delegation_brief" in names
    # status_update was moved to examples/, so it is no longer a built-in
    assert "status_update" not in names
    for path in names.values():
        assert path.suffix == ".yaml"


def test_load_spec_by_builtin_name() -> None:
    spec = load_spec("email_reply")
    assert isinstance(spec, Spec)
    assert spec.type


def test_load_spec_from_yaml_str() -> None:
    spec = load_spec_from_yaml_str(_YAML)
    assert spec.type == "sample"
    assert spec.must_include == ["an owner"]


def test_load_spec_from_absolute_and_relative_path(tmp_path: Path, monkeypatch) -> None:
    f = tmp_path / "contract.yaml"
    f.write_text(_YAML, encoding="utf-8")

    assert load_spec(str(f)).type == "sample"  # absolute

    monkeypatch.chdir(tmp_path)
    assert load_spec("contract.yaml").type == "sample"  # relative
    assert load_spec("contract").type == "sample"  # extension optional


def test_load_spec_accepts_yml_extension(tmp_path: Path) -> None:
    f = tmp_path / "c.yml"
    f.write_text(_YAML, encoding="utf-8")
    assert load_spec(str(tmp_path / "c")).type == "sample"


def test_load_spec_expanduser(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "mine.yaml").write_text(_YAML, encoding="utf-8")
    assert load_spec("~/mine.yaml").type == "sample"


def test_load_spec_missing_raises_with_builtins_listed() -> None:
    with pytest.raises(FileNotFoundError) as exc:
        load_spec("does-not-exist-anywhere")
    msg = str(exc.value)
    assert "email_reply" in msg


def test_resolve_rubric_dir_env_override(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "custom.yaml").write_text(_YAML, encoding="utf-8")
    monkeypatch.setenv("WORKSPEC_RUBRICS_DIR", str(tmp_path))
    resolved = spec_loader._resolve_rubric_dir()
    assert resolved == tmp_path


def test_resolve_rubric_dir_is_inside_package(monkeypatch) -> None:
    # With no override, rubrics resolve to the packaged location next to the module,
    # so they ship inside the wheel rather than living at the repo root.
    monkeypatch.delenv("WORKSPEC_RUBRICS_DIR", raising=False)
    resolved = spec_loader._resolve_rubric_dir()
    package_dir = Path(spec_loader.__file__).resolve().parent
    assert resolved == package_dir / "rubrics"
    assert resolved.is_dir()
