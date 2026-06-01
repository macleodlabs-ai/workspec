"""Smoke tests for the package's public surface."""

from __future__ import annotations

import workspec


def test_public_exports_present() -> None:
    for name in workspec.__all__:
        assert hasattr(workspec, name), f"workspec.{name} missing"


def test_key_symbols_are_the_real_thing() -> None:
    from workspec.engine import WorkSpecAgent
    from workspec.spec_loader import load_spec

    assert workspec.WorkSpecAgent is WorkSpecAgent
    assert workspec.load_spec is load_spec


def test_version_is_a_string() -> None:
    assert isinstance(workspec.__version__, str)
    assert workspec.__version__.count(".") >= 1
