"""Pytest fixtures. Shared helper classes live in ``tests/helpers.py``."""

from __future__ import annotations

import pytest

from tests.helpers import Backend, FakeProvider, backend_params
from workspec.env import load_dotenv

load_dotenv()  # make .env keys visible to integration-backend detection


@pytest.fixture
def fake_provider() -> FakeProvider:
    return FakeProvider()


@pytest.fixture(params=backend_params())
def backend(request: pytest.FixtureRequest) -> Backend:
    """Parametrized over each reachable live backend; skips when none are available."""
    if request.param is None:  # pragma: no cover - only when nothing is reachable
        pytest.skip("no live backend available")
    return request.param
