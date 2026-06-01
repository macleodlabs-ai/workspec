"""Shared base for provider-backed agents.

Both the lint engine (``WorkSpecAgent``) and the drafter (``DraftAgent``) wrap a
``VerdictProvider`` and accept the same construction shape: either a ready
provider instance, or a provider *name* plus connection details that get routed
through :func:`workspec.providers.build_provider`.

This base captures exactly that shared constructor so each agent only adds its
own methods and (optionally) a different default ``max_tokens``.

To avoid circular imports, this module imports from ``workspec.providers`` only.
"""

from __future__ import annotations

from workspec.providers import VerdictProvider, build_provider


class ProviderBackedAgent:
    """Base class owning the provider-resolution constructor.

    Subclasses add their own behaviour and may pass a different ``max_tokens``
    default by calling ``super().__init__(...)``.

    Parameters
    ----------
    provider:
        Either a ready ``VerdictProvider`` instance (used as-is) or a backend
        name (``"anthropic"`` / ``"openai"``) routed through ``build_provider``.
    model, api_key, base_url, max_tokens:
        Passed to ``build_provider`` when ``provider`` is a name. Ignored when a
        provider instance is supplied directly.
    """

    def __init__(
        self,
        provider: VerdictProvider | str = "anthropic",
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 4096,
    ) -> None:
        if isinstance(provider, VerdictProvider):
            self.provider = provider
        else:
            self.provider = build_provider(
                provider,
                model=model,
                api_key=api_key,
                base_url=base_url,
                max_tokens=max_tokens,
            )
