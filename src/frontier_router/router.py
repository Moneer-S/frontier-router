"""The Router class, core routing logic."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field

from frontier_router.capabilities import (
    ALL_PROVIDERS,
    CAPABILITY_DESCRIPTIONS,
    CAPABILITY_MAP,
    Capability,
)
from frontier_router.providers import get_provider
from frontier_router.providers.base import Provider, ProviderError
from frontier_router.routing.rules import classify as classify_rules

logger = logging.getLogger(__name__)


@dataclass
class RouterResult:
    """Return type for Router.query().

    Contains the actual response plus all the metadata about how the routing
    decision was made, so callers can introspect or log it.
    """

    provider: str
    """The provider that actually produced the response."""

    capability: Capability
    """The capability class the task was routed into."""

    confidence: float
    """Classifier confidence in the capability decision, 0.0-1.0."""

    response: str
    """The response text from the provider."""

    fallback_chain: list[str] = field(default_factory=list)
    """Providers that were tried before the successful one (in order)."""

    stub: bool = False
    """Whether this response came from a stub rather than a real API call."""

    def explain(self) -> str:
        """Human-readable one-line explanation of the routing decision."""
        desc = CAPABILITY_DESCRIPTIONS.get(self.capability, str(self.capability))
        chain = ""
        if self.fallback_chain:
            chain = f" | fallback chain: {' -> '.join(self.fallback_chain + [self.provider])}"
        stub_marker = " [STUB]" if self.stub else ""
        return (
            f"routed to {self.provider}{stub_marker} "
            f"| capability: {self.capability.value} ({desc}) "
            f"| confidence: {self.confidence:.2f}{chain}"
        )


class Router:
    """Capability-aware router across frontier model providers.

    Modes:
        - "stub":   all providers are stubs. No keys needed. Good for testing.
        - "real":   providers call real SDKs. Requires matching API keys in env.
        - "hybrid": real calls when keys are present, stub fallback when not.

    Basic usage:

        >>> router = Router(mode="stub")
        >>> result = router.query("Refactor this function")
        >>> result.provider
        'anthropic'
        >>> result.capability
        <Capability.CODE_GENERATION: 'code_generation'>
    """

    def __init__(
        self,
        mode: str = "stub",
        llm_fallback: bool = False,
        classifier: Callable[[str, dict], tuple[Capability, float]] | None = None,
        providers: dict[str, Provider] | None = None,
    ):
        if mode not in {"stub", "real", "hybrid"}:
            raise ValueError(f"mode must be stub|real|hybrid, got {mode!r}")
        self.mode = mode
        self.llm_fallback = llm_fallback
        self._custom_classifier = classifier
        self._providers: dict[str, Provider] = providers or {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def query(
        self,
        task: str,
        context: dict | None = None,
        provider: str | None = None,
    ) -> RouterResult:
        """Route a task to the appropriate provider and return the response.

        Args:
            task:     The prompt / query text.
            context:  Optional context dict (file attachments, flags, etc).
            provider: Explicit provider override. Skips classification.

        Returns:
            RouterResult with the response and routing metadata.
        """
        context = context or {}

        # Stage 1: explicit override
        if provider is not None:
            if provider not in ALL_PROVIDERS:
                raise ValueError(
                    f"Unknown provider {provider!r}. "
                    f"Must be one of {ALL_PROVIDERS}."
                )
            return self._call_provider(
                provider,
                task,
                context,
                capability=Capability.GENERAL,
                confidence=1.0,
                fallback_chain=[],
            )

        # Stage 2 + 3: classification
        capability, confidence = self._classify(task, context)

        # Stage 4: capability map lookup
        chain = CAPABILITY_MAP.get(capability, CAPABILITY_MAP[Capability.GENERAL])

        # Stage 5: call with fallback
        tried: list[str] = []
        last_error: Exception | None = None
        for candidate in chain:
            try:
                return self._call_provider(
                    candidate,
                    task,
                    context,
                    capability=capability,
                    confidence=confidence,
                    fallback_chain=tried,
                )
            except ProviderError as e:
                logger.warning(
                    "Provider %s failed for %s: %s. Trying fallback.",
                    candidate,
                    capability.value,
                    e,
                )
                tried.append(candidate)
                last_error = e
                continue

        # Every provider in the chain failed.
        raise RuntimeError(
            f"All providers in chain {chain} failed for capability "
            f"{capability.value}. Last error: {last_error}"
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _classify(self, task: str, context: dict) -> tuple[Capability, float]:
        """Return (capability, confidence) for the task."""
        if self._custom_classifier is not None:
            return self._custom_classifier(task, context)

        capability, confidence = classify_rules(task, context)
        if confidence >= 0.7:
            return capability, confidence

        if self.llm_fallback:
            from frontier_router.routing.llm import classify as classify_llm
            return classify_llm(task, context, mode=self.mode)

        # Low confidence and no LLM fallback enabled: commit to the rules guess.
        # This is the pragmatic default, better to route somewhere than to error.
        return capability, confidence

    def _call_provider(
        self,
        provider_name: str,
        task: str,
        context: dict,
        capability: Capability,
        confidence: float,
        fallback_chain: list[str],
    ) -> RouterResult:
        """Resolve and invoke a provider, returning a RouterResult."""
        provider = self._resolve_provider(provider_name)
        response = provider.query(task, context)
        return RouterResult(
            provider=provider_name,
            capability=capability,
            confidence=confidence,
            response=response,
            fallback_chain=fallback_chain,
            stub=provider.is_stub,
        )

    def _resolve_provider(self, name: str) -> Provider:
        """Look up or construct a Provider for the given name, honoring mode."""
        if name in self._providers:
            return self._providers[name]

        if self.mode == "stub":
            provider = get_provider(name, real=False)
        elif self.mode == "real":
            provider = get_provider(name, real=True)
        else:  # hybrid
            # Use real if the matching env var is set, else stub for just this provider.
            env_var = _env_var_for(name)
            if env_var and os.environ.get(env_var):
                provider = get_provider(name, real=True)
            else:
                logger.info(
                    "Hybrid mode: no %s set, falling back to stub for %s.",
                    env_var,
                    name,
                )
                provider = get_provider(name, real=False)

        self._providers[name] = provider
        return provider


def _env_var_for(provider: str) -> str | None:
    """Map a provider name to its expected API key env var."""
    return {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GEMINI_API_KEY",
        "xai": "XAI_API_KEY",
    }.get(provider)
