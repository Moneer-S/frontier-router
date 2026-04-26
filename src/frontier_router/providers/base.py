"""Abstract provider interface. All providers conform to this shape."""

from __future__ import annotations

from abc import ABC, abstractmethod


class ProviderError(Exception):
    """Raised when a provider call fails (auth, rate limit, network, etc.).

    The router catches ProviderError and falls through to the next provider
    in the capability chain. Anything that's not a ProviderError bubbles up
    and is treated as a programming error.
    """


class Provider(ABC):
    """Abstract base for all providers (stub and real)."""

    #: Human-readable name, e.g. "anthropic".
    name: str

    #: Whether this is a stub provider. Surfaced on RouterResult so callers
    #: can tell whether a response was real or fake.
    is_stub: bool = False

    @abstractmethod
    def query(self, task: str, context: dict) -> str:
        """Send the task to the provider and return a response string.

        Args:
            task:    The prompt / query text.
            context: Optional context dict. Providers are free to inspect
                     keys like "file", "system_prompt", "max_tokens", etc.
                     and ignore anything they don't understand.

        Returns:
            The response text.

        Raises:
            ProviderError: if the call fails in a recoverable way (so the
                router can try a fallback provider).
        """
        ...
