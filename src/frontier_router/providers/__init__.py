"""Provider implementations.

Each provider has a stub version and a real version. The stub version is
zero-dependency; the real version imports the provider's SDK lazily so the
core package stays lightweight.
"""

from frontier_router.providers.base import Provider, ProviderError


def get_provider(name: str, real: bool = False) -> Provider:
    """Factory: return a Provider instance for the given name.

    Args:
        name: provider identifier ("anthropic", "openai", "google", "xai")
        real: if True, return the real SDK-backed provider; else return a stub.

    Raises:
        ValueError: if name is unknown.
        ProviderError: if real=True and the SDK / key is unavailable.
    """
    if real:
        if name == "anthropic":
            from frontier_router.providers.anthropic import AnthropicProvider
            return AnthropicProvider()
        if name == "openai":
            from frontier_router.providers.openai import OpenAIProvider
            return OpenAIProvider()
        if name == "google":
            from frontier_router.providers.google import GoogleProvider
            return GoogleProvider()
        if name == "xai":
            from frontier_router.providers.xai import XAIProvider
            return XAIProvider()
        raise ValueError(f"Unknown provider: {name!r}")

    from frontier_router.providers.stubs import StubProvider
    return StubProvider(name)


__all__ = ["Provider", "ProviderError", "get_provider"]
