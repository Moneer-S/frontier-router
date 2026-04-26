"""Three minimal examples that run end-to-end in stub mode (no API keys)."""

from __future__ import annotations

from frontier_router import Router
from frontier_router.providers.base import Provider, ProviderError


def example_basic_query() -> None:
    print("=== 1. basic stub query ===")
    router = Router(mode="stub")
    result = router.query("Refactor this login helper to use dependency injection")
    print(result.response)
    print(result.explain())
    print()


def example_provider_override() -> None:
    print("=== 2. explicit provider override ===")
    router = Router(mode="stub")
    result = router.query(
        "What's trending across the team Slack today?",
        provider="xai",  # force Grok, bypasses the classifier
    )
    print(result.response)
    print(result.explain())
    print()


class _FlakyProvider(Provider):
    """Always raises ProviderError, simulates an outage on the primary."""

    is_stub = False

    def __init__(self, name: str):
        self.name = name

    def query(self, task, context):
        raise ProviderError(f"{self.name} is down (simulated)")


def example_fallback_chain() -> None:
    print("=== 3. fallback chain in action ===")
    # code_generation routes anthropic -> openai -> google. We break anthropic
    # and let the router fall through.
    from frontier_router.providers import get_provider

    providers = {
        "anthropic": _FlakyProvider("anthropic"),
        "openai": get_provider("openai", real=False),  # stub
        "google": get_provider("google", real=False),  # stub
    }
    router = Router(mode="stub", providers=providers)
    result = router.query("Debug this stack trace")
    print(result.response)
    print(result.explain())
    print()


if __name__ == "__main__":
    example_basic_query()
    example_provider_override()
    example_fallback_chain()
