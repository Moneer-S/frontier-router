"""Stub providers. Return canned responses. No keys or network required."""

from __future__ import annotations

from frontier_router.providers.base import Provider

_PROVIDER_VOICES = {
    "anthropic": "Claude (stub)",
    "openai":    "GPT (stub)",
    "google":    "Gemini (stub)",
    "xai":       "Grok (stub)",
}


class StubProvider(Provider):
    """Returns a deterministic fake response that echoes the routing decision.

    Useful for:
      - CI pipelines where API keys aren't available
      - Validating the routing logic without burning credits
      - Demos and documentation
    """

    is_stub = True

    def __init__(self, name: str):
        if name not in _PROVIDER_VOICES:
            raise ValueError(f"Unknown stub provider: {name!r}")
        self.name = name

    def query(self, task: str, context: dict) -> str:
        voice = _PROVIDER_VOICES[self.name]
        preview = task.strip().replace("\n", " ")
        if len(preview) > 80:
            preview = preview[:77] + "..."
        return (
            f"[{voice}] Received task: {preview!r}. "
            f"In real mode this would return an actual response. "
            f"Context keys: {sorted(context.keys()) if context else []}."
        )
