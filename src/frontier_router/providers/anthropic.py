"""Anthropic (Claude) provider. Lazy-imports the anthropic SDK."""

from __future__ import annotations

import os

from frontier_router.providers.base import Provider, ProviderError

# Default model. Override by passing model= in context.
DEFAULT_MODEL = "claude-sonnet-4-6"


class AnthropicProvider(Provider):
    """Calls the Anthropic Messages API via the official `anthropic` SDK.

    Install with: pip install frontier-router[anthropic]
    Requires: ANTHROPIC_API_KEY in environment.
    """

    name = "anthropic"
    is_stub = False

    def __init__(self, model: str = DEFAULT_MODEL):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ProviderError(
                "ANTHROPIC_API_KEY is not set in environment. "
                "Set it or use stub/hybrid mode."
            )
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise ProviderError(
                "anthropic SDK is not installed. "
                "Install with: pip install frontier-router[anthropic]"
            ) from e
        self.model = model

    def query(self, task: str, context: dict) -> str:
        import anthropic

        model = context.get("model", self.model)
        system_prompt = context.get("system_prompt")
        max_tokens = context.get("max_tokens", 2048)

        client = anthropic.Anthropic()
        try:
            kwargs = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": task}],
            }
            if system_prompt:
                kwargs["system"] = system_prompt
            message = client.messages.create(**kwargs)
            # Concatenate text blocks; ignore non-text blocks for v0.1.
            return "".join(
                block.text for block in message.content if getattr(block, "type", None) == "text"
            )
        except Exception as e:
            raise ProviderError(f"Anthropic call failed: {e}") from e
