"""OpenAI (GPT) provider. Lazy-imports the openai SDK."""

from __future__ import annotations

import os

from frontier_router.providers.base import Provider, ProviderError

DEFAULT_MODEL = "gpt-5.5"


class OpenAIProvider(Provider):
    """Calls the OpenAI Chat Completions API via the official `openai` SDK.

    Install with: pip install frontier-router[openai]
    Requires: OPENAI_API_KEY in environment.
    """

    name = "openai"
    is_stub = False

    def __init__(self, model: str = DEFAULT_MODEL):
        if not os.environ.get("OPENAI_API_KEY"):
            raise ProviderError(
                "OPENAI_API_KEY is not set in environment. "
                "Set it or use stub/hybrid mode."
            )
        try:
            import openai  # noqa: F401
        except ImportError as e:
            raise ProviderError(
                "openai SDK is not installed. "
                "Install with: pip install frontier-router[openai]"
            ) from e
        self.model = model

    def query(self, task: str, context: dict) -> str:
        import openai

        model = context.get("model", self.model)
        system_prompt = context.get("system_prompt")
        max_tokens = context.get("max_tokens", 2048)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": task})

        client = openai.OpenAI()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            raise ProviderError(f"OpenAI call failed: {e}") from e
