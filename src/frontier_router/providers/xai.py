"""xAI (Grok) provider.

xAI provides an OpenAI-compatible API at https://api.x.ai/v1, so we use the
openai SDK with a custom base_url rather than requiring a separate xai SDK.
This also works with the native `xai-sdk` if installed.
"""

from __future__ import annotations

import os

from frontier_router.providers.base import Provider, ProviderError

DEFAULT_MODEL = "grok-4-latest"
XAI_BASE_URL = "https://api.x.ai/v1"


class XAIProvider(Provider):
    """Calls the xAI Grok API via the OpenAI-compatible endpoint.

    Install with: pip install frontier-router[xai]  (or [openai])
    Requires: XAI_API_KEY in environment.
    """

    name = "xai"
    is_stub = False

    def __init__(self, model: str = DEFAULT_MODEL):
        if not os.environ.get("XAI_API_KEY"):
            raise ProviderError(
                "XAI_API_KEY is not set in environment. "
                "Set it or use stub/hybrid mode."
            )
        try:
            import openai  # noqa: F401
        except ImportError as e:
            raise ProviderError(
                "openai SDK is not installed (used as xAI-compatible client). "
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

        client = openai.OpenAI(
            api_key=os.environ["XAI_API_KEY"],
            base_url=XAI_BASE_URL,
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            raise ProviderError(f"xAI call failed: {e}") from e
