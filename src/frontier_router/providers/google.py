"""Google (Gemini) provider. Lazy-imports the google-genai SDK."""

from __future__ import annotations

import os

from frontier_router.providers.base import Provider, ProviderError

DEFAULT_MODEL = "gemini-3.1-pro"


class GoogleProvider(Provider):
    """Calls the Gemini API via the official `google-genai` SDK.

    Install with: pip install frontier-router[google]
    Requires: GEMINI_API_KEY in environment (or GOOGLE_API_KEY).
    """

    name = "google"
    is_stub = False

    def __init__(self, model: str = DEFAULT_MODEL):
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ProviderError(
                "Neither GEMINI_API_KEY nor GOOGLE_API_KEY is set. "
                "Set one or use stub/hybrid mode."
            )
        try:
            from google import genai  # noqa: F401
        except ImportError as e:
            raise ProviderError(
                "google-genai SDK is not installed. "
                "Install with: pip install frontier-router[google]"
            ) from e
        self.model = model
        self._api_key = api_key

    def query(self, task: str, context: dict) -> str:
        from google import genai

        model = context.get("model", self.model)
        system_prompt = context.get("system_prompt")

        client = genai.Client(api_key=self._api_key)
        try:
            config = {}
            if system_prompt:
                config["system_instruction"] = system_prompt
            resp = client.models.generate_content(
                model=model,
                contents=task,
                config=config if config else None,
            )
            return resp.text or ""
        except Exception as e:
            raise ProviderError(f"Gemini call failed: {e}") from e
