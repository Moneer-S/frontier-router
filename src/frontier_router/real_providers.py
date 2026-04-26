"""Enhanced real-mode providers that wire capability-specific endpoints.

The base providers in ``frontier_router.providers.*`` only call chat
completion endpoints, they're correct for ~7 of the 12 capability classes
but leave the rest as text-only. These wrappers fill the gaps:

- ``EnhancedOpenAIProvider``    -> ``images.generate`` for ``image_generation``
- ``EnhancedXAIProvider``       -> Live Search ``search_parameters`` for
                                   ``realtime_x_timeline``
- ``EnhancedGoogleProvider``    -> Imagen for ``image_generation``,
                                   file uploads for ``massive_doc_analysis``
- ``EnhancedAnthropicProvider`` -> PDF document content blocks for
                                   ``massive_doc_analysis``

Each wrapper re-runs the rules classifier on the task to determine which
endpoint to use, so dispatch works whether the wrapper is called directly
or through Router.

Capabilities that need infrastructure outside an API key (agentic browser,
in-vehicle Tesla, personalized Workspace) are intentionally NOT covered.
They route to the same providers but go through the chat path.

Usage::

    from frontier_router import Router
    from frontier_router.real_providers import enhanced_providers

    router = Router(mode="real", providers=enhanced_providers())
    result = router.query("Generate an image of a sunset over mountains")
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Any

from frontier_router.capabilities import Capability
from frontier_router.providers.anthropic import AnthropicProvider
from frontier_router.providers.base import Provider, ProviderError
from frontier_router.providers.google import GoogleProvider
from frontier_router.providers.openai import OpenAIProvider
from frontier_router.providers.xai import XAIProvider
from frontier_router.routing.rules import classify as classify_rules

logger = logging.getLogger(__name__)

DEFAULT_OPENAI_IMAGE_MODEL = "gpt-image-1"
DEFAULT_GOOGLE_IMAGE_MODEL = "imagen-4.0-generate-001"
DEFAULT_GOOGLE_DOC_MODEL = "gemini-3.1-pro"


def _capability_from_context(task: str, context: dict) -> Capability:
    """Pull the routed capability out of context, or recompute from rules.

    Router doesn't currently inject the capability, so we re-classify on
    entry. If a future Router revision puts ``_capability`` into context,
    we honor that without re-running the rules.
    """
    cap = context.get("_capability")
    if cap is not None:
        try:
            return Capability(cap)
        except ValueError:
            pass
    capability, _ = classify_rules(task, context)
    return capability


def _read_pdf_b64(path: str | os.PathLike) -> str:
    return base64.standard_b64encode(Path(path).read_bytes()).decode("ascii")


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


class EnhancedOpenAIProvider(OpenAIProvider):
    """OpenAI provider that dispatches to ``images.generate`` for image tasks."""

    def __init__(
        self,
        model: str | None = None,
        image_model: str = DEFAULT_OPENAI_IMAGE_MODEL,
    ):
        if model is None:
            super().__init__()
        else:
            super().__init__(model=model)
        self.image_model = image_model

    def query(self, task: str, context: dict) -> str:
        if _capability_from_context(task, context) == Capability.IMAGE_GENERATION:
            return self._generate_image(task, context)
        return super().query(task, context)

    def _generate_image(self, task: str, context: dict) -> str:
        import openai

        model = context.get("image_model", self.image_model)
        size = context.get("size", "1024x1024")
        client = openai.OpenAI()
        try:
            resp = client.images.generate(model=model, prompt=task, size=size, n=1)
            item = resp.data[0]
            b64 = getattr(item, "b64_json", None)
            if b64:
                return f"data:image/png;base64,{b64}"
            url = getattr(item, "url", None)
            if url:
                return url
            raise ProviderError("OpenAI image response had neither b64_json nor url")
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(f"OpenAI image call failed: {e}") from e


# ---------------------------------------------------------------------------
# xAI
# ---------------------------------------------------------------------------


class EnhancedXAIProvider(XAIProvider):
    """xAI provider that flips on Live Search for X-timeline queries.

    xAI's chat completions endpoint accepts a ``search_parameters`` field
    that pulls live data from X (and optionally web/news/RSS). For
    ``realtime_x_timeline`` we restrict sources to X so the moat fires.
    """

    def query(self, task: str, context: dict) -> str:
        if _capability_from_context(task, context) == Capability.REALTIME_X_TIMELINE:
            return self._chat_with_live_search(task, context)
        return super().query(task, context)

    def _chat_with_live_search(self, task: str, context: dict) -> str:
        import openai

        model = context.get("model", self.model)
        system_prompt = context.get("system_prompt")
        max_tokens = context.get("max_tokens", 2048)

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": task})

        search_parameters = context.get("search_parameters") or {
            "mode": "on",
            "sources": [{"type": "x"}],
            "return_citations": True,
        }
        client = openai.OpenAI(
            api_key=os.environ["XAI_API_KEY"],
            base_url="https://api.x.ai/v1",
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                extra_body={"search_parameters": search_parameters},
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            raise ProviderError(f"xAI Live Search call failed: {e}") from e


# ---------------------------------------------------------------------------
# Google
# ---------------------------------------------------------------------------


class EnhancedGoogleProvider(GoogleProvider):
    """Gemini provider with Imagen and file-upload paths."""

    def __init__(
        self,
        model: str | None = None,
        image_model: str = DEFAULT_GOOGLE_IMAGE_MODEL,
        doc_model: str = DEFAULT_GOOGLE_DOC_MODEL,
    ):
        if model is None:
            super().__init__()
        else:
            super().__init__(model=model)
        self.image_model = image_model
        self.doc_model = doc_model

    def query(self, task: str, context: dict) -> str:
        capability = _capability_from_context(task, context)
        if capability == Capability.IMAGE_GENERATION:
            return self._generate_image(task, context)
        if capability == Capability.MASSIVE_DOC_ANALYSIS and context.get("file"):
            return self._chat_with_file(task, context)
        return super().query(task, context)

    def _generate_image(self, task: str, context: dict) -> str:
        from google import genai

        model = context.get("image_model", self.image_model)
        client = genai.Client(api_key=self._api_key)
        try:
            resp = client.models.generate_images(model=model, prompt=task)
            images = getattr(resp, "generated_images", None) or []
            if not images:
                raise ProviderError("Imagen returned no images")
            img = images[0].image
            data = getattr(img, "image_bytes", None)
            if data:
                b64 = base64.standard_b64encode(data).decode("ascii")
                return f"data:image/png;base64,{b64}"
            url = getattr(img, "url", None)
            if url:
                return url
            raise ProviderError("Imagen response had neither bytes nor url")
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(f"Gemini image call failed: {e}") from e

    def _chat_with_file(self, task: str, context: dict) -> str:
        from google import genai

        path = context["file"]
        model = context.get("model", self.doc_model)
        client = genai.Client(api_key=self._api_key)
        try:
            uploaded = client.files.upload(file=str(path))
            resp = client.models.generate_content(
                model=model,
                contents=[task, uploaded],
            )
            return resp.text or ""
        except Exception as e:
            raise ProviderError(f"Gemini file-upload call failed: {e}") from e


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


class EnhancedAnthropicProvider(AnthropicProvider):
    """Claude provider that attaches PDFs as document blocks for doc analysis."""

    def query(self, task: str, context: dict) -> str:
        capability = _capability_from_context(task, context)
        path = context.get("file")
        if (
            capability == Capability.MASSIVE_DOC_ANALYSIS
            and path
            and str(path).lower().endswith(".pdf")
        ):
            return self._chat_with_pdf(task, context, str(path))
        return super().query(task, context)

    def _chat_with_pdf(self, task: str, context: dict, path: str) -> str:
        import anthropic

        model = context.get("model", self.model)
        system_prompt = context.get("system_prompt")
        max_tokens = context.get("max_tokens", 2048)

        try:
            pdf_b64 = _read_pdf_b64(path)
        except OSError as e:
            raise ProviderError(f"Could not read PDF at {path}: {e}") from e

        content = [
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": pdf_b64,
                },
            },
            {"type": "text", "text": task},
        ]
        client = anthropic.Anthropic()
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": content}],
            }
            if system_prompt:
                kwargs["system"] = system_prompt
            message = client.messages.create(**kwargs)
            return "".join(
                block.text
                for block in message.content
                if getattr(block, "type", None) == "text"
            )
        except Exception as e:
            raise ProviderError(f"Anthropic PDF call failed: {e}") from e


# ---------------------------------------------------------------------------
# Lazy factory
# ---------------------------------------------------------------------------


class _LazyProvider(Provider):
    """Defers construction (and key checks) until first ``query`` call.

    Lets ``enhanced_providers()`` return a full dict without forcing the
    user to have keys for providers they won't actually route to.
    """

    is_stub = False

    def __init__(self, name: str, builder):
        self.name = name
        self._builder = builder
        self._inner: Provider | None = None

    def _resolve(self) -> Provider:
        if self._inner is None:
            self._inner = self._builder()
        return self._inner

    def query(self, task: str, context: dict) -> str:
        return self._resolve().query(task, context)


def enhanced_providers() -> dict[str, Provider]:
    """Return a provider dict suitable for ``Router(providers=...)``.

    Each value is a lazy wrapper, the underlying real provider is only
    constructed (and its API key checked) the first time it's actually
    called. So partial-key setups work: missing keys for providers you
    don't route to never error out.
    """
    return {
        "anthropic": _LazyProvider("anthropic", EnhancedAnthropicProvider),
        "openai": _LazyProvider("openai", EnhancedOpenAIProvider),
        "google": _LazyProvider("google", EnhancedGoogleProvider),
        "xai": _LazyProvider("xai", EnhancedXAIProvider),
    }


__all__ = [
    "EnhancedAnthropicProvider",
    "EnhancedGoogleProvider",
    "EnhancedOpenAIProvider",
    "EnhancedXAIProvider",
    "enhanced_providers",
]
