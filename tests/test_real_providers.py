"""Tests for the enhanced real-mode providers.

We don't hit any real API. Each test:
  1. monkeypatches the SDK's top-level module into ``sys.modules``,
  2. constructs the enhanced provider with the relevant env var set,
  3. calls ``query`` and asserts the right code path fired.

That gives us deterministic coverage of the capability-dispatch logic
without any keys or network.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from frontier_router.capabilities import Capability

# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------


class _FakeOpenAIChoiceMessage:
    def __init__(self, content):
        self.content = content


class _FakeOpenAIChoice:
    def __init__(self, content):
        self.message = _FakeOpenAIChoiceMessage(content)


class _FakeOpenAIChatResponse:
    def __init__(self, content):
        self.choices = [_FakeOpenAIChoice(content)]


class _FakeOpenAIImageItem:
    def __init__(self, b64_json=None, url=None):
        self.b64_json = b64_json
        self.url = url


class _FakeOpenAIImageResponse:
    def __init__(self, items):
        self.data = items


def _install_fake_openai(monkeypatch, *, chat_content="hi", image_b64="ZmFrZQ=="):
    """Install a fake `openai` module returning predictable responses.

    Returns the MagicMock client instance so tests can assert on call args.
    """
    fake = types.ModuleType("openai")

    client = MagicMock()
    client.chat.completions.create.return_value = _FakeOpenAIChatResponse(chat_content)
    client.images.generate.return_value = _FakeOpenAIImageResponse(
        [_FakeOpenAIImageItem(b64_json=image_b64)]
    )

    fake.OpenAI = MagicMock(return_value=client)
    monkeypatch.setitem(sys.modules, "openai", fake)
    return client


def _install_fake_genai(monkeypatch):
    google_pkg = types.ModuleType("google")
    genai_mod = types.ModuleType("google.genai")

    client = MagicMock()
    client.models.generate_content.return_value = MagicMock(text="gemini chat")
    client.models.generate_images.return_value = MagicMock(
        generated_images=[MagicMock(image=MagicMock(image_bytes=b"PNGDATA"))]
    )
    client.files.upload.return_value = MagicMock(name="uploaded-file-handle")

    genai_mod.Client = MagicMock(return_value=client)
    google_pkg.genai = genai_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", google_pkg)
    monkeypatch.setitem(sys.modules, "google.genai", genai_mod)
    return client


def _install_fake_anthropic(monkeypatch):
    anthropic_mod = types.ModuleType("anthropic")
    client = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "claude on a pdf"
    client.messages.create.return_value = MagicMock(content=[text_block])
    anthropic_mod.Anthropic = MagicMock(return_value=client)
    monkeypatch.setitem(sys.modules, "anthropic", anthropic_mod)
    return client


# ---------------------------------------------------------------------------
# OpenAI: images.generate dispatch
# ---------------------------------------------------------------------------


def test_openai_routes_image_to_images_generate(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    client = _install_fake_openai(monkeypatch)

    from frontier_router.real_providers import EnhancedOpenAIProvider

    provider = EnhancedOpenAIProvider()
    result = provider.query(
        "Generate an image of a sunset",
        {"_capability": Capability.IMAGE_GENERATION.value},
    )
    assert result.startswith("data:image/png;base64,")
    client.images.generate.assert_called_once()
    client.chat.completions.create.assert_not_called()


def test_openai_chat_path_for_non_image_task(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    client = _install_fake_openai(monkeypatch, chat_content="refactored code")

    from frontier_router.real_providers import EnhancedOpenAIProvider

    provider = EnhancedOpenAIProvider()
    result = provider.query("Refactor this function", {})
    assert result == "refactored code"
    client.chat.completions.create.assert_called_once()
    client.images.generate.assert_not_called()


def test_openai_image_dispatch_via_reclassify(monkeypatch):
    """Without _capability in context, the wrapper re-runs rules and still routes correctly."""
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    client = _install_fake_openai(monkeypatch)

    from frontier_router.real_providers import EnhancedOpenAIProvider

    provider = EnhancedOpenAIProvider()
    provider.query("Draw an image of a cat", {})
    client.images.generate.assert_called_once()


# ---------------------------------------------------------------------------
# xAI: Live Search dispatch
# ---------------------------------------------------------------------------


def test_xai_live_search_for_x_timeline(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "test")
    client = _install_fake_openai(monkeypatch, chat_content="trending: AI safety")

    from frontier_router.real_providers import EnhancedXAIProvider

    provider = EnhancedXAIProvider()
    out = provider.query(
        "What's trending on X this morning?",
        {"_capability": Capability.REALTIME_X_TIMELINE.value},
    )
    assert "trending" in out

    # Live Search should be passed via extra_body.search_parameters
    call_kwargs = client.chat.completions.create.call_args.kwargs
    extra_body = call_kwargs.get("extra_body") or {}
    assert "search_parameters" in extra_body
    sources = extra_body["search_parameters"].get("sources", [])
    assert {"type": "x"} in sources


def test_xai_chat_path_when_not_x_timeline(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "test")
    client = _install_fake_openai(monkeypatch, chat_content="just chat")

    from frontier_router.real_providers import EnhancedXAIProvider

    provider = EnhancedXAIProvider()
    provider.query("Hello there", {})
    call_kwargs = client.chat.completions.create.call_args.kwargs
    # Plain chat path should not pass extra_body.
    assert "extra_body" not in call_kwargs or not call_kwargs["extra_body"]


# ---------------------------------------------------------------------------
# Google: Imagen + file upload
# ---------------------------------------------------------------------------


def test_google_routes_image_to_imagen(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    client = _install_fake_genai(monkeypatch)

    from frontier_router.real_providers import EnhancedGoogleProvider

    provider = EnhancedGoogleProvider()
    out = provider.query(
        "Create an image of a mountain",
        {"_capability": Capability.IMAGE_GENERATION.value},
    )
    assert out.startswith("data:image/png;base64,")
    client.models.generate_images.assert_called_once()
    client.models.generate_content.assert_not_called()


def test_google_uploads_file_for_massive_doc(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    client = _install_fake_genai(monkeypatch)

    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake\n")

    from frontier_router.real_providers import EnhancedGoogleProvider

    provider = EnhancedGoogleProvider()
    out = provider.query(
        "Summarize this report",
        {"_capability": Capability.MASSIVE_DOC_ANALYSIS.value, "file": str(pdf)},
    )
    assert out == "gemini chat"
    client.files.upload.assert_called_once()
    client.models.generate_content.assert_called_once()


def test_google_chat_path_for_plain_task(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    client = _install_fake_genai(monkeypatch)

    from frontier_router.real_providers import EnhancedGoogleProvider

    provider = EnhancedGoogleProvider()
    out = provider.query("Tell me a fact about lighthouses", {})
    assert out == "gemini chat"
    client.models.generate_images.assert_not_called()
    client.files.upload.assert_not_called()


# ---------------------------------------------------------------------------
# Anthropic: PDF document blocks
# ---------------------------------------------------------------------------


def test_anthropic_attaches_pdf_for_massive_doc(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    client = _install_fake_anthropic(monkeypatch)

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake\n")

    from frontier_router.real_providers import EnhancedAnthropicProvider

    provider = EnhancedAnthropicProvider()
    out = provider.query(
        "Summarize this PDF",
        {"_capability": Capability.MASSIVE_DOC_ANALYSIS.value, "file": str(pdf)},
    )
    assert out == "claude on a pdf"

    call_kwargs = client.messages.create.call_args.kwargs
    msgs = call_kwargs["messages"]
    user_content = msgs[0]["content"]
    types_present = {block.get("type") for block in user_content}
    assert "document" in types_present
    assert "text" in types_present


def test_anthropic_chat_path_when_not_pdf(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    client = _install_fake_anthropic(monkeypatch)

    from frontier_router.real_providers import EnhancedAnthropicProvider

    provider = EnhancedAnthropicProvider()
    out = provider.query("Refactor this login helper", {})
    assert out == "claude on a pdf"  # fake response
    call_kwargs = client.messages.create.call_args.kwargs
    msgs = call_kwargs["messages"]
    # plain string content, not a list
    assert isinstance(msgs[0]["content"], str)


# ---------------------------------------------------------------------------
# Lazy factory
# ---------------------------------------------------------------------------


def test_enhanced_providers_factory_is_lazy(monkeypatch):
    """No keys set: factory still returns a dict; missing keys only error on call."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)

    from frontier_router.real_providers import enhanced_providers

    providers = enhanced_providers()
    assert set(providers.keys()) == {"anthropic", "openai", "google", "xai"}

    # Touching one without keys should error from the *underlying* provider,
    # not from the factory. ProviderError is the contract.
    from frontier_router.providers.base import ProviderError
    with pytest.raises(ProviderError):
        providers["openai"].query("hi", {})
