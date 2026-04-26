"""Tests for Router end-to-end behavior in stub mode."""

from __future__ import annotations

import pytest

from frontier_router import Capability, Router, RouterResult
from frontier_router.providers.base import Provider, ProviderError

# --- Basic stub mode flow ---------------------------------------------------


def test_stub_mode_returns_stub_response():
    router = Router(mode="stub")
    result = router.query("Refactor this login helper")
    assert isinstance(result, RouterResult)
    assert result.stub is True
    assert result.provider == "anthropic"  # code_generation → Claude
    assert result.capability == Capability.CODE_GENERATION
    assert "stub" in result.response.lower()


def test_stub_routes_long_doc_to_google():
    router = Router(mode="stub")
    result = router.query("Summarize this report in detail", context={"file": "thesis.pdf"})
    assert result.provider == "google"
    assert result.capability == Capability.MASSIVE_DOC_ANALYSIS


def test_stub_routes_realtime_x_to_xai():
    router = Router(mode="stub")
    result = router.query("What's trending on X this afternoon?")
    assert result.provider == "xai"
    assert result.capability == Capability.REALTIME_X_TIMELINE


# --- Explicit overrides -----------------------------------------------------


def test_explicit_provider_override_bypasses_classification():
    router = Router(mode="stub")
    result = router.query("Write a function to sum a list", provider="xai")
    assert result.provider == "xai"
    assert result.capability == Capability.GENERAL
    assert result.confidence == 1.0


def test_invalid_provider_raises_value_error():
    router = Router(mode="stub")
    with pytest.raises(ValueError, match="Unknown provider"):
        router.query("hi", provider="bogus")


def test_invalid_mode_raises_value_error():
    with pytest.raises(ValueError, match="mode must be"):
        Router(mode="nope")


# --- Fallback chain ---------------------------------------------------------


class _BrokenProvider(Provider):
    """Provider that always fails, exercises the fallback path."""

    is_stub = False

    def __init__(self, name: str):
        self.name = name

    def query(self, task, context):
        raise ProviderError(f"{self.name} is down for the test")


class _EchoProvider(Provider):
    """Provider that always succeeds with a known response."""

    is_stub = False

    def __init__(self, name: str):
        self.name = name

    def query(self, task, context):
        return f"[{self.name}] ok"


def test_fallback_chain_populated_when_primary_fails():
    # code_generation routes anthropic → openai → google.
    providers = {
        "anthropic": _BrokenProvider("anthropic"),
        "openai": _EchoProvider("openai"),
    }
    router = Router(mode="stub", providers=providers)
    result = router.query("Refactor this function")
    assert result.provider == "openai"
    assert result.fallback_chain == ["anthropic"]
    assert "[openai] ok" in result.response


def test_all_providers_failing_raises_runtime_error():
    providers = {
        "anthropic": _BrokenProvider("anthropic"),
        "openai": _BrokenProvider("openai"),
        "google": _BrokenProvider("google"),
    }
    router = Router(mode="stub", providers=providers)
    with pytest.raises(RuntimeError, match="All providers"):
        router.query("Refactor this function")


# --- RouterResult.explain() -------------------------------------------------


def test_explain_contains_key_fields():
    router = Router(mode="stub")
    result = router.query("Refactor this login helper")
    text = result.explain()
    assert "anthropic" in text
    assert "code_generation" in text
    assert "confidence" in text
    assert "STUB" in text


def test_explain_includes_fallback_chain_when_populated():
    providers = {
        "anthropic": _BrokenProvider("anthropic"),
        "openai": _EchoProvider("openai"),
    }
    router = Router(mode="stub", providers=providers)
    result = router.query("Refactor this function")
    text = result.explain()
    assert "fallback chain" in text
    assert "anthropic -> openai" in text


# --- Custom classifier ------------------------------------------------------


def test_custom_classifier_is_used():
    def always_image(task, context):
        return Capability.IMAGE_GENERATION, 0.99

    router = Router(mode="stub", classifier=always_image)
    result = router.query("whatever")
    assert result.capability == Capability.IMAGE_GENERATION
    # IMAGE_GENERATION primary is openai.
    assert result.provider == "openai"
