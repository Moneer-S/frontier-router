"""LLM-as-router fallback classifier.

Invoked by Router when the rules classifier returns confidence < 0.7 AND the
caller opted in with `Router(llm_fallback=True)`. Uses Claude Haiku by default
as the cheapest/fastest frontier-tier classifier.

Classifications are cached to disk at `.frontier_router_cache/{sha256}.json`
so identical tasks don't re-classify.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

from frontier_router.capabilities import CAPABILITY_DESCRIPTIONS, Capability

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
CACHE_DIR = Path(".frontier_router_cache")


def classify(task: str, context: dict, mode: str) -> tuple[Capability, float]:
    """Classify a task via LLM, with on-disk caching and graceful fallback.

    Returns (Capability.GENERAL, 0.5) when mode == 'stub' so the router still
    gets a usable signal without making any API call. On any LLM/API error,
    returns (Capability.GENERAL, 0.3).
    """
    if mode == "stub":
        return Capability.GENERAL, 0.5

    cached = _read_cache(task)
    if cached is not None:
        return cached

    try:
        result = _call_claude(task)
    except Exception as e:
        logger.warning("LLM classifier failed (%s). Falling back to GENERAL.", e)
        return Capability.GENERAL, 0.3

    _write_cache(task, result)
    return result


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _cache_path(task: str) -> Path:
    digest = hashlib.sha256(task.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{digest}.json"


def _read_cache(task: str) -> tuple[Capability, float] | None:
    path = _cache_path(task)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return Capability(data["capability"]), float(data["confidence"])
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning("Corrupt classifier cache %s: %s. Ignoring.", path, e)
        return None


def _write_cache(task: str, result: tuple[Capability, float]) -> None:
    capability, confidence = result
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path(task).write_text(
            json.dumps({"capability": capability.value, "confidence": confidence}),
            encoding="utf-8",
        )
    except OSError as e:
        # Cache writes are best-effort; a read-only FS shouldn't crash classification.
        logger.warning("Could not write classifier cache: %s", e)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def _build_prompt(task: str) -> str:
    catalog = "\n".join(
        f"- {cap.value}: {desc}"
        for cap, desc in CAPABILITY_DESCRIPTIONS.items()
    )
    # Truncate ludicrously long tasks so the classifier call stays cheap;
    # we only need the shape/intent, not every token.
    snippet = task if len(task) <= 8000 else task[:8000] + " [...truncated for classifier]"
    return (
        "You are a router classifying a user task into one capability class.\n"
        "Return ONLY a single-line JSON object with keys 'capability' and 'confidence'.\n"
        "No prose, no markdown fences.\n\n"
        "Capabilities:\n"
        f"{catalog}\n\n"
        "Task:\n"
        f"{snippet}\n\n"
        'Respond exactly like: {"capability": "code_generation", "confidence": 0.87}'
    )


def _call_claude(task: str) -> tuple[Capability, float]:
    import anthropic  # noqa: F401, lazy, only needed on the real path

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    import anthropic as _anthropic
    client = _anthropic.Anthropic()
    message = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=128,
        messages=[{"role": "user", "content": _build_prompt(task)}],
    )
    text = "".join(
        block.text for block in message.content if getattr(block, "type", None) == "text"
    ).strip()
    return _parse_response(text)


def _parse_response(text: str) -> tuple[Capability, float]:
    """Pull the JSON object out of the model response and validate it."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object in response: {text!r}")
    payload = json.loads(text[start : end + 1])
    capability = Capability(payload["capability"])
    confidence = float(payload["confidence"])
    confidence = max(0.0, min(1.0, confidence))
    return capability, confidence
