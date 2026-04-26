# Architecture

How `frontier-router` makes a routing decision, from `query()` call to response.

---

## Overview

```
                    ┌─────────────────────┐
     query(task) ─> │    Router.query()   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  1. Explicit override?│
                    │     (provider=...)   │
                    └──────────┬──────────┘
                               │ no
                    ┌──────────▼──────────┐
                    │  2. Rules classifier │──── ~80% of traffic
                    │     (heuristics)     │
                    └──────────┬──────────┘
                               │ ambiguous
                    ┌──────────▼──────────┐
                    │  3. LLM-as-router    │──── opt-in, ~20% of traffic
                    │     (fallback)       │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  4. Capability map   │
                    │     (provider lookup)│
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  5. Provider call    │
                    │     (with fallback)  │
                    └──────────┬──────────┘
                               │
                         RouterResult
```

---

## Stage 1: Explicit override

If the caller passes `provider="grok"`, routing is bypassed. The task goes directly to the specified provider.

```python
router.query("anything", provider="grok")  # skips classification
```

This exists because (a) sometimes the user knows better than the router, and (b) during development you want to pin a provider to test behavior.

---

## Stage 2: Rules-based classifier

Fast, deterministic, zero API cost. A series of keyword and structure heuristics map the task to a capability class.

Examples:

- Task contains code blocks OR phrases like "refactor," "debug," "write a function" → `code_generation`
- Task is > 50K tokens OR contains file attachments with `.pdf`/`.docx` → `massive_doc_analysis`
- Task mentions "X," "Twitter," "trending," "what's happening" → `realtime_x_timeline`
- Task mentions "draw," "generate image," "illustrate" → `image_generation`

Rules live in [`src/frontier_router/routing/rules.py`](../src/frontier_router/routing/rules.py) and are intentionally simple. Each rule returns `(capability, confidence)`. If any rule returns confidence ≥ 0.7, the classifier commits and we skip Stage 3.

The rules cover ~80% of real queries because most queries announce their intent clearly. The ambiguous 20% are the interesting ones.

---

## Stage 3: LLM-as-router fallback

For tasks the rules can't classify with confidence, a cheap model (default: Claude Haiku, configurable) classifies the query into the capability taxonomy.

Off by default because (a) it adds a round-trip, (b) it costs money, (c) the rules cover most real traffic. Opt in with `Router(llm_fallback=True)`.

When enabled, classifications are cached at `.frontier_router_cache/` by query hash, so identical queries don't re-classify.

---

## Stage 4: Capability map

The capability → provider mapping is loaded from `capabilities.py`, which encodes the matrix in [`docs/capability-matrix.md`](capability-matrix.md) as Python data structures:

```python
CAPABILITY_MAP = {
    "code_generation":       ["anthropic", "openai"],
    "long_context_synthesis":["google", "anthropic"],
    "realtime_x_timeline":   ["xai", "openai"],
    ...
}
```

The first entry is primary, the rest are fallbacks in order.

---

## Stage 5: Provider call with fallback

The router calls the primary provider. If the call fails (rate limit, auth error, timeout, or missing key in stub mode), it falls through to the next provider in the fallback chain.

The result is returned as a `RouterResult`:

```python
@dataclass
class RouterResult:
    provider: str            # which provider actually answered
    capability: str          # which capability class was routed
    confidence: float        # classifier confidence
    response: str            # the actual response text
    fallback_chain: list[str]  # providers tried before success
    stub: bool               # whether this was a stub response
```

---

## Modes

### `stub` mode (default for testing)

Providers return canned responses that include the capability class and provider name. Useful for:

- Validating the routing logic without burning API credits
- CI pipelines where keys aren't available
- Demos

### `real` mode

Providers call their actual SDKs. Requires the matching API keys in environment variables. Missing keys degrade to stubs for that specific provider (with a warning), so partial key availability doesn't crash the router.

### `hybrid` mode

Runs the decision in real mode if keys are present for the routed provider, else falls back to a stub. Useful when you have some keys but not all.

---

## Extension points

**Adding a capability class:**
1. Add a rule to `rules.py` that detects the capability from task content
2. Add the provider mapping to `capabilities.py`
3. Update `docs/capability-matrix.md` with reasoning

**Adding a provider:**
1. Subclass `providers.base.Provider` and implement `async def query(task, context) -> str`
2. Register it in `providers/__init__.py`
3. Add it to the relevant capability maps

**Swapping the LLM-as-router classifier:**
Pass a custom classifier function to `Router(classifier=my_classifier)`. Signature: `(task: str, context: dict) -> tuple[str, float]`.

---

## What's explicitly out of scope

- **Streaming.** Non-trivial to design across four different SDK patterns. Roadmap item.
- **Tool use / function calling.** Provider-specific, hard to abstract cleanly. MCP is the right layer for this, not the router.
- **Caching of responses.** Only the classification is cached. Response caching is the caller's responsibility.
- **Cost optimization.** OpenRouter solves this better. The router picks on capability, not price.
- **Agentic loops.** Single-turn routing only. Multi-turn agent frameworks (LangGraph, etc.) can use this as their model-selection layer.
