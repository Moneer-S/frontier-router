# frontier-router

**A 2026 snapshot of how to think about routing between frontier AI assistants.**

The model still matters. But at the frontier, the practical day-to-day difference between top proprietary assistants is increasingly shaped by the surfaces around the model: data access, hardware, integrations, modality, and the product they ship inside.

This repo is a thesis with a small prototype attached. The thesis is the point. The router is the artifact.

---

## Status

v0.1 thesis prototype. The routing logic and capability matrix are the point. Stub mode is the reliable demo path. Real provider integrations are experimental and will drift as APIs, model names, and product surfaces change.

This is intentionally a snapshot in time. The exact model names, rankings, and provider surfaces will move. The point is not that this exact matrix stays true forever. The point is that model selection is becoming a routing problem across capability moats, and writing it down clarifies the thinking.

---

## Core thesis

Anyone who uses Claude, GPT, Gemini, and Grok side by side for a week notices the same thing: on any given query, the raw-intelligence gap between them is often smaller than the gap created by everything around them.

Gemini has years of your Google data. Grok has native X timeline access and a hardware path through Tesla. Claude has the deepest code and reasoning surface and the MCP ecosystem. GPT has the broadest product surface, image generation, and the largest third-party tooling layer.

Power users already route between these manually:

- You ask Grok what is happening on X.
- You ask Gemini to summarize a 400-page PDF.
- You ask Claude to refactor a codebase.
- You ask GPT to generate an image or run a long browsing task.

The routing logic lives in the user's head. `frontier-router` makes that implicit logic explicit.

**The model still matters. But the moat is increasingly around the model.**

**The router is less important than the matrix. The matrix is less important than the thesis.**

---

## Why this matters

Two trends are running at the same time:

1. Top-end raw capability is compressing. The leaderboards are noisy and the gaps narrow every few months.
2. Proprietary surfaces are diverging. Each frontier provider is building a moat that other providers cannot easily replicate: data, hardware, integrations, modality, ambient context.

That changes the practical question. For many real workflows, "which model is smartest" is less useful than "which model has the right surface for this task." That is a routing problem, not a benchmark problem.

This repo is my attempt to name the surfaces, write down a routing matrix I actually use, and ship a small artifact that demonstrates the shape.

---

## Capability matrix

Each cell is an opinionated 2026 judgment based on sustained daily use, not benchmarks.

| Capability                  | Primary                 | Secondary  | Reasoning                                                              |
|-----------------------------|-------------------------|------------|------------------------------------------------------------------------|
| Code generation / refactor  | Claude                  | GPT        | Sustained reasoning over large diffs, clean tool use, conventions.     |
| Long-context synthesis      | Gemini                  | Claude     | 1M-2M context holds up under load; caching architecture is ahead.      |
| Real-time / X timeline      | Grok                    | GPT (web)  | Native X integration is non-replicable. Nobody else has the firehose.  |
| In-vehicle / Tesla context  | Grok                    | (none)     | Hardware and data moat. The car *is* the context.                      |
| Image generation            | GPT / Gemini            | (parity)   | `gpt-image-1` and Gemini 2.5 Flash Image ("nano-banana") sit at rough parity; pick by ecosystem. |
| Agentic browser / computer  | GPT (Operator) / Claude | (split)    | Operator for general web; Claude for structured tool-chain work.       |
| Personalized Google context | Gemini                  | (none)     | Longitudinal Gmail, Drive, YouTube, Search, Maps data.                 |
| Structured reasoning        | Claude                  | GPT        | Most reliable on multi-step logic and adhering to schemas.             |
| Creative long-form writing  | Claude                  | GPT        | Voice control and stylistic range.                                     |
| Massive doc analysis        | Gemini                  | Claude     | Context window plus session stability on 500-page PDFs.                |

Full reasoning, per-cell, in [`docs/capability-matrix.md`](docs/capability-matrix.md). Expect drift. That is the point.

---

## Hardware as a model surface

The next differentiation layer is not just apps and APIs. Hardware is becoming part of the model surface.

Frontier assistants are starting to extend into:

- Smart glasses
- Cars
- Phones with deep OS integration
- Wearables and earbuds
- Home and ambient devices
- AR interfaces
- Anything that continuously captures context

This matters because the model with the best hardware surface may know things other models cannot know: what you are looking at, where you are, what you are doing, what you are driving, what you hear, what your calendar and route and environment imply, and what action the device can take next.

Tesla and Grok is the cleanest current example, but the broader pattern is not Tesla-specific. The next moat may not be a smarter chat box. It may be a model attached to the device that sees, hears, drives, navigates, records, reminds, and acts.

A future version of this matrix will need a hardware axis: which provider has the right embodiment for this task, not just the right weights.

---

## Scope note

This matrix focuses on GPT, Claude, Gemini, and Grok because those are the proprietary frontier assistants I use side by side. It is not a complete global model benchmark.

It does not attempt to rank:

- Open-weight models (Llama, Mistral, Qwen, DeepSeek, and so on)
- Chinese frontier labs
- Local and self-hosted models

Those matter for cost, privacy, fine-tuning, sovereignty, and local deployment. They are outside the scope of this v0.1 thesis. Different problem, different repo.

---

## What would falsify this thesis?

The "moat is around the model" thesis weakens if either of the following happens:

1. One provider opens a durable raw-intelligence gap so large that users accept worse tools, worse context, worse interfaces, and worse integrations just to access it. In that world, the model still wins on its own merits and the surfaces are noise.
2. Every major provider converges on the same context access, the same tool surfaces, the same hardware integrations, and the same interoperability protocols. In that world, the surfaces stop being differentiators and capability becomes commodity again.

For now, I see the opposite of both: top-end capability is compressing while proprietary surfaces are diverging. I expect that to hold for the next 12 to 24 months. After that, who knows. The AI race could still become winner-takes-all.

---

## How the toy router works

Three-stage decision:

1. **Rules-based classifier.** Fast, deterministic. Keyword and structure heuristics map the task to a capability class. Handles most real queries.
2. **LLM-as-router fallback.** For ambiguous tasks, a cheap model classifies the query into the capability taxonomy. Off by default.
3. **User override.** Explicit provider selection always wins. `query(task, provider="grok")` bypasses routing entirely.

Architecture in [`docs/architecture.md`](docs/architecture.md). The implementation is intentionally small.

---

## Quickstart

```bash
git clone https://github.com/Moneer-S/frontier-router.git
cd frontier-router
pip install -e .
```

Stub mode (no keys, fake providers, good for exploring the routing logic):

```python
from frontier_router import Router

router = Router(mode="stub")
result = router.query("Summarize this 300-page PDF", context={"file": "report.pdf"})
print(result.provider)   # "google" (Gemini)
print(result.capability) # Capability.MASSIVE_DOC_ANALYSIS
print(result.response)   # canned stub response
```

CLI:

```bash
frontier-router "Refactor this function" --context file=auth.py
frontier-router "What is on X about xAI today?" --provider grok
frontier-router "Summarize this long document" --explain
# routed to google | capability: massive_doc_analysis | confidence: 0.80
```

OpenAI-compatible HTTP shim (so any OpenAI client can route through this):

```bash
frontier-router serve --port 8080
# POST /v1/chat/completions   (OpenAI-compatible)
# POST /route                 (native: returns routing decision + response)
# GET  /capabilities          (capability matrix)
```

Real mode exists and works for the chat-style capabilities, but it is experimental and depends on each provider's SDK staying stable. See `frontier_router.real_providers` for capability-specific endpoints (image generation, X Live Search, file upload, PDF document blocks). Treat real mode as a sketch, not a production runtime.

---

## Prior art

Honest positioning:

- **[OpenRouter](https://openrouter.ai/)** routes on the inference side: price, latency, availability, across ~100 models. It optimizes the substrate. This repo is about the layer above: which model surface to reach for in the first place.
- **[LiteLLM](https://github.com/BerriAI/litellm)** is a unified client across providers. It solves abstraction. Routing decisions are still left to the caller.
- **[LangChain](https://langchain.com/) / [LlamaIndex](https://www.llamaindex.ai/)** are orchestration and app frameworks. Model routing is a sub-feature, not the thesis.
- **[MCP](https://modelcontextprotocol.io/)** is Anthropic's connector protocol. Complementary: MCP defines how models reach tools, this repo is about which model to reach for first.

`frontier-router`'s specific contribution: capability-aware routing framed around proprietary-surface moats. One layer above LiteLLM, one layer below LangChain.

---

## Roadmap and open questions

Detailed version in [`docs/roadmap.md`](docs/roadmap.md). The interesting questions are not features:

- What does proactive routing look like when the router can observe state changes (calendar, inbox, telemetry) rather than wait for queries?
- When provider SDKs converge on a common protocol, does the router become a protocol adapter or stay a decision layer?
- Does the "which AI" question disappear behind the router the way DNS disappeared behind browsers?
- Where does the hardware-surface axis go in the matrix once glasses and ambient devices are mainstream?

---

## License

MIT. See [`LICENSE`](LICENSE).

---

*v0.1.0 (2026-04). A snapshot. Expect drift.*
