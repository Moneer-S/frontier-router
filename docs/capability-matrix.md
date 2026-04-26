# Capability Matrix

The opinionated core of `frontier-router`. Every routing decision flows from the judgments below.

All judgments come from sustained multi-provider use, not benchmark citations. Benchmarks drift faster than they can be updated. Lived use drifts slower and is more predictive of what I actually feel running these tools side by side day to day.

Versioned. Expect drift. Changelog at the bottom.

---

## Scope note

This matrix focuses on **GPT, Claude, Gemini, and Grok** because those are the proprietary frontier assistants I use side by side. It does not attempt to rank open-weight, Chinese, or local models. Those matter for cost, privacy, fine-tuning, sovereignty, and local deployment, but they are outside the scope of this v0.1 thesis. Different problem, different repo.

---

## Scoring rubric

For each capability class, providers are scored on:

- **Quality.** Does the output actually solve the task?
- **Moat.** Can another provider meaningfully replicate this capability, or is there a structural barrier (data, hardware, integration)?
- **Reliability.** Does it hold up under load (long sessions, large inputs, concurrent use)?
- **Latency.** Is the response time acceptable for this use case?

Primary = best in class. Secondary = acceptable fallback. A dash (-) means no viable alternative.

---

## Capability classes

### `code_generation`

| Provider | Quality | Moat                   | Notes                                                                 |
|----------|---------|------------------------|-----------------------------------------------------------------------|
| Claude   | Primary | Strongest RL-on-code loop | Sustained reasoning over large diffs. Cleanest tool use. Best at following existing code conventions. |
| GPT      | Secondary | Broad training corpus | Comparable on isolated snippets. Degrades sooner on multi-file edits. |
| Gemini   | -       | Long context helps      | Strong on repo-scale reads. Weaker on write-quality vs Claude/GPT.    |
| Grok     | -       | -                      | Not the differentiator here.                                          |

**Routing heuristic:** any task mentioning code, functions, refactoring, debugging, or containing code blocks routes to Claude. Falls through to GPT if Claude is unavailable.

---

### `long_context_synthesis`

| Provider | Quality | Moat                                   | Notes                                                                 |
|----------|---------|----------------------------------------|-----------------------------------------------------------------------|
| Gemini   | Primary | 1M-2M context with caching architecture | Older turns cache cleanly, session state holds up. Empirically the only provider that doesn't visibly degrade past ~200K tokens. |
| Claude   | Secondary | 200K native, Projects for persistence  | Strong quality on the context it holds. Window is the ceiling.        |
| GPT      | -       | 128K+ but degrades under load           | Latency spikes past ~100K. UI lag compounds the problem.              |
| Grok     | -       | -                                      | Not the differentiator here.                                          |

**Routing heuristic:** inputs > 50K tokens, or phrases like "long document", "summarize this PDF", "across the whole codebase" route to Gemini.

---

### `realtime_x_timeline`

| Provider | Quality | Moat                        | Notes                                                                 |
|----------|---------|-----------------------------|-----------------------------------------------------------------------|
| Grok     | Primary | Native X firehose access    | Non-replicable. Nobody else has this data ingestion pipeline.         |
| GPT      | Secondary | Web search fallback          | Can retrieve X posts via browse, but not with Grok's freshness or depth. |
| Claude   | -       | Web search fallback          | Same as GPT but less aggressive at surfacing social data.             |
| Gemini   | -       | Google Search integration    | Strong for web, weak for X specifically.                              |

**Routing heuristic:** any query mentioning X, Twitter, tweets, trending, "what's happening", real-time social sentiment routes to Grok.

---

### `in_vehicle_context`

| Provider | Quality | Moat                        | Notes                                                                 |
|----------|---------|-----------------------------|-----------------------------------------------------------------------|
| Grok     | Primary | Tesla integration            | Vehicle state, navigation context, and FSD handoff loop are hardware-level. |
| Others   | -       | -                           | No path to parity without a hardware partnership.                     |

**Routing heuristic:** primarily a deployment-context decision, not a query-content decision. If the router is running in-vehicle, Grok is the default for contextual queries. This is the cleanest example today of the hardware-as-model-surface thesis.

---

### `image_generation`

| Provider | Quality | Moat                        | Notes                                                                 |
|----------|---------|-----------------------------|-----------------------------------------------------------------------|
| GPT      | Primary | `gpt-image-1` + Sora lineage | Strong on text rendering inside images and tight instruction adherence. |
| Gemini   | Primary | Gemini 2.5 Flash Image ("nano-banana") + Imagen | At rough parity with `gpt-image-1`. Stronger at iterative edits and conversational refinement of an existing image. |
| Claude   | -       | No native generation         | Can describe, cannot generate.                                        |
| Grok     | -       | Aurora / Flux variants       | Viable, not differentiated.                                           |

**Routing heuristic:** image-generation phrases route to GPT first, fall through to Gemini. A Gemini-first config is equally defensible, especially for users who want conversational editing of an existing image.

---

### `agentic_browser`

| Provider | Quality | Moat                        | Notes                                                                 |
|----------|---------|-----------------------------|-----------------------------------------------------------------------|
| GPT      | Primary (general web) | Operator | Furthest along on general-web agentic tasks. Handles logged-in flows and e-commerce better than peers. |
| Claude   | Primary (structured) | Claude in Chrome + MCP | Better at structured agentic work via tool chains. |
| Gemini   | -       | Project Mariner-class work   | Progressing but not yet differentiated in production.                 |
| Grok     | -       | -                           | Not a current focus area.                                             |

**Routing heuristic:** split. "Book a flight", "fill out a form", "navigate this site" goes to GPT. "Use these MCP tools to accomplish X" goes to Claude.

Note: in v0.1 these routing targets only deliver the chat path through an API key. The actual agentic surfaces (Operator, Claude in Chrome) are UI-resident and not callable from this router alone.

---

### `personalized_google_context`

| Provider | Quality | Moat                        | Notes                                                                 |
|----------|---------|-----------------------------|-----------------------------------------------------------------------|
| Gemini   | Primary | Longitudinal Google data     | Gmail (potentially years of it), Drive, YouTube, Search, Maps, Android signal. A time-machine dataset no other provider can replicate. |
| Others   | -       | -                           | MCP connectors let them reach into Gmail/Drive, but without the historical depth or implicit profile. |

**Routing heuristic:** "my email", "my calendar", "my documents", "based on my history" routes to Gemini when the user has opted into Google personalization.

Note: this moat is mostly UI-resident. The Gemini app pulls Workspace context. The bare Gemini API does not, unless the caller wires connectors themselves. The router routes intent correctly. Realizing the moat needs the right surface.

---

### `structured_reasoning`

| Provider | Quality | Moat                        | Notes                                                                 |
|----------|---------|-----------------------------|-----------------------------------------------------------------------|
| Claude   | Primary | Best schema adherence        | Most reliable on multi-step logic, JSON mode, tool-use schemas.       |
| GPT      | Secondary | Strong reasoning across the lineup | Closes the gap on pure reasoning. Schema adherence slightly weaker than Claude. |
| Gemini   | -       | Strong raw capability        | Less predictable on strict output formats.                            |
| Grok     | -       | -                           | Not the differentiator.                                               |

**Routing heuristic:** tasks involving JSON output, structured data transformation, decision trees, multi-step plans route to Claude.

---

### `creative_longform_writing`

| Provider | Quality | Moat                        | Notes                                                                 |
|----------|---------|-----------------------------|-----------------------------------------------------------------------|
| Claude   | Primary | Voice control, stylistic range | Best at sustained tone, fiction, essayistic writing. |
| GPT      | Secondary | Broad stylistic capability   | Strong but less distinctive voice by default.                         |
| Gemini   | -       | Competitive                  | Less differentiated on this axis.                                     |
| Grok     | -       | Distinctive irreverent voice | Useful in narrow cases where that voice is wanted.                    |

**Routing heuristic:** fiction, essays, stylized writing tasks route to Claude.

---

### `massive_doc_analysis`

| Provider | Quality | Moat                        | Notes                                                                 |
|----------|---------|-----------------------------|-----------------------------------------------------------------------|
| Gemini   | Primary | Context window + session stability | Only provider where 500-page PDFs actually work in production. |
| Claude   | Secondary | Projects for persistence     | Strong on the 200K it can hold.                                       |
| GPT      | -       | 128K+ but degrades           | Not reliable past ~100K tokens in practice.                           |
| Grok     | -       | -                           | Not the differentiator.                                               |

**Routing heuristic:** PDF / large file inputs, phrases like "analyze this report", "summarize this book" route to Gemini.

---

## The hardware axis (not yet in the matrix)

Every cell above is task-vs-provider. The next version of this matrix needs a third axis: the device or surface the model is attached to.

A model on smart glasses, in a vehicle, on a wearable, or embedded in a phone OS has access to context other providers cannot reach: live audio, gaze, location, motion, driving state, ambient sensors, what is on the screen right now. Tesla and Grok is the cleanest live example. Glasses, phones, and ambient devices are coming.

For v0.1 this lives in prose, not in the matrix. When hardware surfaces are common enough that routing should know about them, a "surface" column gets added: chat, browser, vehicle, glasses, ambient. The capability map then becomes capability x provider x surface, not just capability x provider.

The broader claim: the next moat may not be a smarter chat box. It may be a model attached to a device that sees, hears, drives, navigates, records, reminds, and acts.

---

## Meta notes

**The matrix is wrong in places.** That is not a bug; it is the point. Sustained parallel use produces judgments that benchmarks miss, and those judgments go stale in weeks.

**The matrix drifts toward "it depends".** As raw capability compresses, the differentiators become structural: data moats, hardware integrations, proprietary surfaces. Future capability classes will evolve toward those structural axes rather than pure task-type axes.

**This is my view.** Different usage patterns will produce different matrices. That is fine. Fork it.

---

## Changelog

- **v0.1.0** (2026-04). Initial matrix. Ten capability classes across GPT, Claude, Gemini, Grok. Scope explicitly limited to those four proprietary frontier assistants.
