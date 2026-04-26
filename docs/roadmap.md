# Roadmap

Not a schedule. A map of open questions and the direction of travel.

---

## v0.1 (current)

- Core router with rules-based classification.
- Capability matrix for four proprietary frontier providers.
- Stub mode + experimental real mode.
- CLI + Python API + OpenAI-compatible HTTP shim.
- Basic test coverage in stub mode (no live API keys required).

## v0.2

**Real SDK integrations.** Current `real` mode imports each provider's SDK lazily, but the call signatures are minimal. v0.2 brings them up to parity with each provider's native client: system prompts, generation config, model selection per capability class.

**Streaming.** Common streaming interface across providers. Non-trivial because the four SDKs stream differently; the router would normalize to an async iterator of text chunks plus a final metadata event.

**Confidence scores.** Every `RouterResult` reports a classifier confidence. v0.2 adds provider-side confidence: did the provider return a response that matches the expected capability shape (e.g., was the claimed "code generation" actually a code block)?

**LLM-as-router caching.** Opt-in, with disk-backed caching so repeated ambiguous queries do not re-classify.

## v0.3

**MCP-native tool routing.** When the task involves tool use, the question is not just "which provider is best at this capability" but "which provider's MCP connector surface has the tools needed for this task." The router would need awareness of which connectors are live on which providers.

**Multi-provider fan-out.** For high-stakes queries, send to multiple providers in parallel and either return all responses for caller comparison or synthesize a consensus. Opt-in per query.

**Per-provider cost and rate-limit awareness.** Router factors remaining budget and current rate-limit state into routing decisions. Extends into a `priority={"quality"|"latency"|"cost"}` argument.

## v1.0

**Hardware-surface axis.** Once smart glasses, ambient devices, and phone-OS integrations are common enough to route differently, the matrix becomes capability x provider x surface, not just capability x provider. The router learns which device the user is on and routes accordingly.

**Learned routing.** Users provide feedback (was this the right provider?) and a learned model improves the rules classifier over time. Anonymous usage telemetry is opt-in.

**Protocol convergence adapter.** As providers converge on common protocols (OpenAI-compatible endpoints, standardized tool-use schemas, unified streaming), the router becomes less about picking SDKs and more about picking capabilities. The abstraction holds either way.

---

## Open questions (not commitments)

### Proactive routing

Today the router is reactive: `query()` comes in, routing decision happens, response returns. All agentic AI today is reactive in this sense: you prompt, it responds, session ends.

The step-change is observation without a trigger. A router that drafts the follow-up email before you remember you owed one. That notices the calendar conflict three days out. That summarizes overnight noise before you open the laptop. The technical pieces exist in isolation (agentic loops, scheduled execution, tool access, memory), but nobody has stitched them into something that is both reliable enough to trust and bounded enough not to be terrifying.

Open questions:

- Where does the observation loop live? In the router, or in a layer above it?
- How do you bound the action space so proactive behavior does not become hallucinated action?
- What is the UX for "I noticed X and was about to do Y, approve?"
- How does the router decide when a query is worth initiating vs. staying silent?

Honest answer: this is upstream of the router. Proactive behavior is an agent property. The router is a routing layer. But routing becomes more interesting when the agent on top is proactive, because the right model for "scheduled 6am inbox triage" is different from the right model for "real-time decision support during a live event."

### Hardware as the next moat

If the model is attached to a device that sees, hears, drives, navigates, records, reminds, and acts, the routing problem changes shape. The provider with the right embodiment for a task may be more important than the provider with the smartest weights for that task.

Open questions:

- When the user is on glasses, does "draft a reply" route differently than when they are on a laptop?
- Does in-vehicle context belong as a capability class, a surface modifier, or both?
- How does the router stay honest when only one provider has the hardware surface that matters for a use case?

### The "which AI" question disappearing

The thesis of this repo is that raw capability compresses while proprietary surfaces diverge. Taken to its endpoint, the user stops thinking about which model is answering because the router makes the decision transparently.

That is basically what happened to DNS. Nobody thinks about which DNS resolver answered their query. The abstraction won.

Whether the "which AI" abstraction wins depends on:

- Whether any single provider ships an ecosystem compelling enough that users accept being locked in (mass-market path).
- Whether orchestration layers like this one ship interfaces good enough to be invisible (power-user to mainstream path).
- Whether regulation forces interoperability (forced-convergence path).

I do not know which one wins. I suspect it is a mix: the mass market locks into whichever ecosystem owns the default phone/car slot, while a smaller but more influential power-user segment keeps orchestrating. That is the outcome for keyboards and browsers today: most users accept defaults, a minority customize heavily, and the customized minority is disproportionately represented in the people building what everyone else uses.

### When do capability judgments stop being one-person views?

The capability matrix is my opinionated view. It will be wrong in places, because I do not use every provider enough in every lane to have calibrated judgment across the whole matrix.

Possible v1.0 direction: aggregate capability judgments from contributors. That has its own problem: opinion aggregation on the internet tends toward either loud-voice dominance or lowest-common-denominator averaging.

Possible answer: structured A/B. For each capability class, run periodic blind evaluations where contributors receive outputs from multiple providers and vote without knowing which is which. Matrix updates reflect the blind-eval results.

That is a lot of infrastructure for a routing library. Probably out of scope for v0.1. Probably the right direction if this becomes something people actually use.

---

## What this project is not

- Not trying to be the best Python SDK for any single provider. Use the provider's SDK directly if that is what you want.
- Not trying to be a general agent framework. LangGraph, AutoGen, and friends handle that.
- Not trying to solve inference-side routing. OpenRouter is excellent at that.
- Not trying to rank open-weight, Chinese, or local models. Different problem, different repo.
- Not trying to be opinion-free. The capability matrix is explicitly opinionated. Fork and disagree.

The value is the specific layer: capability-aware routing across proprietary-surface moats. Everything downstream of that is scope creep.
