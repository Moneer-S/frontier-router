"""Walk the capability matrix, one representative prompt per capability class.

Run in stub mode so the output is deterministic and no API keys are needed.
This file doubles as living documentation of the routing matrix.
"""

from __future__ import annotations

from frontier_router import Router
from frontier_router.capabilities import Capability

# One example query per capability. Keep them short and unambiguous so the
# rules classifier nails them without the LLM fallback.
EXAMPLES: list[tuple[Capability, str, dict]] = [
    (Capability.CODE_GENERATION,
        "Refactor this function to use async/await", {}),
    (Capability.LONG_CONTEXT_SYNTHESIS,
        "Synthesize conclusions across the entire codebase", {}),
    (Capability.REALTIME_X_TIMELINE,
        "What's trending on X right now?", {}),
    (Capability.IN_VEHICLE_CONTEXT,
        "Navigate me to the nearest Supercharger in my Tesla", {}),
    (Capability.IMAGE_GENERATION,
        "Draw an image of a lighthouse at dusk", {}),
    (Capability.AGENTIC_BROWSER_GENERAL,
        "Book a flight to Tokyo online for next Friday", {}),
    (Capability.AGENTIC_BROWSER_STRUCTURED,
        "Use these MCP tools to run the integration tests", {}),
    (Capability.PERSONALIZED_GOOGLE_CONTEXT,
        "Summarize my inbox from last week", {}),
    (Capability.STRUCTURED_REASONING,
        "Return as JSON: top 5 US cities by population", {}),
    (Capability.CREATIVE_LONGFORM_WRITING,
        "Write a short story about a lighthouse keeper", {}),
    (Capability.MASSIVE_DOC_ANALYSIS,
        "Summarize this report in detail", {"file": "annual_report.pdf"}),
]


def main() -> None:
    router = Router(mode="stub")
    header = f"{'Expected':<32} {'Routed':<32} Provider"
    print(header)
    print("-" * len(header))
    for expected, task, context in EXAMPLES:
        result = router.query(task, context=context)
        match = "OK " if result.capability == expected else "XX "
        print(
            f"{match}{expected.value:<29} "
            f"{result.capability.value:<32} "
            f"{result.provider}"
        )


if __name__ == "__main__":
    main()
