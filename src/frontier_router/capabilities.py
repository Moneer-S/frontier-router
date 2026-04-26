"""Capability taxonomy and provider priority mapping.

This module encodes the judgments from docs/capability-matrix.md as Python
data structures. Update both when changing a routing decision.
"""

from enum import Enum


class Capability(str, Enum):
    """Capability classes the router can identify and route to."""

    CODE_GENERATION = "code_generation"
    LONG_CONTEXT_SYNTHESIS = "long_context_synthesis"
    REALTIME_X_TIMELINE = "realtime_x_timeline"
    IN_VEHICLE_CONTEXT = "in_vehicle_context"
    IMAGE_GENERATION = "image_generation"
    AGENTIC_BROWSER_GENERAL = "agentic_browser_general"
    AGENTIC_BROWSER_STRUCTURED = "agentic_browser_structured"
    PERSONALIZED_GOOGLE_CONTEXT = "personalized_google_context"
    STRUCTURED_REASONING = "structured_reasoning"
    CREATIVE_LONGFORM_WRITING = "creative_longform_writing"
    MASSIVE_DOC_ANALYSIS = "massive_doc_analysis"
    GENERAL = "general"  # default when no specific capability is detected


# Provider identifiers used throughout the codebase.
ANTHROPIC = "anthropic"
OPENAI = "openai"
GOOGLE = "google"
XAI = "xai"

ALL_PROVIDERS = [ANTHROPIC, OPENAI, GOOGLE, XAI]


# Primary → secondary provider ordering per capability.
# First entry is the primary. Fallback order is left-to-right.
# Reasoning for each ordering lives in docs/capability-matrix.md.
CAPABILITY_MAP: dict[Capability, list[str]] = {
    Capability.CODE_GENERATION:            [ANTHROPIC, OPENAI, GOOGLE],
    Capability.LONG_CONTEXT_SYNTHESIS:     [GOOGLE, ANTHROPIC, OPENAI],
    Capability.REALTIME_X_TIMELINE:        [XAI, OPENAI, GOOGLE],
    Capability.IN_VEHICLE_CONTEXT:         [XAI],
    Capability.IMAGE_GENERATION:           [OPENAI, GOOGLE, XAI],
    Capability.AGENTIC_BROWSER_GENERAL:    [OPENAI, ANTHROPIC],
    Capability.AGENTIC_BROWSER_STRUCTURED: [ANTHROPIC, OPENAI],
    Capability.PERSONALIZED_GOOGLE_CONTEXT:[GOOGLE],
    Capability.STRUCTURED_REASONING:       [ANTHROPIC, OPENAI, GOOGLE],
    Capability.CREATIVE_LONGFORM_WRITING:  [ANTHROPIC, OPENAI, GOOGLE],
    Capability.MASSIVE_DOC_ANALYSIS:       [GOOGLE, ANTHROPIC],
    Capability.GENERAL:                    [ANTHROPIC, OPENAI, GOOGLE, XAI],
}


# Short human-readable descriptions, used in --explain output and logs.
CAPABILITY_DESCRIPTIONS: dict[Capability, str] = {
    Capability.CODE_GENERATION:            "code generation, refactoring, or debugging",
    Capability.LONG_CONTEXT_SYNTHESIS:     "synthesis across long context (>50K tokens)",
    Capability.REALTIME_X_TIMELINE:        "real-time X / social timeline queries",
    Capability.IN_VEHICLE_CONTEXT:         "in-vehicle / Tesla context queries",
    Capability.IMAGE_GENERATION:           "image generation",
    Capability.AGENTIC_BROWSER_GENERAL:    "general web agentic tasks (browse, fill forms)",
    Capability.AGENTIC_BROWSER_STRUCTURED: "structured agentic tasks via tool chains",
    Capability.PERSONALIZED_GOOGLE_CONTEXT:"queries requiring personal Google data",
    Capability.STRUCTURED_REASONING:       "structured reasoning with strict output format",
    Capability.CREATIVE_LONGFORM_WRITING:  "creative long-form writing",
    Capability.MASSIVE_DOC_ANALYSIS:       "analysis of large documents (PDFs, books)",
    Capability.GENERAL:                    "general query with no specific capability signal",
}
