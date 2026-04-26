"""Rules-based capability classifier.

Ordered list of (predicate, capability, confidence) tuples. First match wins.
Each rule picks its own confidence based on how specific the match is; any
confidence >= 0.7 tells the Router to commit without falling through to the
optional LLM classifier.

Add rules by appending to RULES. Put more specific / higher-signal rules
before more generic ones, the order encodes priority when signals overlap.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from frontier_router.capabilities import Capability

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CODE_EXTENSIONS: tuple[str, ...] = (
    ".py", ".pyi", ".ipynb",
    ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
    ".go", ".rs", ".java", ".kt", ".scala", ".clj",
    ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".hh",
    ".rb", ".php", ".swift", ".m", ".mm",
    ".cs", ".fs", ".vb",
    ".sh", ".bash", ".zsh", ".fish", ".ps1",
    ".sql", ".html", ".css", ".scss", ".less",
    ".lua", ".r", ".pl", ".ex", ".exs", ".erl",
    ".toml", ".yaml", ".yml",
)

DOC_EXTENSIONS: tuple[str, ...] = (".pdf", ".docx", ".epub", ".txt")

# Roughly 50K tokens, the point where Gemini's long-context advantage begins
# to dominate Claude/GPT in practice (see docs/capability-matrix.md).
MASSIVE_DOC_SIZE_BYTES = 200_000

# Task length thresholds, in raw characters.
MASSIVE_TASK_CHARS = 50_000
LONG_TASK_CHARS = 20_000


# ---------------------------------------------------------------------------
# Pattern tables
# ---------------------------------------------------------------------------


def _compile(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


_X_PATTERNS = _compile([
    r"\b(X|Twitter)\s+(trending|timeline|posts|tweets)\b",
    r"what(?:'s| is)\s+(happening|trending)\s+on\s+(X|Twitter)\b",
    r"\blatest\s+(posts|tweets)\s+about\b",
    r"\breal[- ]?time\s+sentiment\b",
])

_VEHICLE_PATTERNS = _compile([
    r"\b(my|the)\s+(car|Tesla|vehicle)\b",
    r"\bnavigate\s+(me\s+)?to\b",
    r"\bwhile\s+driving\b",
    r"\bFSD\b",
    r"\bautopilot\b",
])

_GOOGLE_PATTERNS = _compile([
    r"\bmy\s+(gmail|inbox|calendar|drive|documents|youtube\s+history|search\s+history)\b",
    r"\bbased\s+on\s+my\s+(history|data|past)\b",
])

_MCP_PATTERNS = _compile([
    r"\bMCP\b",
    r"\btool\s+chain\b",
    r"\buse\s+(these|the)\s+tools?\s+to\b",
])

_IMAGE_PATTERNS = _compile([
    r"\b(draw|generate|create|make|render)\s+(an?\s+)?"
    r"(image|picture|illustration|logo|diagram|painting)\b",
    r"\billustrate\b",
    r"\brender\s+an\s+image\b",
])

_AGENTIC_GENERAL_PATTERNS = _compile([
    r"\b(book|order|purchase|buy)\s+(a\s+|an\s+)?.+\s+(online|on\s+a\s+website)\b",
    r"\bfill\s+out\s+(the\s+|this\s+)?form\b",
    r"\bnavigate\s+to\s+and\s+do\b",
])

_CODE_KW_PATTERNS = _compile([
    r"\brefactor\b",
    r"\bdebug\b",
    r"\bfunction\b",
    r"\bbug\b",
    r"\bcompile\b",
    r"\btraceback\b",
    r"\bstack\s+trace\b",
    r"\bunit\s+test\b",
    r"\bwrite\s+a\s+(class|function|method)\b",
])

_MASSIVE_DOC_TEXT_PATTERNS = _compile([
    r"\bsummarize\s+this\s+(document|pdf|report|book)\b",
    r"\banalyze\s+this\s+(doc|report)\b",
])

_LONG_CTX_PATTERNS = _compile([
    r"\bacross\s+the\s+(whole|entire)\s+(codebase|repo|document)\b",
    r"\bsynthesize\b",
    r"\blong\s+context\b",
])

_STRUCTURED_PATTERNS = _compile([
    r"\breturn\s+(as\s+)?JSON\b",
    r"\boutput\s+format\s*:",
    r"\bschema\s*:",
    r"\bstrict\s+format\b",
    r"\bmulti[- ]?step\s+(plan|logic)\b",
])

_CREATIVE_PATTERNS = _compile([
    # "write" + up to a few adjectives + a creative-noun. Covers "write a short
    # story", "write an epic poem", "write the next chapter".
    r"\bwrite\s+(?:a|an|the|some|my)?\s*(?:\w+\s+){0,3}?"
    r"(story|essay|poem|novel|chapter|article|blog\s+post)\b",
    r"\bin\s+the\s+style\s+of\b",
    r"\bcreative\s+writing\b",
])

# ---------------------------------------------------------------------------
# Predicate helpers
# ---------------------------------------------------------------------------


def _any(task: str, patterns: list[re.Pattern]) -> bool:
    return any(p.search(task) for p in patterns)


def _has_code_block(task: str) -> bool:
    return "```" in task


def _has_code_file(context: dict) -> bool:
    path = context.get("file")
    if not path:
        return False
    return str(path).lower().endswith(CODE_EXTENSIONS)


def _has_large_doc(context: dict) -> bool:
    path = context.get("file")
    if not path:
        return False
    p = str(path).lower()
    if not p.endswith(DOC_EXTENSIONS):
        return False
    size = context.get("size")
    if isinstance(size, int) and size > MASSIVE_DOC_SIZE_BYTES:
        return True
    # PDFs and EPUBs, sized or not, are nearly always "massive" in routing terms.
    return p.endswith((".pdf", ".epub"))


def _task_very_long(task: str) -> bool:
    return len(task) > MASSIVE_TASK_CHARS


def _task_long(task: str) -> bool:
    return len(task) > LONG_TASK_CHARS


def _looks_like_schema(task: str) -> bool:
    """True if the task contains a {k: v ...} block suggesting JSON/schema output."""
    if "{" not in task or "}" not in task:
        return False
    return re.search(r"\{[^{}]*:[^{}]*\}", task) is not None


# ---------------------------------------------------------------------------
# Rules table, ordered. First match wins.
# ---------------------------------------------------------------------------

Rule = tuple[Callable[[str, dict], bool], Capability, float]

RULES: list[Rule] = [
    # Domain-specific signals, these trump "has code" and similar generic cues.
    (lambda t, c: _any(t, _VEHICLE_PATTERNS),
        Capability.IN_VEHICLE_CONTEXT, 0.85),
    (lambda t, c: _any(t, _X_PATTERNS),
        Capability.REALTIME_X_TIMELINE, 0.9),
    (lambda t, c: _any(t, _GOOGLE_PATTERNS),
        Capability.PERSONALIZED_GOOGLE_CONTEXT, 0.85),
    (lambda t, c: _any(t, _MCP_PATTERNS),
        Capability.AGENTIC_BROWSER_STRUCTURED, 0.85),

    # Structural signals: attached files and code blocks.
    (lambda t, c: _has_code_block(t),
        Capability.CODE_GENERATION, 0.95),
    (lambda t, c: _has_code_file(c),
        Capability.CODE_GENERATION, 0.9),
    (lambda t, c: _has_large_doc(c),
        Capability.MASSIVE_DOC_ANALYSIS, 0.9),

    # Content-length-driven: massive tasks route to doc-analysis.
    (lambda t, c: _task_very_long(t) or _any(t, _MASSIVE_DOC_TEXT_PATTERNS),
        Capability.MASSIVE_DOC_ANALYSIS, 0.8),

    # Task-type keywords.
    (lambda t, c: _any(t, _IMAGE_PATTERNS),
        Capability.IMAGE_GENERATION, 0.8),
    (lambda t, c: _any(t, _AGENTIC_GENERAL_PATTERNS),
        Capability.AGENTIC_BROWSER_GENERAL, 0.75),
    (lambda t, c: _any(t, _CODE_KW_PATTERNS),
        Capability.CODE_GENERATION, 0.75),
    (lambda t, c: _any(t, _CREATIVE_PATTERNS),
        Capability.CREATIVE_LONGFORM_WRITING, 0.75),
    (lambda t, c: _any(t, _LONG_CTX_PATTERNS) or _task_long(t),
        Capability.LONG_CONTEXT_SYNTHESIS, 0.75),
    (lambda t, c: _any(t, _STRUCTURED_PATTERNS) or _looks_like_schema(t),
        Capability.STRUCTURED_REASONING, 0.75),
]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def classify(task: str, context: dict) -> tuple[Capability, float]:
    """Return (capability, confidence) for the task.

    Confidence >= 0.7 means 'commit, don't fall through to LLM classifier'.
    No match returns (Capability.GENERAL, 0.3).
    """
    context = context or {}
    for predicate, capability, confidence in RULES:
        if predicate(task, context):
            return capability, confidence
    return Capability.GENERAL, 0.3
