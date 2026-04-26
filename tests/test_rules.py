"""Tests for the rules-based classifier."""

from __future__ import annotations

import pytest

from frontier_router.capabilities import Capability
from frontier_router.routing.rules import classify

# --- Capability-specific rules ----------------------------------------------


def test_code_generation_from_code_block():
    task = "What's wrong here?\n```python\ndef foo(): return 1\n```"
    cap, conf = classify(task, {})
    assert cap == Capability.CODE_GENERATION
    assert conf >= 0.9


def test_code_generation_from_file_extension():
    cap, conf = classify("what does this do", {"file": "app.py"})
    assert cap == Capability.CODE_GENERATION
    assert conf >= 0.7


def test_code_generation_from_keyword_refactor():
    cap, conf = classify("Please refactor the login flow", {})
    assert cap == Capability.CODE_GENERATION
    assert conf >= 0.7


def test_code_generation_from_write_a_function():
    cap, conf = classify("Write a function that sums a list", {})
    assert cap == Capability.CODE_GENERATION


def test_massive_doc_from_pdf_file():
    cap, conf = classify("any notes?", {"file": "annual_report.pdf"})
    assert cap == Capability.MASSIVE_DOC_ANALYSIS
    assert conf >= 0.7


def test_massive_doc_from_summarize_phrase():
    cap, _ = classify("Summarize this report for me", {})
    assert cap == Capability.MASSIVE_DOC_ANALYSIS


def test_massive_doc_from_very_long_task():
    task = "a" * 60_000
    cap, _ = classify(task, {})
    assert cap == Capability.MASSIVE_DOC_ANALYSIS


def test_long_context_from_phrase():
    cap, _ = classify("Find all bugs across the entire codebase", {})
    assert cap == Capability.LONG_CONTEXT_SYNTHESIS


def test_long_context_from_length():
    task = "x" * 25_000
    cap, _ = classify(task, {})
    assert cap == Capability.LONG_CONTEXT_SYNTHESIS


def test_realtime_x_timeline():
    cap, conf = classify("What's trending on X right now?", {})
    assert cap == Capability.REALTIME_X_TIMELINE
    assert conf >= 0.7


def test_realtime_x_latest_tweets():
    cap, _ = classify("Show me the latest tweets about the election", {})
    assert cap == Capability.REALTIME_X_TIMELINE


def test_in_vehicle_my_car():
    cap, conf = classify("My car is making a weird noise", {})
    assert cap == Capability.IN_VEHICLE_CONTEXT
    assert conf >= 0.7


def test_in_vehicle_fsd():
    cap, _ = classify("Is FSD safe on highways?", {})
    assert cap == Capability.IN_VEHICLE_CONTEXT


def test_image_generation_draw():
    cap, conf = classify("Draw an image of a sunset over mountains", {})
    assert cap == Capability.IMAGE_GENERATION
    assert conf >= 0.7


def test_image_generation_illustrate():
    cap, _ = classify("Illustrate this concept for a book cover", {})
    assert cap == Capability.IMAGE_GENERATION


def test_agentic_browser_general_book_flight():
    cap, _ = classify("Book a flight to Tokyo online for next Tuesday", {})
    assert cap == Capability.AGENTIC_BROWSER_GENERAL


def test_agentic_browser_general_fill_form():
    cap, _ = classify("Fill out this form with my info", {})
    assert cap == Capability.AGENTIC_BROWSER_GENERAL


def test_agentic_browser_structured_mcp():
    cap, _ = classify("Use these MCP tools to grep the repo", {})
    assert cap == Capability.AGENTIC_BROWSER_STRUCTURED


def test_personalized_google_context_gmail():
    cap, _ = classify("Summarize my inbox from last week", {})
    assert cap == Capability.PERSONALIZED_GOOGLE_CONTEXT


def test_personalized_google_context_history():
    cap, _ = classify("Based on my search history, what should I read next?", {})
    assert cap == Capability.PERSONALIZED_GOOGLE_CONTEXT


def test_structured_reasoning_json():
    cap, _ = classify("Return as JSON: the capital of each EU country", {})
    assert cap == Capability.STRUCTURED_REASONING


def test_structured_reasoning_schema_braces():
    cap, _ = classify("Output matching {name: string, age: int}", {})
    assert cap == Capability.STRUCTURED_REASONING


def test_creative_longform_story():
    cap, _ = classify("Write a story about a lighthouse keeper", {})
    assert cap == Capability.CREATIVE_LONGFORM_WRITING


def test_creative_longform_in_the_style_of():
    cap, _ = classify("Explain recursion in the style of Hemingway", {})
    assert cap == Capability.CREATIVE_LONGFORM_WRITING


# --- Default fallback -------------------------------------------------------


def test_general_fallback_no_match():
    cap, conf = classify("The mitochondria is the powerhouse of the cell.", {})
    assert cap == Capability.GENERAL
    assert conf == pytest.approx(0.3)


def test_general_fallback_empty():
    cap, conf = classify("", {})
    assert cap == Capability.GENERAL
    assert conf == pytest.approx(0.3)


# --- Confidence / threshold behavior ----------------------------------------


def test_confidence_threshold_commits_on_strong_match():
    # Strong domain signals (code block) should clear the 0.7 commit threshold.
    _, conf = classify("```py\nprint(1)\n```", {})
    assert conf >= 0.7


def test_fallback_below_commit_threshold():
    # Totally generic content should not clear the commit threshold.
    _, conf = classify("Tell me a fact.", {})
    assert conf < 0.7


# --- Priority / ordering ----------------------------------------------------


def test_vehicle_beats_code_keywords():
    cap, _ = classify("My Tesla bug is weird when debug mode is on", {})
    assert cap == Capability.IN_VEHICLE_CONTEXT
