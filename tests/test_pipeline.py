"""Behavior-focused tests for the local NLPulse MVP pipeline."""

import pytest

from dashboard.app import validate_text
from src.pipeline import analyze_text


def test_non_empty_analysis_returns_expected_sections() -> None:
    result = analyze_text("Please help: the login error is blocking my account.")

    assert result["text"]
    assert {"classification", "sentiment", "urgency", "entities", "summary", "keywords"} <= result.keys()
    assert result["classification"]["label"] == "request"
    assert isinstance(result["keywords"], list)


@pytest.mark.parametrize("value", ["", "   ", "\n\t"])
def test_empty_input_is_rejected_before_analysis(value: str) -> None:
    with pytest.raises(ValueError, match="Enter some text"):
        validate_text(value)


def test_classification_and_urgency_follow_support_rules() -> None:
    result = analyze_text("Critical outage! Please fix it immediately!!")

    assert result["classification"]["label"] == "request"
    assert result["urgency"]["level"] == "critical"
    assert {"critical", "outage", "immediately"} <= set(result["urgency"]["matched_terms"])


def test_entity_extraction_finds_common_contact_entities() -> None:
    result = analyze_text("Contact support@example.com on 2026-10-16 or visit https://example.com.")

    entities = {(entity["type"], entity["text"]) for entity in result["entities"]}
    assert ("email", "support@example.com") in entities
    assert ("date", "2026-10-16") in entities
    assert ("url", "https://example.com") in entities
