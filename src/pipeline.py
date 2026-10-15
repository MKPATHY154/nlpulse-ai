"""Composable NLPulse analysis pipeline."""

from __future__ import annotations

from typing import Any

from .classifier import classify
from .entities import extract_entities
from .keywords import extract_keywords
from .preprocessing import normalize_text
from .sentiment import analyze_sentiment
from .summarization import summarize
from .urgency import assess_urgency


def analyze_text(text: str) -> dict[str, Any]:
    """Run all NLPulse analyzers and return JSON-serializable results."""
    normalized = normalize_text(text)
    if not normalized:
        raise ValueError("Text must contain at least one non-whitespace character")
    classification = classify(normalized)
    sentiment = analyze_sentiment(normalized)
    urgency = assess_urgency(normalized)
    return {
        "text": normalized,
        "classification": {"label": classification.label, "confidence": classification.confidence, "matched_terms": classification.matched_terms},
        "sentiment": {"label": sentiment.label, "score": sentiment.score, "positive_terms": sentiment.positive_terms, "negative_terms": sentiment.negative_terms},
        "urgency": {"level": urgency.level, "score": urgency.score, "matched_terms": urgency.matched_terms},
        "entities": [{"text": entity.text, "type": entity.type} for entity in extract_entities(normalized)],
        "summary": summarize(normalized),
        "keywords": extract_keywords(normalized),
    }
