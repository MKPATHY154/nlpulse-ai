"""Transparent rule-based intent classification."""

from __future__ import annotations

from dataclasses import dataclass

from .preprocessing import tokenize


@dataclass(frozen=True)
class Classification:
    """Intent label and confidence produced by the rule classifier."""

    label: str
    confidence: float
    matched_terms: list[str]


_RULES: tuple[tuple[str, frozenset[str]], ...] = (
    ("complaint", frozenset({"broken", "bug", "error", "wrong", "unhappy", "complaint", "terrible"})),
    ("request", frozenset({"please", "need", "want", "could", "would", "request", "help"})),
    ("question", frozenset({"what", "why", "when", "where", "who", "how", "question"})),
    ("feedback", frozenset({"feedback", "suggestion", "recommend", "improve", "love", "like"})),
)


def classify(text: str) -> Classification:
    """Classify text into a small stable set of support-oriented intents."""
    words = set(tokenize(text))
    scored = [(label, sorted(words & terms)) for label, terms in _RULES]
    label, matched = max(scored, key=lambda item: len(item[1]), default=("other", []))
    if not matched:
        label = "question" if text.rstrip().endswith("?") else "other"
    confidence = min(0.99, 0.55 + 0.12 * len(matched)) if matched else 0.35
    return Classification(label, round(confidence, 2), matched)
