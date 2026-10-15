"""Rule-based urgency detection."""

from __future__ import annotations

from dataclasses import dataclass

from .preprocessing import tokenize


@dataclass(frozen=True)
class Urgency:
    """Urgency level, score, and terms that contributed to it."""

    level: str
    score: int
    matched_terms: list[str]


_WEIGHTS = {"urgent": 3, "asap": 3, "immediately": 3, "emergency": 4, "critical": 4, "outage": 4, "deadline": 2, "today": 2, "now": 2}


def assess_urgency(text: str) -> Urgency:
    """Estimate urgency from explicit urgency vocabulary and punctuation."""
    words = tokenize(text)
    matched = sorted({word for word in words if word in _WEIGHTS})
    score = sum(_WEIGHTS[word] for word in matched) + (1 if text.count("!") >= 2 else 0)
    level = "critical" if score >= 5 else "high" if score >= 3 else "medium" if score >= 1 else "low"
    return Urgency(level, min(score, 10), matched)
