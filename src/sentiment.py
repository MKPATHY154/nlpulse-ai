"""Small lexicon-based sentiment scoring with no model downloads."""

from __future__ import annotations

from dataclasses import dataclass

from .preprocessing import tokenize


@dataclass(frozen=True)
class Sentiment:
    """Sentiment label, normalized score, and lexicon evidence."""

    label: str
    score: float
    positive_terms: list[str]
    negative_terms: list[str]


POSITIVE = frozenset("amazing awesome good great happy helpful love excellent solved thanks thank satisfied".split())
NEGATIVE = frozenset("bad broken terrible awful angry hate unhappy poor frustrated issue problem fail failed".split())


def analyze_sentiment(text: str) -> Sentiment:
    """Return a deterministic sentiment estimate in the range [-1, 1]."""
    words = tokenize(text)
    positive = sorted({word for word in words if word in POSITIVE})
    negative = sorted({word for word in words if word in NEGATIVE})
    raw = len(positive) - len(negative)
    score = max(-1.0, min(1.0, raw / max(1, len(positive) + len(negative))))
    label = "positive" if score > 0.15 else "negative" if score < -0.15 else "neutral"
    return Sentiment(label, round(score, 2), positive, negative)
