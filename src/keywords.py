"""Frequency-based keyword extraction."""

from __future__ import annotations

from collections import Counter

from .preprocessing import tokenize


STOPWORDS = frozenset("a an and are as at be by for from has have i in is it of on or our that the this to was we with you your".split())


def extract_keywords(text: str, limit: int = 10) -> list[str]:
    """Return up to ``limit`` meaningful keywords ordered by frequency."""
    counts = Counter(word for word in tokenize(text) if word not in STOPWORDS and len(word) > 2)
    return [word for word, _ in counts.most_common(limit)]
