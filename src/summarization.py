"""Extractive summarization suitable for short support messages."""

from __future__ import annotations

from .keywords import STOPWORDS
from .preprocessing import sentences, tokenize


def summarize(text: str, max_sentences: int = 2) -> str:
    """Select the most informative sentences without requiring an NLP model."""
    parts = sentences(text)
    if max_sentences <= 0 or not parts:
        return ""
    if len(parts) <= max_sentences:
        return " ".join(parts)

    # Ignore connective words and reward terms that make a sentence distinct.
    # Ranking all sentences globally avoids repeatedly selecting a long,
    # low-information sentence merely because it contains many stopwords.
    frequencies: dict[str, int] = {}
    sentence_words: list[list[str]] = []
    for part in parts:
        words = [word for word in tokenize(part) if word not in STOPWORDS and len(word) > 2]
        sentence_words.append(words)
        for word in words:
            frequencies[word] = frequencies.get(word, 0) + 1
    ranked = sorted(
        enumerate(sentence_words),
        key=lambda item: (
            sum(frequencies[word] for word in set(item[1])) + len(set(item[1])) * 0.25,
            -item[0],
        ),
        reverse=True,
    )
    selected = {index for index, _ in ranked[:max_sentences]}
    return " ".join(sentence for index, sentence in enumerate(parts) if index in selected)
