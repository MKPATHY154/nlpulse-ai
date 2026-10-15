"""Extractive summarization suitable for short support messages."""

from __future__ import annotations

from .preprocessing import sentences, tokenize


def summarize(text: str, max_sentences: int = 2) -> str:
    """Select the most informative sentences without requiring an NLP model."""
    parts = sentences(text)
    if len(parts) <= max_sentences:
        return " ".join(parts)
    frequencies: dict[str, int] = {}
    for word in tokenize(text):
        frequencies[word] = frequencies.get(word, 0) + 1
    ranked = sorted(enumerate(parts), key=lambda item: (sum(frequencies.get(word, 0) for word in tokenize(item[1])), -item[0]), reverse=True)
    selected = {index for index, _ in ranked[:max_sentences]}
    return " ".join(sentence for index, sentence in enumerate(parts) if index in selected)
