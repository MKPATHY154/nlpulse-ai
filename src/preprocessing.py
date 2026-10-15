"""Text normalization helpers used by the NLPulse analysis pipeline."""

from __future__ import annotations

import re


_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Return text with normalized whitespace while preserving punctuation and case."""
    return _WHITESPACE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase words, retaining apostrophes inside words."""
    return re.findall(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*", normalize_text(text).lower())


def sentences(text: str) -> list[str]:
    """Split text into non-empty sentences using common terminal punctuation."""
    cleaned = normalize_text(text)
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", cleaned) if part.strip()]
