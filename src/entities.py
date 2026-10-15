"""Lightweight named-entity extraction using regular expressions."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Entity:
    """A detected entity and its coarse type."""

    text: str
    type: str


_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("url", re.compile(r"\bhttps?://[^\s]+", re.IGNORECASE)),
    ("date", re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b")),
    ("money", re.compile(r"(?:[$€£]\s?\d+(?:[.,]\d{2})?|\b\d+(?:[.,]\d{2})?\s?(?:USD|EUR|GBP)\b)", re.IGNORECASE)),
    ("phone", re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)")),
)


def extract_entities(text: str) -> list[Entity]:
    """Extract common contact, date, URL, and monetary entities."""
    found: list[tuple[int, int, Entity]] = []
    for kind, pattern in _PATTERNS:
        found.extend(
            (match.start(), match.end(), Entity(match.group(0).rstrip(".,);"), kind))
            for match in pattern.finditer(text)
        )

    # Prefer the first matching pattern (dates before phones, for example) and
    # do not return multiple entity types for the same character span.
    selected: list[tuple[int, int, Entity]] = []
    for start, end, entity in sorted(found, key=lambda item: (item[0], item[1] - item[0])):
        if any(start < existing_end and end > existing_start for existing_start, existing_end, _ in selected):
            continue
        selected.append((start, end, entity))
    return [entity for _, _, entity in sorted(selected, key=lambda item: item[0])]
