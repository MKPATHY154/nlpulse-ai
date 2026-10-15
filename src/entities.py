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
    ("date", re.compile(
        r"\b(?:"
        r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}|"
        r"\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|"
        r"Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
        r"Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}|"
        r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
        r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
        r"Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,)?\s+\d{4}"
        r")\b", re.IGNORECASE)),
    ("money", re.compile(
        r"(?:[$€£]\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?|"
        r"[$€£]\s?\d+(?:[.,]\d{2})?|\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s?"
        r"(?:USD|EUR|GBP)\b|\b\d+(?:[.,]\d{2})?\s?(?:USD|EUR|GBP)\b)", re.IGNORECASE)),
    ("phone", re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)")),
)

_ENTITY_PRIORITY = {"email": 0, "url": 1, "date": 2, "money": 3, "phone": 4}
_TRAILING_PUNCTUATION = ".,;:)]}"


def extract_entities(text: str) -> list[Entity]:
    """Extract common contact, date, URL, and monetary entities."""
    found: list[tuple[int, int, Entity]] = []
    for kind, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            value = match.group(0).rstrip(_TRAILING_PUNCTUATION)
            if value:
                found.append((match.start(), match.start() + len(value), Entity(value, kind)))

    # Resolve conflicts by entity specificity, then by longer span.  In
    # particular, a written date should win over the phone pattern's loose
    # digit/separator match, while URLs and emails remain intact.
    selected: list[tuple[int, int, Entity]] = []
    for start, end, entity in sorted(
        found,
        key=lambda item: (_ENTITY_PRIORITY[item[2].type], item[0], -(item[1] - item[0])),
    ):
        if any(start < existing_end and end > existing_start for existing_start, existing_end, _ in selected):
            continue
        selected.append((start, end, entity))
    return [entity for _, _, entity in sorted(selected, key=lambda item: item[0])]
