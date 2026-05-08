"""
Text chunking.

Splits a long document string into overlapping chunks suitable for embedding.
Deliberately simple: splits on sentence/paragraph boundaries where possible,
falls back to character-level splitting. No external dependencies.
"""
from __future__ import annotations

import re


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    """
    Split *text* into chunks of approximately *chunk_size* characters
    with *chunk_overlap* characters of overlap between consecutive chunks.

    Strategy:
    1. Split into paragraphs (double newline).
    2. Accumulate paragraphs until the chunk would exceed chunk_size.
    3. When a single paragraph is larger than chunk_size, split by sentence.
    4. If still too large, hard-split by character.

    Returns a list of non-empty stripped strings.
    """
    # Normalise whitespace
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Expand any paragraph that exceeds chunk_size into sentences
    units: list[str] = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            units.append(para)
        else:
            units.extend(_split_sentences(para, chunk_size))

    return _combine_units(units, chunk_size, chunk_overlap)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str, chunk_size: int) -> list[str]:
    sentences = _SENTENCE_RE.split(text)
    result: list[str] = []
    for sentence in sentences:
        if len(sentence) <= chunk_size:
            result.append(sentence)
        else:
            # Hard split as last resort
            result.extend(
                sentence[i : i + chunk_size]
                for i in range(0, len(sentence), chunk_size)
            )
    return result


def _combine_units(units: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Combine small units into chunks, respecting size and overlap."""
    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for unit in units:
        unit_len = len(unit)
        if current_len + unit_len > chunk_size and current_parts:
            chunks.append(" ".join(current_parts))
            # Retain overlap: keep trailing characters worth ~overlap chars
            overlap_parts: list[str] = []
            overlap_len = 0
            for part in reversed(current_parts):
                if overlap_len + len(part) > overlap:
                    break
                overlap_parts.insert(0, part)
                overlap_len += len(part)
            current_parts = overlap_parts
            current_len = overlap_len

        current_parts.append(unit)
        current_len += unit_len

    if current_parts:
        chunks.append(" ".join(current_parts))

    return [c for c in chunks if c.strip()]
