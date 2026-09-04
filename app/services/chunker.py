"""Split raw text into overlapping chunks suitable for embedding.

Uses a simple recursive splitter: try to break on paragraph, then
sentence, then word boundaries, so chunks don't cut mid-word/mid-sentence
where avoidable, while keeping the implementation dependency-free.
"""
import re

SEPARATORS = ["\n\n", "\n", ". ", " "]


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    text = re.sub(r"\s+\n", "\n", text).strip()
    if not text:
        return []

    chunks = _split(text, chunk_size, SEPARATORS)

    # Apply overlap by re-joining with sliding window over the split pieces.
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = overlapped[-1][-overlap:]
        overlapped.append((prev_tail + " " + chunks[i]).strip())
    return overlapped


def _split(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    if not separators:
        # Hard split as last resort
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep, rest_seps = separators[0], separators[1:]
    pieces = text.split(sep)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = (current + sep + piece) if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(piece) > chunk_size:
                chunks.extend(_split(piece, chunk_size, rest_seps))
                current = ""
            else:
                current = piece
    if current:
        chunks.append(current)

    return [c.strip() for c in chunks if c.strip()]
