import re
from dataclasses import dataclass
from pathlib import Path

import tiktoken

_ENCODER = tiktoken.get_encoding("cl100k_base")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


@dataclass(frozen=True)
class Chunk:
    page_number: int
    chunk_index: int
    content: str
    token_count: int


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_END.split(text)
    return [p.strip() for p in parts if p.strip()]


def _token_count(text: str) -> int:
    return len(_ENCODER.encode(text))


def _tail_tokens(text: str, n: int) -> str:
    if n <= 0:
        return ""
    tokens = _ENCODER.encode(text)
    if len(tokens) <= n:
        return text
    return _ENCODER.decode(tokens[-n:])


def chunk_text(
    pages: list[tuple[int, str]],
    max_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    """Pack sentences into token-budgeted chunks, preserving page boundaries.

    Each output chunk inherits the page number of the sentence that opened it.
    A chunk closes when adding the next sentence would exceed max_tokens;
    the next chunk is seeded with the tail of the previous at overlap_tokens.
    """
    chunks: list[Chunk] = []
    chunk_index = 0

    carry_text = ""
    carry_tokens = 0
    current_page = None
    current_sentences: list[str] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal chunk_index, carry_text, carry_tokens, current_sentences, current_tokens, current_page
        if not current_sentences:
            return
        content = " ".join(current_sentences).strip()
        if not content:
            current_sentences = []
            current_tokens = 0
            return
        chunks.append(
            Chunk(
                page_number=current_page if current_page is not None else 1,
                chunk_index=chunk_index,
                content=content,
                token_count=_token_count(content),
            )
        )
        chunk_index += 1
        # Seed the next chunk with the tail of this one.
        carry_text = _tail_tokens(content, overlap_tokens)
        carry_tokens = _token_count(carry_text) if carry_text else 0
        current_sentences = []
        current_tokens = 0
        current_page = None

    for page_num, page_text in pages:
        sentences = _split_sentences(page_text)
        for sent in sentences:
            sent_tokens = _token_count(sent)
            # If a single sentence alone exceeds the budget, emit it as its own chunk.
            if sent_tokens > max_tokens:
                flush()  # flush anything in progress first
                chunks.append(
                    Chunk(
                        page_number=page_num,
                        chunk_index=chunk_index,
                        content=sent,
                        token_count=sent_tokens,
                    )
                )
                chunk_index += 1
                carry_text = _tail_tokens(sent, overlap_tokens)
                carry_tokens = _token_count(carry_text) if carry_text else 0
                continue

            # Starting a new chunk? Open with carry_text for overlap.
            if not current_sentences:
                current_page = page_num
                if carry_text:
                    current_sentences.append(carry_text)
                    current_tokens = carry_tokens
                    carry_text = ""
                    carry_tokens = 0

            # Does adding this sentence blow the budget? Flush and retry.
            if current_tokens + sent_tokens > max_tokens:
                flush()
                current_page = page_num
                if carry_text:
                    current_sentences.append(carry_text)
                    current_tokens = carry_tokens
                    carry_text = ""
                    carry_tokens = 0

            current_sentences.append(sent)
            current_tokens += sent_tokens

    flush()
    return chunks


def parse_pdf(path: str | Path) -> list[tuple[int, str]]:
    """Extract text per page from a PDF. Returns list of (page_number, text)."""
    from unstructured.partition.pdf import partition_pdf

    elements = partition_pdf(filename=str(path), strategy="fast")

    per_page: dict[int, list[str]] = {}
    for el in elements:
        page = getattr(el.metadata, "page_number", None) or 1
        text = str(el).strip()
        if text:
            per_page.setdefault(page, []).append(text)

    return [(p, " ".join(per_page[p])) for p in sorted(per_page.keys())]


def chunk_pdf(
    path: str | Path, max_tokens: int, overlap_tokens: int
) -> tuple[list[Chunk], int]:
    """Parse a PDF and chunk it. Returns (chunks, page_count)."""
    pages = parse_pdf(path)
    page_count = len(pages) if pages else 0
    return chunk_text(pages, max_tokens=max_tokens, overlap_tokens=overlap_tokens), page_count
