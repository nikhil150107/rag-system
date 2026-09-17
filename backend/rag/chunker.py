"""Structure-aware recursive text chunker."""
import re
from dataclasses import dataclass
from typing import List, Optional
from .document_parser import ParsedPage


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    text: str
    chunk_index: int
    page_number: Optional[int]
    document_id: str


class RecursiveChunker:
    """
    Structure-aware text chunker that splits text hierarchically:
    1. Paragraphs (\n\n)
    2. Lines (\n)
    3. Sentences (.!?)
    4. Words (spaces)
    5. Characters (fallback)

    Targets configurable token length (approx 1 token ~= 4 chars / 0.75 words) with overlap.
    """

    def __init__(
        self,
        target_tokens: int = 500,
        overlap_tokens: int = 80,
        chars_per_token: float = 4.0
    ):
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.chars_per_token = chars_per_token

        self.max_chars = int(target_tokens * chars_per_token)
        self.overlap_chars = int(overlap_tokens * chars_per_token)

        # Separator hierarchy
        self.separators = [
            r"\n\n+",                      # Paragraphs
            r"\n+",                        # Lines
            r"(?<=[.!?])\s+",              # Sentence boundaries
            r"\s+",                        # Words
            r""                            # Characters
        ]

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count based on character length and word count."""
        if not text:
            return 0
        word_count = len(text.split())
        char_count = len(text)
        return int(max(word_count * 1.3, char_count / self.chars_per_token))

    def _split_text_by_separator(self, text: str, separator_idx: int) -> List[str]:
        """Recursively split text into segments within target size."""
        if len(text) <= self.max_chars or separator_idx >= len(self.separators):
            return [text.strip()] if text.strip() else []

        sep = self.separators[separator_idx]
        if sep == "":
            # Character-level fallback
            step = max(1, self.max_chars - self.overlap_chars)
            chunks = []
            for i in range(0, len(text), step):
                chunk = text[i:i + self.max_chars].strip()
                if chunk:
                    chunks.append(chunk)
            return chunks

        # Split by regex pattern
        splits = [s for s in re.split(sep, text) if s.strip()]
        if not splits or len(splits) == 1:
            # Fallback to next finer separator
            return self._split_text_by_separator(text, separator_idx + 1)

        # Merge splits into chunks with target size and overlap
        chunks: List[str] = []
        current_pieces: List[str] = []
        current_len = 0

        for piece in splits:
            piece_len = len(piece)
            if piece_len > self.max_chars:
                if current_pieces:
                    chunks.append(" ".join(current_pieces).strip())
                    current_pieces = []
                    current_len = 0
                sub_chunks = self._split_text_by_separator(piece, separator_idx + 1)
                chunks.extend(sub_chunks)
                continue

            if current_len + piece_len + 1 > self.max_chars and current_pieces:
                chunks.append(" ".join(current_pieces).strip())

                # Overlap logic: extract trailing content up to overlap_chars
                overlap_pieces: List[str] = []
                overlap_len = 0
                for p in reversed(current_pieces):
                    if overlap_len + len(p) + 1 <= self.overlap_chars:
                        overlap_pieces.insert(0, p)
                        overlap_len += len(p) + 1
                    else:
                        break

                # If no whole split piece fits in overlap, capture trailing words
                if not overlap_pieces and current_pieces:
                    last_piece = current_pieces[-1]
                    words = last_piece.split()
                    tail_words = []
                    tail_len = 0
                    for w in reversed(words):
                        if tail_len + len(w) + 1 <= self.overlap_chars:
                            tail_words.insert(0, w)
                            tail_len += len(w) + 1
                        else:
                            break
                    if tail_words:
                        overlap_pieces = [" ".join(tail_words)]
                        overlap_len = tail_len

                current_pieces = overlap_pieces
                current_len = overlap_len

            current_pieces.append(piece)
            current_len += piece_len + 1

        if current_pieces:
            chunk_str = " ".join(current_pieces).strip()
            if chunk_str:
                chunks.append(chunk_str)

        return chunks

    def chunk_text(self, text: str) -> List[str]:
        """Split arbitrary text into structure-aware chunks."""
        cleaned = text.strip()
        if not cleaned:
            return []
        return self._split_text_by_separator(cleaned, 0)

    def chunk_pages(self, pages: List[ParsedPage], document_id: str) -> List[TextChunk]:
        """
        Chunk a list of ParsedPage objects while preserving page numbers and assigning
        sequential chunk indices.
        """
        all_chunks: List[TextChunk] = []
        chunk_counter = 0

        for page in pages:
            raw_chunks = self.chunk_text(page.text)
            for chunk_text in raw_chunks:
                if chunk_text.strip():
                    all_chunks.append(
                        TextChunk(
                            text=chunk_text.strip(),
                            chunk_index=chunk_counter,
                            page_number=page.page_number,
                            document_id=document_id
                        )
                    )
                    chunk_counter += 1

        return all_chunks
