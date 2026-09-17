"""Unit tests for structure-aware recursive chunker."""
import pytest
from rag.chunker import RecursiveChunker, TextChunk
from rag.document_parser import ParsedPage


def test_chunker_empty_text():
    chunker = RecursiveChunker(target_tokens=100, overlap_tokens=20)
    chunks = chunker.chunk_text("")
    assert chunks == []
    chunks_whitespace = chunker.chunk_text("   \n\n   ")
    assert chunks_whitespace == []


def test_chunker_normal_paragraph_text():
    chunker = RecursiveChunker(target_tokens=50, overlap_tokens=10)
    text = (
        "Paragraph 1 is about introduction to RAG systems.\n\n"
        "Paragraph 2 discusses vector databases and embeddings in detail.\n\n"
        "Paragraph 3 covers generative responses and prompt engineering."
    )
    chunks = chunker.chunk_text(text)
    assert len(chunks) >= 1
    # Check all content is preserved
    joined = " ".join(chunks)
    assert "introduction to RAG systems" in joined
    assert "vector databases" in joined
    assert "generative responses" in joined


def test_chunker_long_text_splitting():
    chunker = RecursiveChunker(target_tokens=30, overlap_tokens=5)
    # Generate long text with multiple sentences
    sentences = [f"This is sentence number {i} containing important information about topic X." for i in range(25)]
    text = " ".join(sentences)
    chunks = chunker.chunk_text(text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) > 0


def test_chunker_overlap():
    chunker = RecursiveChunker(target_tokens=30, overlap_tokens=10)
    text = (
        "Alpha section explains the core fundamentals. "
        "Beta section goes into advanced retrieval architectures. "
        "Gamma section covers deployment and production scaling."
    )
    chunks = chunker.chunk_text(text)
    if len(chunks) > 1:
        # Check that there is overlap or continuous progression
        assert any(word in chunks[1] for word in chunks[0].split()[-3:])


def test_chunk_pages_preserves_page_numbers():
    chunker = RecursiveChunker(target_tokens=50, overlap_tokens=10)
    pages = [
        ParsedPage(page_number=1, text="Page 1 text content discussing machine learning."),
        ParsedPage(page_number=2, text="Page 2 text content discussing deep learning architectures."),
        ParsedPage(page_number=None, text="TXT document content without page number.")
    ]
    doc_id = "test_doc_123"
    text_chunks = chunker.chunk_pages(pages, document_id=doc_id)

    assert len(text_chunks) == 3
    assert text_chunks[0].page_number == 1
    assert text_chunks[0].chunk_index == 0
    assert text_chunks[0].document_id == doc_id

    assert text_chunks[1].page_number == 2
    assert text_chunks[1].chunk_index == 1

    assert text_chunks[2].page_number is None
    assert text_chunks[2].chunk_index == 2
