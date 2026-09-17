"""Unit tests for retriever, hashing, deduplication, and candidate thresholding."""
import pytest
import chromadb
from rag.retriever import RAGRetriever
from rag.document_parser import ParsedPage
from rag.chunker import RecursiveChunker


class MockEmbeddingService:
    """Deterministic lightweight mock embedding service for offline tests."""
    def embed_texts(self, texts):
        # Return deterministic dummy vectors based on text length
        return [[float(len(t) % 10), 0.5, 0.2] for t in texts]

    def embed_query(self, query):
        return [float(len(query) % 10), 0.5, 0.2]


@pytest.fixture
def in_memory_retriever():
    # Ephemeral in-memory ChromaDB client for isolated testing
    client = chromadb.EphemeralClient()
    mock_emb = MockEmbeddingService()
    retriever = RAGRetriever(
        chroma_client=client,
        collection_name="test_collection",
        embedding_service=mock_emb,
        chunker=RecursiveChunker(target_tokens=50, overlap_tokens=10),
        distance_threshold=0.6
    )
    return retriever


def test_sha256_document_hashing():
    bytes1 = b"Sample document content for hashing test"
    bytes2 = b"Sample document content for hashing test"
    bytes3 = b"Different document content"

    hash1 = RAGRetriever.calculate_document_hash(bytes1)
    hash2 = RAGRetriever.calculate_document_hash(bytes2)
    hash3 = RAGRetriever.calculate_document_hash(bytes3)

    assert hash1 == hash2
    assert hash1 != hash3
    assert len(hash1) == 64


def test_duplicate_prevention(in_memory_retriever):
    file_bytes = b"Unique document content for deduplication test."
    filename = "doc1.txt"
    pages = [ParsedPage(page_number=None, text="Unique document content for deduplication test.")]

    # First ingestion -> success, duplicate: false
    res1 = in_memory_retriever.ingest_document(file_bytes, filename, pages)
    assert res1["success"] is True
    assert res1["duplicate"] is False
    assert res1["chunks_created"] >= 1
    assert in_memory_retriever.collection.count() == res1["chunks_created"]

    # Second ingestion with same file bytes -> duplicate: true, no extra chunks created
    initial_count = in_memory_retriever.collection.count()
    res2 = in_memory_retriever.ingest_document(file_bytes, filename, pages)
    assert res2["success"] is True
    assert res2["duplicate"] is True
    assert res2["message"] == "Document already indexed"
    assert res2["chunks_created"] == 0
    assert in_memory_retriever.collection.count() == initial_count


def test_metadata_structure(in_memory_retriever):
    file_bytes = b"PDF content on page 3"
    filename = "manual.pdf"
    pages = [ParsedPage(page_number=3, text="This is text on page 3 of the operations manual.")]

    res = in_memory_retriever.ingest_document(file_bytes, filename, pages)
    doc_id = res["document_id"]

    records = in_memory_retriever.collection.get(where={"document_id": doc_id})
    metadatas = records["metadatas"]
    assert len(metadatas) > 0
    meta = metadatas[0]

    assert meta["document_id"] == doc_id
    assert meta["filename"] == filename
    assert meta["page_number"] == 3
    assert meta["chunk_index"] == 0
    assert meta["total_chunks"] >= 1
    assert "created_at" in meta


def test_retrieval_threshold_filtering(in_memory_retriever):
    file_bytes = b"Python Flask and ChromaDB architecture details"
    filename = "arch.txt"
    pages = [ParsedPage(page_number=None, text="Python Flask and ChromaDB architecture details.")]
    in_memory_retriever.ingest_document(file_bytes, filename, pages)

    # Retrieval with threshold 0.6
    res = in_memory_retriever.retrieve("Flask architecture", top_k_candidates=8, max_selected_chunks=5)
    # Regardless of mock distance, verify structure
    assert "context_found" in res
    assert "sources" in res
    assert "chunks" in res
    if res["context_found"]:
        for src in res["sources"]:
            assert src["distance"] <= in_memory_retriever.distance_threshold
            assert "snippet" in src
            assert "filename" in src
