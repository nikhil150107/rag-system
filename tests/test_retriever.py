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
        distance_threshold=1.0
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

    # Retrieval with threshold 1.0
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


def test_broad_thematic_query_retrieval():
    """Regression test: verify broad/meta queries (e.g. 'which topic is covered in this pdf?') retrieve candidates successfully."""
    client = chromadb.EphemeralClient()
    mock_emb = MockEmbeddingService()
    retriever = RAGRetriever(
        chroma_client=client,
        collection_name="broad_query_test",
        embedding_service=mock_emb,
        chunker=RecursiveChunker(target_tokens=50, overlap_tokens=10),
        distance_threshold=1.0
    )
    file_bytes = b"Design and Analysis of Algorithms Assignment: Matrix Chain Multiplication using Dynamic Programming."
    filename = "DAA exp 6.pdf"
    pages = [ParsedPage(page_number=1, text=file_bytes.decode('utf-8'))]
    retriever.ingest_document(file_bytes, filename, pages)

    # Broad query
    res = retriever.retrieve("which topic is covered in this pdf?", top_k_candidates=8, max_selected_chunks=5)
    assert res["context_found"] is True
    assert len(res["sources"]) >= 1
    assert res["sources"][0]["filename"] == "DAA exp 6.pdf"


def test_default_distance_threshold_resolution(monkeypatch):
    """Ensure default distance threshold is 1.0 (not 0.6) both with and without env override."""
    client = chromadb.EphemeralClient()
    mock_emb = MockEmbeddingService()

    # Without env var set, default should be 1.0
    monkeypatch.delenv("RAG_DISTANCE_THRESHOLD", raising=False)
    retriever_default = RAGRetriever(
        chroma_client=client,
        collection_name="test_default_thresh",
        embedding_service=mock_emb
    )
    assert retriever_default.distance_threshold == 1.0

    # With env var set to custom value, it respects the env var
    monkeypatch.setenv("RAG_DISTANCE_THRESHOLD", "0.85")
    retriever_env = RAGRetriever(
        chroma_client=client,
        collection_name="test_env_thresh",
        embedding_service=mock_emb
    )
    assert retriever_env.distance_threshold == 0.85


def test_document_isolation_filtering():
    """Regression test: verify retrieval is strictly isolated when document_hash is provided."""
    client = chromadb.EphemeralClient()
    mock_emb = MockEmbeddingService()
    retriever = RAGRetriever(
        chroma_client=client,
        collection_name="doc_isolation_test",
        embedding_service=mock_emb,
        distance_threshold=1.0
    )

    # Ingest Document A: Resume
    resume_bytes = b"Jake Smith Software Engineer Resume: Python, Machine Learning, FastAPI, Cloud Systems."
    resume_pages = [ParsedPage(page_number=1, text=resume_bytes.decode("utf-8"))]
    res_a = retriever.ingest_document(resume_bytes, "resume.pdf", resume_pages)
    hash_resume = res_a["document_id"]

    # Ingest Document B: DAA
    daa_bytes = b"Design and Analysis of Algorithms Assignment: Matrix Chain Multiplication using Dynamic Programming."
    daa_pages = [ParsedPage(page_number=1, text=daa_bytes.decode("utf-8"))]
    res_b = retriever.ingest_document(daa_bytes, "DAA exp 6.pdf", daa_pages)
    hash_daa = res_b["document_id"]

    assert hash_resume != hash_daa

    # 1. Query with DAA document hash -> MUST ONLY return DAA chunks
    res_query_daa = retriever.retrieve("Which topic is covered?", document_hash=hash_daa)
    assert res_query_daa["context_found"] is True
    assert len(res_query_daa["sources"]) >= 1
    for src in res_query_daa["sources"]:
        assert src["document_id"] == hash_daa
        assert src["filename"] == "DAA exp 6.pdf"
        assert "resume" not in src["filename"].lower()

    # 2. Reverse query with Resume document hash -> MUST ONLY return Resume chunks
    res_query_resume = retriever.retrieve("What skills are listed?", document_hash=hash_resume)
    assert res_query_resume["context_found"] is True
    assert len(res_query_resume["sources"]) >= 1
    for src in res_query_resume["sources"]:
        assert src["document_id"] == hash_resume
        assert src["filename"] == "resume.pdf"
        assert "daa" not in src["filename"].lower()

    # 3. Query without document hash (backward compatibility) -> searches whole collection
    res_query_all = retriever.retrieve("Software Algorithms", document_hash=None)
    assert res_query_all["context_found"] is True
    assert len(res_query_all["sources"]) >= 1


def test_get_indexed_documents():
    """Verify retriever.get_indexed_documents() returns all distinct indexed documents."""
    client = chromadb.EphemeralClient()
    mock_emb = MockEmbeddingService()
    retriever = RAGRetriever(
        chroma_client=client,
        collection_name="get_docs_test",
        embedding_service=mock_emb
    )

    doc1_bytes = b"Document 1 Content"
    doc2_bytes = b"Document 2 Content Different"

    retriever.ingest_document(doc1_bytes, "doc1.pdf", [ParsedPage(page_number=1, text=doc1_bytes.decode("utf-8"))])
    retriever.ingest_document(doc2_bytes, "doc2.txt", [ParsedPage(page_number=1, text=doc2_bytes.decode("utf-8"))])

    docs = retriever.get_indexed_documents()
    assert len(docs) == 2
    filenames = [d["filename"] for d in docs]
    assert "doc1.pdf" in filenames
    assert "doc2.txt" in filenames
    assert all("document_id" in d for d in docs)
    assert all("chunks_count" in d for d in docs)


