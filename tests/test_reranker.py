"""Unit tests for CrossEncoder reranker service."""
import pytest
from unittest.mock import MagicMock, patch
from rag.reranker import RerankerService


def test_reranker_receives_correct_pairs():
    reranker = RerankerService(model_name="mock-model")
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.8, 0.4]
    reranker._model = mock_model

    query = "What is quantum error correction?"
    candidates = [
        {"filename": "doc1.pdf", "full_text": "Surface codes enable quantum error correction.", "distance": 0.25},
        {"filename": "doc2.pdf", "full_text": "Classical processors use transistor logic gates.", "distance": 0.35}
    ]

    reranked = reranker.rerank(query, candidates, top_k=2)

    # Verify model was called with [[query, chunk_text_1], [query, chunk_text_2]]
    mock_model.predict.assert_called_once_with([
        [query, "Surface codes enable quantum error correction."],
        [query, "Classical processors use transistor logic gates."]
    ])
    assert len(reranked) == 2


def test_candidates_sorted_by_reranker_score_descending():
    reranker = RerankerService(model_name="mock-model")
    mock_model = MagicMock()
    # Scores for A, B, C
    mock_model.predict.return_value = [1.2, 4.8, 2.5]
    reranker._model = mock_model

    candidates = [
        {"filename": "A.pdf", "full_text": "Content A", "distance": 0.2},
        {"filename": "B.pdf", "full_text": "Content B", "distance": 0.4},
        {"filename": "C.pdf", "full_text": "Content C", "distance": 0.3}
    ]

    reranked = reranker.rerank("test query", candidates, top_k=3)

    # Expected order: B (4.8), C (2.5), A (1.2)
    assert reranked[0]["filename"] == "B.pdf"
    assert reranked[0]["reranker_score"] == 4.8
    assert reranked[1]["filename"] == "C.pdf"
    assert reranked[1]["reranker_score"] == 2.5
    assert reranked[2]["filename"] == "A.pdf"
    assert reranked[2]["reranker_score"] == 1.2


def test_only_final_k_candidates_returned():
    reranker = RerankerService(model_name="mock-model")
    mock_model = MagicMock()
    mock_model.predict.return_value = [5.0, 4.0, 3.0, 2.0]
    reranker._model = mock_model

    candidates = [
        {"filename": f"doc_{i}.pdf", "full_text": f"Content {i}", "distance": 0.1 * i}
        for i in range(4)
    ]

    reranked = reranker.rerank("query", candidates, top_k=2)
    assert len(reranked) == 2
    assert reranked[0]["filename"] == "doc_0.pdf"
    assert reranked[1]["filename"] == "doc_1.pdf"


def test_zero_candidates_does_not_invoke_model():
    reranker = RerankerService(model_name="mock-model")
    mock_model = MagicMock()
    reranker._model = mock_model

    result = reranker.rerank("query", [], top_k=5)
    assert result == []
    mock_model.predict.assert_not_called()


def test_reranker_failure_falls_back_safely():
    reranker = RerankerService(model_name="mock-model")
    mock_model = MagicMock()
    mock_model.predict.side_effect = RuntimeError("GPU out of memory or inference crash")
    reranker._model = mock_model

    candidates = [
        {"filename": "first.pdf", "full_text": "First content", "distance": 0.15},
        {"filename": "second.pdf", "full_text": "Second content", "distance": 0.25}
    ]

    # Should not raise exception; falls back to vector candidates
    fallback_result = reranker.rerank("query", candidates, top_k=2)
    assert len(fallback_result) == 2
    assert fallback_result[0]["filename"] == "first.pdf"
    assert fallback_result[1]["filename"] == "second.pdf"


def test_metadata_preserved_during_reranking():
    reranker = RerankerService(model_name="mock-model")
    mock_model = MagicMock()
    mock_model.predict.return_value = [3.75]
    reranker._model = mock_model

    candidate = {
        "filename": "specs.pdf",
        "page_number": 4,
        "chunk_index": 2,
        "document_id": "sha256_hash_value_123",
        "distance": 0.31,
        "snippet": "Short excerpt...",
        "full_text": "Full text of chunk on page 4."
    }

    reranked = reranker.rerank("query", [candidate], top_k=1)
    assert len(reranked) == 1
    item = reranked[0]

    assert item["filename"] == "specs.pdf"
    assert item["page_number"] == 4
    assert item["chunk_index"] == 2
    assert item["document_id"] == "sha256_hash_value_123"
    assert item["distance"] == 0.31
    assert item["reranker_score"] == 3.75
    assert item["snippet"] == "Short excerpt..."
    assert item["full_text"] == "Full text of chunk on page 4."
