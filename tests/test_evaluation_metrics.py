"""Unit tests for pure evaluation metric calculations (Recall@K, MRR)."""
import pytest
from evaluation.retrieval_evaluator import calculate_recall_at_k, calculate_mrr


def test_recall_at_k_exact_and_substring_match():
    retrieved = ["doc_b.pdf", "doc_a.pdf", "doc_c.pdf"]

    # Target is at rank 2
    assert calculate_recall_at_k(retrieved, "doc_a.pdf", k=1) == 0.0
    assert calculate_recall_at_k(retrieved, "doc_a.pdf", k=2) == 1.0
    assert calculate_recall_at_k(retrieved, "doc_a.pdf", k=3) == 1.0
    assert calculate_recall_at_k(retrieved, "doc_a.pdf", k=5) == 1.0


def test_recall_at_k_missing_target():
    retrieved = ["doc_b.pdf", "doc_c.pdf"]
    assert calculate_recall_at_k(retrieved, "doc_z.pdf", k=5) == 0.0
    assert calculate_recall_at_k([], "doc_a.pdf", k=3) == 0.0
    assert calculate_recall_at_k(retrieved, None, k=3) == 0.0


def test_mrr_calculation_ranks():
    # Rank 1 -> 1.0
    assert calculate_mrr(["doc_a.pdf", "doc_b.pdf", "doc_c.pdf"], "doc_a.pdf") == 1.0

    # Rank 2 -> 0.5
    assert calculate_mrr(["doc_b.pdf", "doc_a.pdf", "doc_c.pdf"], "doc_a.pdf") == 0.5

    # Rank 3 -> 1/3 ~ 0.3333333333333333
    assert pytest.approx(calculate_mrr(["doc_b.pdf", "doc_c.pdf", "doc_a.pdf"], "doc_a.pdf"), 0.001) == 1.0 / 3.0

    # Rank 4 -> 0.25
    assert calculate_mrr(["doc_b.pdf", "doc_c.pdf", "doc_d.pdf", "doc_a.pdf"], "doc_a.pdf") == 0.25


def test_mrr_not_found():
    assert calculate_mrr(["doc_b.pdf", "doc_c.pdf"], "doc_a.pdf") == 0.0
    assert calculate_mrr([], "doc_a.pdf") == 0.0
    assert calculate_mrr(["doc_b.pdf"], None) == 0.0
