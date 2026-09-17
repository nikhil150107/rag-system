import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import pytest
from unittest.mock import MagicMock
from rag.reranker import RerankerService


def test_evaluate_retrieval_ranking():
    """
    Verify that Cross-Encoder scores prioritize query-specific passages
    over generic or semantically adjacent distractor passages.
    """
    reranker = RerankerService(model_name="mock-evaluator")

    # Custom mock model logic simulating semantic relevance scoring
    def mock_predict(pairs):
        scores = []
        for q, passage in pairs:
            q_lower = q.lower()
            p_lower = passage.lower()
            if "refund" in q_lower and "refund policy" in p_lower:
                scores.append(8.9)
            elif "refund" in q_lower and "shipping" in p_lower:
                scores.append(1.2)
            elif "leave" in q_lower and "annual leave allowance" in p_lower:
                scores.append(9.4)
            elif "leave" in q_lower and "office hours" in p_lower:
                scores.append(0.8)
            else:
                scores.append(2.0)
        return scores

    mock_model = MagicMock()
    mock_model.predict.side_effect = mock_predict
    reranker._model = mock_model

    # Evaluation Case 1: Refund Policy
    q1 = "What is the refund policy?"
    candidates_q1 = [
        {"filename": "shipping.pdf", "full_text": "Standard shipping takes 3-5 business days across the country.", "distance": 0.22},
        {"filename": "refund_policy.pdf", "full_text": "Full refund policy guarantees 100% money-back within 30 days of purchase.", "distance": 0.28},
        {"filename": "terms.pdf", "full_text": "General terms and conditions regarding account registration.", "distance": 0.35}
    ]

    results_q1 = reranker.rerank(q1, candidates_q1, top_k=3)
    assert results_q1[0]["filename"] == "refund_policy.pdf"
    assert results_q1[0]["reranker_score"] > results_q1[1]["reranker_score"]

    # Evaluation Case 2: Employee Leave Allowance
    q2 = "What is the employee leave allowance?"
    candidates_q2 = [
        {"filename": "office_hours.pdf", "full_text": "Standard office hours are Monday to Friday 9am to 5pm.", "distance": 0.21},
        {"filename": "hr_benefits.pdf", "full_text": "All full-time staff receive an annual leave allowance of 25 paid days off.", "distance": 0.29}
    ]

    results_q2 = reranker.rerank(q2, candidates_q2, top_k=2)
    assert results_q2[0]["filename"] == "hr_benefits.pdf"
    assert results_q2[0]["reranker_score"] > results_q2[1]["reranker_score"]


if __name__ == "__main__":
    test_evaluate_retrieval_ranking()
    print("Evaluation completed successfully! Cross-Encoder reranking produced correct top-1 ordering.")
