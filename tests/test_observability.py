"""Unit tests for ObservabilityManager, RequestTracker, and JSONL metrics storage."""
import json
import os
import tempfile
from pathlib import Path
import pytest
from rag.observability import ObservabilityManager, RequestTracker


def test_request_tracker_timing_and_fields():
    tracker = RequestTracker(request_id="req_test_123", question="What is Titan?")
    tracker.record_reformulation(search_query="What is Project Titan?", latency_ms=15.2, was_reformulated=True)
    tracker.record_retrieval(
        vector_count=8,
        threshold_count=6,
        reranked_count=5,
        final_count=5,
        context_found=True,
        best_dist=0.18,
        best_rerank=4.52,
        retrieval_latency_ms=25.0,
        rerank_latency_ms=18.0
    )
    tracker.record_llm(latency_ms=250.0, fallback_used=False)

    record = tracker.finish()
    assert record["request_id"] == "req_test_123"
    assert record["reformulation_used"] is True
    assert record["vector_candidates"] == 8
    assert record["threshold_candidates"] == 6
    assert record["reranked_candidates"] == 5
    assert record["final_context_count"] == 5
    assert record["best_distance"] == 0.18
    assert record["best_reranker_score"] == 4.52
    assert record["context_found"] is True
    assert record["fallback_used"] is False
    assert record["total_latency_ms"] >= 0.0
    assert "timestamp" in record
    assert "question" not in record  # By default, questions are not logged


def test_observability_manager_jsonl_appending_and_privacy():
    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = Path(tmpdir) / "test_metrics.jsonl"
        obs = ObservabilityManager(metrics_path=str(jsonl_path), enabled=True, log_questions=False)

        # First request
        t1 = obs.start_request("Secret question about user data")
        t1.record_retrieval(8, 5, 5, 5, True, 0.2, 3.1, 20.0, 15.0)
        t1.record_llm(200.0, False)
        rec1 = obs.log_request_metrics(t1)

        # Second request (with fallback)
        t2 = obs.start_request("Unrelated query")
        t2.record_retrieval(8, 0, 0, 0, False, None, None, 18.0, 0.0)
        t2.record_llm(150.0, True)
        rec2 = obs.log_request_metrics(t2)

        assert jsonl_path.exists()
        lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

        parsed1 = json.loads(lines[0])
        parsed2 = json.loads(lines[1])

        assert parsed1["request_id"].startswith("req_")
        assert parsed2["request_id"].startswith("req_")
        assert "Secret question" not in jsonl_path.read_text(encoding="utf-8")

        # Test Aggregated Metrics
        agg = obs.get_aggregated_metrics()
        assert agg["total_requests"] == 2
        assert agg["context_found_rate"] == 0.5
        assert agg["fallback_rate"] == 0.5
        assert agg["avg_total_latency_ms"] > 0


def test_observability_log_questions_when_enabled():
    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = Path(tmpdir) / "dev_metrics.jsonl"
        obs = ObservabilityManager(metrics_path=str(jsonl_path), enabled=True, log_questions=True)

        t = obs.start_request("Explicit question logged in dev mode")
        obs.log_request_metrics(t)

        content = jsonl_path.read_text(encoding="utf-8")
        assert "Explicit question logged in dev mode" in content


def test_observability_error_recording():
    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = Path(tmpdir) / "error_metrics.jsonl"
        obs = ObservabilityManager(metrics_path=str(jsonl_path), enabled=True, log_questions=False)

        t = obs.start_request("Will fail")
        t.record_error("OpenAI rate limit error (mocked)")
        rec = obs.log_request_metrics(t)

        assert rec["error"] == "OpenAI rate limit error (mocked)"
