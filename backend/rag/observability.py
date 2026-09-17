"""Observability and telemetry manager for RAG requests."""
import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger("rag_observability")


class RequestTracker:
    """Tracks latency breakdown and metrics for a single request lifecycle."""

    def __init__(self, request_id: str, question: str = "", log_questions: bool = False):
        self.request_id = request_id
        self.raw_question = question if log_questions else ""
        self.question_length = len(question.strip())
        self.search_query_length = self.question_length
        self.reformulation_used = False
        self.vector_candidates = 0
        self.threshold_candidates = 0
        self.reranked_candidates = 0
        self.final_context_count = 0
        self.best_distance: Optional[float] = None
        self.best_reranker_score: Optional[float] = None
        self.context_found = False
        self.fallback_used = False
        self.error: Optional[str] = None

        self._start_time = time.perf_counter()
        self.reformulation_latency_ms = 0.0
        self.retrieval_latency_ms = 0.0
        self.reranking_latency_ms = 0.0
        self.llm_latency_ms = 0.0
        self.total_latency_ms = 0.0

    def record_reformulation(self, search_query: str, latency_ms: float, was_reformulated: bool):
        self.search_query_length = len(search_query.strip())
        self.reformulation_used = was_reformulated
        self.reformulation_latency_ms = round(latency_ms, 2)

    def record_retrieval(
        self,
        vector_count: int,
        threshold_count: int,
        reranked_count: int,
        final_count: int,
        context_found: bool,
        best_dist: Optional[float],
        best_rerank: Optional[float],
        retrieval_latency_ms: float,
        rerank_latency_ms: float
    ):
        self.vector_candidates = vector_count
        self.threshold_candidates = threshold_count
        self.reranked_candidates = reranked_count
        self.final_context_count = final_count
        self.context_found = context_found
        self.best_distance = best_dist
        self.best_reranker_score = best_rerank
        self.retrieval_latency_ms = round(retrieval_latency_ms, 2)
        self.reranking_latency_ms = round(rerank_latency_ms, 2)

    def record_llm(self, latency_ms: float, fallback_used: bool):
        self.llm_latency_ms = round(latency_ms, 2)
        self.fallback_used = fallback_used

    def record_error(self, error_message: str):
        self.error = error_message

    def finish(self) -> Dict[str, Any]:
        """Complete timing and generate metric dictionary."""
        self.total_latency_ms = round((time.perf_counter() - self._start_time) * 1000.0, 2)
        now_iso = datetime.now(timezone.utc).isoformat()

        metric_data: Dict[str, Any] = {
            "timestamp": now_iso,
            "request_id": self.request_id,
            "question_length": self.question_length,
            "search_query_length": self.search_query_length,
            "reformulation_used": self.reformulation_used,
            "vector_candidates": self.vector_candidates,
            "threshold_candidates": self.threshold_candidates,
            "reranked_candidates": self.reranked_candidates,
            "final_context_count": self.final_context_count,
            "best_distance": self.best_distance,
            "best_reranker_score": self.best_reranker_score,
            "context_found": self.context_found,
            "fallback_used": self.fallback_used,
            "reformulation_latency_ms": self.reformulation_latency_ms,
            "retrieval_latency_ms": self.retrieval_latency_ms,
            "reranking_latency_ms": self.reranking_latency_ms,
            "llm_latency_ms": self.llm_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "error": self.error
        }
        if self.raw_question:
            metric_data["question"] = self.raw_question
        return metric_data


class ObservabilityManager:
    """
    Central observability manager handling structured JSONL metric logging,
    privacy protection, and latency aggregation.
    """

    def __init__(
        self,
        metrics_path: Optional[str] = None,
        enabled: bool = True,
        log_questions: bool = False
    ):
        self.enabled = enabled
        self.log_questions = log_questions
        self._lock = threading.Lock()
        self._memory_records: List[Dict[str, Any]] = []

        if metrics_path:
            self.metrics_file = Path(metrics_path).resolve()
        else:
            default_path = os.getenv("RAG_METRICS_PATH", "./logs/rag_metrics.jsonl")
            self.metrics_file = Path(default_path).resolve()

        if self.enabled:
            self.metrics_file.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def generate_request_id() -> str:
        """Generate unique request identifier."""
        return f"req_{uuid.uuid4().hex[:12]}"

    def start_request(self, question: str = "") -> RequestTracker:
        """Start tracking a new request."""
        req_id = self.generate_request_id()
        return RequestTracker(
            request_id=req_id,
            question=question,
            log_questions=self.log_questions
        )

    def log_request_metrics(self, tracker: RequestTracker) -> Dict[str, Any]:
        """Finish request tracking, append JSONL log, and update in-memory metrics."""
        record = tracker.finish()

        if not self.enabled:
            return record

        with self._lock:
            self._memory_records.append(record)
            # Keep up to last 1000 in memory for fast aggregation
            if len(self._memory_records) > 1000:
                self._memory_records.pop(0)

            try:
                with open(self.metrics_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
            except Exception as e:
                logger.error(f"Failed writing metric to {self.metrics_file}: {str(e)}")

        # Structured log entry
        logger.info(
            f"request_id={record['request_id']} "
            f"vector_candidates={record['vector_candidates']} "
            f"threshold_candidates={record['threshold_candidates']} "
            f"reranked={record['reranked_candidates']} "
            f"context_found={record['context_found']} "
            f"fallback_used={record['fallback_used']} "
            f"latency_ms={record['total_latency_ms']}"
        )
        return record

    def get_aggregated_metrics(self) -> Dict[str, Any]:
        """Compute aggregated operational metrics from recent requests."""
        with self._lock:
            records = list(self._memory_records)

        if not records:
            return {
                "total_requests": 0,
                "context_found_rate": 0.0,
                "fallback_rate": 0.0,
                "avg_total_latency_ms": 0.0,
                "p95_total_latency_ms": 0.0,
                "avg_retrieval_latency_ms": 0.0,
                "avg_reranking_latency_ms": 0.0,
                "avg_llm_latency_ms": 0.0
            }

        total = len(records)
        context_found_count = sum(1 for r in records if r.get("context_found") is True)
        fallback_count = sum(1 for r in records if r.get("fallback_used") is True)

        total_latencies = [r.get("total_latency_ms", 0.0) for r in records]
        retrieval_latencies = [r.get("retrieval_latency_ms", 0.0) for r in records]
        rerank_latencies = [r.get("reranking_latency_ms", 0.0) for r in records]
        llm_latencies = [r.get("llm_latency_ms", 0.0) for r in records]

        sorted_totals = sorted(total_latencies)
        p95_idx = int(len(sorted_totals) * 0.95)
        p95_total = sorted_totals[min(p95_idx, len(sorted_totals) - 1)]

        return {
            "total_requests": total,
            "context_found_rate": round(context_found_count / total, 3),
            "fallback_rate": round(fallback_count / total, 3),
            "avg_total_latency_ms": round(sum(total_latencies) / total, 2),
            "p95_total_latency_ms": round(p95_total, 2),
            "avg_retrieval_latency_ms": round(sum(retrieval_latencies) / total, 2),
            "avg_reranking_latency_ms": round(sum(rerank_latencies) / total, 2),
            "avg_llm_latency_ms": round(sum(llm_latencies) / total, 2)
        }
