"""Cross-Encoder re-ranking service."""
import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class RerankerService:
    """
    Second-stage re-ranking service using a CrossEncoder model.
    Scores (query, passage) pairs to re-order candidate chunks by relevance.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model: Optional[CrossEncoder] = None

    @property
    def model(self) -> CrossEncoder:
        """Lazy-load the CrossEncoder model once."""
        if self._model is None:
            logger.info(f"Loading CrossEncoder model: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Re-rank candidates using the CrossEncoder.

        Args:
            query: The user query string.
            candidates: List of candidate dictionaries containing at least
                        'full_text' (or 'text') and metadata.
            top_k: Maximum number of top candidates to return.

        Returns:
            Top K candidates sorted by reranker_score descending.
            If reranking fails, falls back gracefully to original vector candidate order.
        """
        if not candidates:
            return []

        # Prepare sentence pairs: [query, chunk_text]
        pairs = []
        for c in candidates:
            chunk_content = c.get("full_text") or c.get("text") or c.get("snippet", "")
            pairs.append([query, chunk_content])

        try:
            scores = self.model.predict(pairs)
            # Attach reranker_score to each candidate
            scored_candidates = []
            for c, score in zip(candidates, scores):
                c_copy = dict(c)
                c_copy["reranker_score"] = round(float(score), 4)
                scored_candidates.append(c_copy)

            # Sort by reranker_score descending (higher = more relevant)
            scored_candidates.sort(key=lambda x: x["reranker_score"], reverse=True)
            return scored_candidates[:top_k]

        except Exception as e:
            logger.warning(
                f"CrossEncoder reranking failed ({str(e)}). "
                f"Falling back safely to vector retrieval candidate ranking."
            )
            # Fallback: keep candidates as-is, set reranker_score to None
            fallback_candidates = []
            for c in candidates:
                c_copy = dict(c)
                c_copy["reranker_score"] = None
                fallback_candidates.append(c_copy)
            return fallback_candidates[:top_k]
