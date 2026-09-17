"""Embedding service wrapping SentenceTransformer."""
from typing import List
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Generates dense vector embeddings using SentenceTransformer."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the embedding model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        if not texts:
            return []
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single search query."""
        embedding = self.model.encode([query], show_progress_bar=False)
        return embedding[0].tolist()
