"""Retriever handling SHA-256 hashing, duplicate prevention, candidate filtering, and Cross-Encoder reranking."""
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from .chunker import RecursiveChunker, TextChunk
from .document_parser import ParsedPage
from .embeddings import EmbeddingService
from .reranker import RerankerService

logger = logging.getLogger(__name__)


class RAGRetriever:
    """
    Manages document indexing into ChromaDB with SHA-256 deduplication and
    a multi-stage retrieval pipeline:
    Stage 1: Bi-encoder vector search (ChromaDB top-K)
    Stage 2: Cosine distance threshold filtering
    Stage 3: Cross-Encoder re-ranking
    Stage 4: Final top-K context selection
    """

    def __init__(
        self,
        chroma_client: chromadb.ClientAPI,
        collection_name: str = "documents",
        embedding_service: Optional[EmbeddingService] = None,
        chunker: Optional[RecursiveChunker] = None,
        reranker_service: Optional[RerankerService] = None,
        distance_threshold: Optional[float] = None,
        initial_retrieval_k: int = 8,
        final_context_k: int = 5
    ):
        self.chroma_client = chroma_client
        self.collection_name = collection_name
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.embedding_service = embedding_service or EmbeddingService()
        self.chunker = chunker or RecursiveChunker()
        self.reranker_service = reranker_service or RerankerService()
        if distance_threshold is not None:
            self.distance_threshold = float(distance_threshold)
        else:
            import os
            self.distance_threshold = float(os.getenv("RAG_DISTANCE_THRESHOLD", 1.0))
        self.initial_retrieval_k = int(initial_retrieval_k)
        self.final_context_k = int(final_context_k)

    @staticmethod
    def calculate_document_hash(file_bytes: bytes) -> str:
        """Calculate deterministic SHA-256 hash from original file bytes."""
        return hashlib.sha256(file_bytes).hexdigest()

    def is_document_indexed(self, document_id: str) -> bool:
        """Check if a document with this SHA-256 hash already exists in ChromaDB."""
        try:
            results = self.collection.get(
                where={"document_id": document_id},
                limit=1
            )
            return bool(results and results.get("ids"))
        except Exception:
            return False

    def ingest_document(
        self,
        file_bytes: bytes,
        filename: str,
        pages: List[ParsedPage]
    ) -> Dict[str, Any]:
        """
        Ingest document pages: check duplicates, chunk, embed, and store in ChromaDB.
        """
        document_id = self.calculate_document_hash(file_bytes)

        # Duplicate check
        if self.is_document_indexed(document_id):
            logger.info(f"Duplicate document skipped: {filename} (SHA-256: {document_id})")
            return {
                "success": True,
                "duplicate": True,
                "document_id": document_id,
                "filename": filename,
                "message": "Document already indexed",
                "chunks_created": 0,
                "chunks_added": 0
            }

        # Chunk document pages
        chunks = self.chunker.chunk_pages(pages, document_id)
        if not chunks:
            raise ValueError("No text chunks could be generated from document.")

        # Generate embeddings
        chunk_texts = [c.text for c in chunks]
        embeddings = self.embedding_service.embed_texts(chunk_texts)

        # Prepare IDs and Chroma-compatible metadata
        ids = [f"{document_id}_{c.chunk_index}" for c in chunks]
        now_iso = datetime.now(timezone.utc).isoformat()
        metadatas = [
            {
                "document_id": document_id,
                "filename": filename,
                "chunk_id": f"{document_id}_{c.chunk_index}",
                "chunk_index": c.chunk_index,
                "total_chunks": len(chunks),
                "page_number": c.page_number if c.page_number is not None else "N/A",
                "created_at": now_iso
            }
            for c in chunks
        ]

        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunk_texts,
            metadatas=metadatas
        )

        logger.info(f"Ingested document '{filename}' ({len(chunks)} chunks, SHA-256: {document_id})")

        return {
            "success": True,
            "duplicate": False,
            "document_id": document_id,
            "filename": filename,
            "chunks_created": len(chunks),
            "chunks_added": len(chunks)
        }

    def retrieve(
        self,
        question: str,
        top_k_candidates: Optional[int] = None,
        max_selected_chunks: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Multi-stage retrieval pipeline:
        Stage 1: Vector retrieval (top_k_candidates, default 8)
        Stage 2: Distance threshold filtering (distance <= threshold)
        Stage 3: Cross-Encoder re-ranking
        Stage 4: Final selection (max_selected_chunks, default 5)
        """
        init_k = top_k_candidates if top_k_candidates is not None else self.initial_retrieval_k
        final_k = max_selected_chunks if max_selected_chunks is not None else self.final_context_k

        if self.collection.count() == 0:
            logger.info("Vector database is empty. No candidates retrieved.")
            return {
                "context_found": False,
                "sources": [],
                "chunks": []
            }

        # ---------------------------------------------------------
        # Stage 1: Vector Retrieval (Bi-encoder embeddings)
        # ---------------------------------------------------------
        q_embedding = self.embedding_service.embed_query(question)
        num_candidates = min(init_k, self.collection.count())
        query_results = self.collection.query(
            query_embeddings=[q_embedding],
            n_results=num_candidates
        )

        documents = query_results.get("documents", [[]])[0]
        distances = query_results.get("distances", [[]])[0]
        metadatas = query_results.get("metadatas", [[]])[0]

        logger.info(f"Question: '{question}' | Vector candidates retrieved: {len(documents)}")

        # ---------------------------------------------------------
        # Stage 2: Distance Threshold Filtering
        # ---------------------------------------------------------
        surviving_candidates: List[Dict[str, Any]] = []
        for doc, dist, meta in zip(documents, distances, metadatas):
            dist_val = float(dist)
            if dist_val <= self.distance_threshold:
                page_val = meta.get("page_number")
                page_num = page_val if isinstance(page_val, int) else None

                clean_text = " ".join(doc.split())
                snippet = clean_text[:180] + "..." if len(clean_text) > 180 else clean_text

                surviving_candidates.append({
                    "filename": meta.get("filename", "unknown"),
                    "page_number": page_num,
                    "chunk_index": meta.get("chunk_index", 0),
                    "document_id": meta.get("document_id", "unknown"),
                    "distance": round(dist_val, 4),
                    "snippet": snippet,
                    "full_text": doc
                })

        logger.info(f"Candidates after distance threshold (<= {self.distance_threshold}): {len(surviving_candidates)}")

        if not surviving_candidates:
            return {
                "context_found": False,
                "sources": [],
                "chunks": []
            }

        # ---------------------------------------------------------
        # Stage 3: Cross-Encoder Re-Ranking
        # ---------------------------------------------------------
        reranked_candidates = self.reranker_service.rerank(
            query=question,
            candidates=surviving_candidates,
            top_k=final_k
        )

        logger.info(f"Candidates selected after Cross-Encoder reranking: {len(reranked_candidates)}")

        sources = []
        chunks = []
        for item in reranked_candidates:
            sources.append({
                "filename": item["filename"],
                "page_number": item["page_number"],
                "chunk_index": item["chunk_index"],
                "document_id": item["document_id"],
                "distance": item["distance"],
                "reranker_score": item.get("reranker_score"),
                "snippet": item["snippet"],
                "full_text": item["full_text"]
            })
            chunks.append(item["full_text"])

        return {
            "context_found": True,
            "sources": sources,
            "chunks": chunks
        }
