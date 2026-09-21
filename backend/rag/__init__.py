"""RAG module package."""
from .document_parser import parse_document, DocumentParsingError, ParsedPage
from .chunker import RecursiveChunker, TextChunk
from .embeddings import EmbeddingService
from .reranker import RerankerService
from .query_reformulator import QueryReformulator
from .prompts import (
    GROUNDED_SYSTEM_PROMPT,
    FALLBACK_SYSTEM_PROMPT,
    REFORMULATION_SYSTEM_PROMPT,
    build_grounded_user_prompt,
    build_reformulation_user_prompt,
)
from .observability import ObservabilityManager, RequestTracker
from .retriever import RAGRetriever
from .llm import (
    get_llm_client,
    get_llm_config,
    format_llm_error,
    run_diagnostic_probe,
    check_ollama_health,
    normalize_ollama_base_url,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_DEEPSEEK_BASE_URL,
)

__all__ = [
    "parse_document",
    "DocumentParsingError",
    "ParsedPage",
    "RecursiveChunker",
    "TextChunk",
    "EmbeddingService",
    "RerankerService",
    "QueryReformulator",
    "ObservabilityManager",
    "RequestTracker",
    "GROUNDED_SYSTEM_PROMPT",
    "FALLBACK_SYSTEM_PROMPT",
    "REFORMULATION_SYSTEM_PROMPT",
    "build_grounded_user_prompt",
    "build_reformulation_user_prompt",
    "RAGRetriever",
    "get_llm_client",
    "get_llm_config",
    "format_llm_error",
    "run_diagnostic_probe",
    "check_ollama_health",
    "normalize_ollama_base_url",
    "DEFAULT_LLM_PROVIDER",
    "DEFAULT_LLM_MODEL",
    "DEFAULT_LLM_BASE_URL",
    "DEFAULT_OLLAMA_MODEL",
    "DEFAULT_OLLAMA_BASE_URL",
    "DEFAULT_DEEPSEEK_MODEL",
    "DEFAULT_DEEPSEEK_BASE_URL",
]
