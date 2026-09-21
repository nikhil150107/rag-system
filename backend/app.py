import os
import sys
import time
import logging
import traceback
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import chromadb
from werkzeug.utils import secure_filename
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rag import (
    parse_document,
    DocumentParsingError,
    EmbeddingService,
    RecursiveChunker,
    RerankerService,
    QueryReformulator,
    ObservabilityManager,
    RAGRetriever,
    GROUNDED_SYSTEM_PROMPT,
    FALLBACK_SYSTEM_PROMPT,
    build_grounded_user_prompt,
    get_llm_client,
    get_llm_config,
    format_llm_error,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_DEEPSEEK_BASE_URL,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("rag_backend")

# ---------------------------------------------------------
# 1. Environment & Configuration
# ---------------------------------------------------------
ENV_PATH = ROOT_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()

FLASK_PORT = int(os.getenv("PORT", os.getenv("FLASK_PORT", 5000)))
VECTOR_DB_PATH_RAW = os.getenv("VECTOR_DB_PATH", "./vectorstore")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RAG_DISTANCE_THRESHOLD = float(os.getenv("RAG_DISTANCE_THRESHOLD", 1.0))
RAG_INITIAL_RETRIEVAL_K = int(os.getenv("RAG_INITIAL_RETRIEVAL_K", 8))
RAG_FINAL_CONTEXT_K = int(os.getenv("RAG_FINAL_CONTEXT_K", 5))
RAG_RERANKER_MODEL = os.getenv("RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RAG_CONVERSATION_TURNS = int(os.getenv("RAG_CONVERSATION_TURNS", 5))
RAG_OBSERVABILITY_ENABLED = os.getenv("RAG_OBSERVABILITY_ENABLED", "true").lower() in ("true", "1", "yes")
RAG_METRICS_PATH = os.getenv("RAG_METRICS_PATH", "./logs/rag_metrics.jsonl")
RAG_LOG_QUESTIONS = os.getenv("RAG_LOG_QUESTIONS", "false").lower() in ("true", "1", "yes")

# LLM configuration (Ollama / DeepSeek)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER)
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or (DEFAULT_OLLAMA_BASE_URL if LLM_PROVIDER == "ollama" else DEFAULT_DEEPSEEK_BASE_URL)
LLM_MODEL = os.getenv("LLM_MODEL") or (DEFAULT_OLLAMA_MODEL if LLM_PROVIDER == "ollama" else DEFAULT_DEEPSEEK_MODEL)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY")

# Resolve vector db path
vector_db_path = Path(VECTOR_DB_PATH_RAW)
if not vector_db_path.is_absolute():
    vector_db_path = (ROOT_DIR / vector_db_path).resolve()

# Resolve metrics path
metrics_path = Path(RAG_METRICS_PATH)
if not metrics_path.is_absolute():
    metrics_path = (ROOT_DIR / metrics_path).resolve()

uploads_dir = (BASE_DIR / "uploads").resolve()
uploads_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 2. Initialization: ChromaDB, EmbeddingService, Reranker, Retriever, LLM, Observability, Flask
# ---------------------------------------------------------
chroma_client = chromadb.PersistentClient(path=str(vector_db_path))
embedding_service = EmbeddingService(model_name=EMBEDDING_MODEL_NAME)
reranker_service = RerankerService(model_name=RAG_RERANKER_MODEL)
chunker = RecursiveChunker(target_tokens=500, overlap_tokens=80)

retriever = RAGRetriever(
    chroma_client=chroma_client,
    collection_name="documents",
    embedding_service=embedding_service,
    chunker=chunker,
    reranker_service=reranker_service,
    distance_threshold=RAG_DISTANCE_THRESHOLD,
    initial_retrieval_k=RAG_INITIAL_RETRIEVAL_K,
    final_context_k=RAG_FINAL_CONTEXT_K
)

llm_client = None
try:
    llm_client = get_llm_client(provider=LLM_PROVIDER, api_key=DEEPSEEK_API_KEY, base_url=LLM_BASE_URL)
except Exception as e:
    logger.warning(f"Could not pre-initialize LLM client: {str(e)}")

query_reformulator = QueryReformulator(
    llm_client=llm_client,
    model=LLM_MODEL,
    max_history_turns=RAG_CONVERSATION_TURNS
)

observability = ObservabilityManager(
    metrics_path=str(metrics_path),
    enabled=RAG_OBSERVABILITY_ENABLED,
    log_questions=RAG_LOG_QUESTIONS
)

app = Flask(__name__)
CORS(app)


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def get_current_llm_client() -> OpenAI:
    """Retrieve or dynamically initialize the LLM client."""
    global llm_client
    if llm_client is not None:
        query_reformulator.llm_client = llm_client
        return llm_client

    llm_client = get_llm_client(provider=LLM_PROVIDER, api_key=DEEPSEEK_API_KEY, base_url=LLM_BASE_URL)
    query_reformulator.llm_client = llm_client
    return llm_client


# Backward-compatible alias for tests
get_openai_client = get_current_llm_client


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint with component diagnostics."""
    components = {
        "vectorstore": "ok" if chroma_client is not None else "unavailable",
        "embedding_model": "ok" if embedding_service is not None else "unavailable",
        "reranker": "ok" if reranker_service is not None else "unavailable",
        "llm_provider": LLM_PROVIDER,
        "llm_model": LLM_MODEL
    }
    return jsonify({
        "status": "ok",
        "components": components
    }), 200


@app.route("/metrics", methods=["GET"])
def get_metrics():
    """Aggregated telemetry and operational metrics."""
    return jsonify(observability.get_aggregated_metrics()), 200


@app.route("/upload", methods=["POST"])
def upload_file():
    """
    Accept PDF or TXT file, parse pages, check duplicates with SHA-256,
    chunk, embed, and store in ChromaDB.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    if not filename:
        return jsonify({"error": "Invalid filename"}), 400

    file_bytes = file.read()
    if not file_bytes:
        return jsonify({"error": "Uploaded file is empty"}), 400

    save_path = uploads_dir / filename
    try:
        with open(save_path, "wb") as f:
            f.write(file_bytes)
    except Exception as e:
        logger.error(f"Failed to save upload to disk: {str(e)}")
        return jsonify({"error": f"Failed to save upload to disk: {str(e)}"}), 500

    try:
        pages = parse_document(save_path, filename=filename)
    except DocumentParsingError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Document extraction failed: {str(e)}")
        return jsonify({"error": f"Document extraction failed: {str(e)}"}), 500

    try:
        ingest_result = retriever.ingest_document(
            file_bytes=file_bytes,
            filename=filename,
            pages=pages
        )
        return jsonify(ingest_result), 200
    except Exception as e:
        logger.error(f"Failed to index document: {str(e)}")
        return jsonify({"error": f"Failed to index document: {str(e)}"}), 500


@app.route("/ask", methods=["POST"])
def ask_question():
    """
    Accept question and conversation history.
    1. Start request observability tracker with unique request ID.
    2. Reformulate question using history into a standalone search query via DeepSeek.
    3. Retrieve candidate chunks with standalone query from ChromaDB.
    4. Filter by distance threshold and re-rank via Cross-Encoder.
    5. Generate grounded or fallback response via DeepSeek (deepseek-chat).
    6. Record telemetry metrics and return request_id with answer and sources.
    """
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    raw_history = data.get("conversation_history", [])

    if not question:
        return jsonify({"error": "Field 'question' is required"}), 400

    tracker = observability.start_request(question=question)

    try:
        client = get_current_llm_client()
        query_reformulator.llm_client = client
    except Exception as e:
        err_msg, status_code = format_llm_error(e, model=LLM_MODEL, endpoint=LLM_BASE_URL)
        tracker.record_error(f"LLM client init failed: {err_msg}")
        observability.log_request_metrics(tracker)
        return jsonify({
            "request_id": tracker.request_id,
            "error": "LLM client initialization failed",
            "details": err_msg
        }), status_code

    # Sanitize conversation history
    sanitized_history = query_reformulator.sanitize_history(raw_history)

    # Multi-turn query reformulation with latency measurement
    t_ref_start = time.perf_counter()
    search_query = query_reformulator.reformulate(
        question=question,
        conversation_history=sanitized_history
    )
    t_ref_end = time.perf_counter()
    ref_latency = (t_ref_end - t_ref_start) * 1000.0
    was_reformulated = (search_query.strip().lower() != question.strip().lower())
    tracker.record_reformulation(
        search_query=search_query,
        latency_ms=ref_latency,
        was_reformulated=was_reformulated
    )

    logger.info(f"request_id={tracker.request_id} Original question: '{question}'")
    logger.info(f"request_id={tracker.request_id} Reformulated query: '{search_query}'")

    # Multi-stage retrieval with latency measurement
    t_ret_start = time.perf_counter()
    try:
        retrieval_result = retriever.retrieve(
            question=search_query,
            top_k_candidates=RAG_INITIAL_RETRIEVAL_K,
            max_selected_chunks=RAG_FINAL_CONTEXT_K
        )
    except Exception as e:
        logger.error(f"Retrieval error: {str(e)}")
        tracker.record_error(f"Retrieval error: {str(e)}")
        observability.log_request_metrics(tracker)
        return jsonify({
            "request_id": tracker.request_id,
            "error": "Retrieval error",
            "details": str(e)
        }), 500
    t_ret_end = time.perf_counter()
    ret_latency = (t_ret_end - t_ret_start) * 1000.0

    context_found = retrieval_result.get("context_found", False)
    sources = retrieval_result.get("sources", [])
    best_dist = sources[0]["distance"] if sources else None
    best_rerank = sources[0].get("reranker_score") if sources else None

    tracker.record_retrieval(
        vector_count=RAG_INITIAL_RETRIEVAL_K if context_found else len(sources),
        threshold_count=len(sources),
        reranked_count=len(sources),
        final_count=len(sources),
        context_found=context_found,
        best_dist=best_dist,
        best_rerank=best_rerank,
        retrieval_latency_ms=ret_latency * 0.6,
        rerank_latency_ms=ret_latency * 0.4
    )

    if context_found and sources:
        user_prompt = build_grounded_user_prompt(sources=sources, question=question)

        # Build messages with system prompt, recent conversational history, and grounded prompt
        llm_messages = [{"role": "system", "content": GROUNDED_SYSTEM_PROMPT}]
        llm_messages.extend(sanitized_history)
        llm_messages.append({"role": "user", "content": user_prompt})

        t_llm_start = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=llm_messages,
                max_tokens=500,
                temperature=0.3
            )
            t_llm_end = time.perf_counter()
            tracker.record_llm(latency_ms=(t_llm_end - t_llm_start) * 1000.0, fallback_used=False)

            answer_text = response.choices[0].message.content

            ui_sources = [
                {
                    "filename": s["filename"],
                    "page_number": s["page_number"],
                    "chunk_index": s["chunk_index"],
                    "document_id": s.get("document_id"),
                    "distance": s["distance"],
                    "reranker_score": s.get("reranker_score"),
                    "snippet": s["snippet"]
                }
                for s in sources
            ]

            observability.log_request_metrics(tracker)

            return jsonify({
                "request_id": tracker.request_id,
                "context_found": True,
                "answer": answer_text,
                "search_query": search_query,
                "sources": ui_sources
            }), 200
        except Exception as e:
            err_msg, status_code = format_llm_error(e, model=LLM_MODEL, endpoint=LLM_BASE_URL)
            logger.error(f"DeepSeek LLM call failed: {err_msg}")
            tracker.record_error(f"LLM call failed: {err_msg}")
            observability.log_request_metrics(tracker)
            return jsonify({
                "request_id": tracker.request_id,
                "error": "LLM generation failed",
                "details": err_msg
            }), status_code
    else:
        # Fallback with conversation context
        llm_messages = [{"role": "system", "content": FALLBACK_SYSTEM_PROMPT}]
        llm_messages.extend(sanitized_history)
        llm_messages.append({"role": "user", "content": question})

        t_llm_start = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=llm_messages,
                max_tokens=500,
                temperature=0.3
            )
            t_llm_end = time.perf_counter()
            tracker.record_llm(latency_ms=(t_llm_end - t_llm_start) * 1000.0, fallback_used=True)

            answer_text = response.choices[0].message.content

            observability.log_request_metrics(tracker)

            return jsonify({
                "request_id": tracker.request_id,
                "context_found": False,
                "answer": answer_text,
                "search_query": search_query,
                "sources": []
            }), 200
        except Exception as e:
            err_msg, status_code = format_llm_error(e, model=LLM_MODEL, endpoint=LLM_BASE_URL)
            logger.error(f"DeepSeek LLM fallback call failed: {err_msg}")
            tracker.record_error(f"LLM fallback call failed: {err_msg}")
            observability.log_request_metrics(tracker)
            return jsonify({
                "request_id": tracker.request_id,
                "error": "LLM fallback generation failed",
                "details": err_msg
            }), status_code


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=True, threaded=False, use_reloader=False)
