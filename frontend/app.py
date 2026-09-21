"""Document Q&A RAG Assistant - Standalone Streamlit Application.

Executes the modular RAG pipeline directly in-process with singleton model caching
via @st.cache_resource and DeepSeek API (deepseek-chat) for high performance.
"""
import os
import sys
import time
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import streamlit as st
import chromadb
from openai import OpenAI

# ---------------------------------------------------------
# 1. Environment & Path Configuration
# ---------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"

# Ensure backend directory is prioritized at the top of sys.path
for path_dir in [str(ROOT_DIR), str(BACKEND_DIR)]:
    if path_dir in sys.path:
        sys.path.remove(path_dir)
    sys.path.insert(0, path_dir)

ENV_PATH = ROOT_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()

# Import modular RAG components directly from backend/rag with fallback
try:
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
        run_diagnostic_probe,
        DEFAULT_LLM_PROVIDER,
        DEFAULT_LLM_MODEL,
        DEFAULT_LLM_BASE_URL,
    )
except ImportError:
    from backend.rag import (
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
        run_diagnostic_probe,
        DEFAULT_LLM_PROVIDER,
        DEFAULT_LLM_MODEL,
        DEFAULT_LLM_BASE_URL,
    )

# ---------------------------------------------------------
# 2. Page Configuration & Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="Document Q&A RAG Assistant (DeepSeek)",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed_documents" not in st.session_state:
    st.session_state.indexed_documents = []


# ---------------------------------------------------------
# 3. Singleton RAG Pipeline Resource Initialization
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Initializing RAG embedding & reranking models...")
def get_rag_components() -> Dict[str, Any]:
    """
    Initialize and cache singleton instances of ChromaDB, Bi-Encoder,
    Cross-Encoder, Chunker, Retriever, and Observability manager.
    Cached across Streamlit reruns to optimize performance and stay within 1 GB RAM.
    """
    vector_db_path = (ROOT_DIR / os.getenv("VECTOR_DB_PATH", "./vectorstore")).resolve()
    metrics_path = (ROOT_DIR / os.getenv("RAG_METRICS_PATH", "./logs/rag_metrics.jsonl")).resolve()

    vector_db_path.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    embedding_model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    reranker_model_name = os.getenv("RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    dist_threshold = float(os.getenv("RAG_DISTANCE_THRESHOLD", 0.6))
    initial_k = int(os.getenv("RAG_INITIAL_RETRIEVAL_K", 8))
    final_k = int(os.getenv("RAG_FINAL_CONTEXT_K", 5))

    chroma_client = chromadb.PersistentClient(path=str(vector_db_path))
    embedding_service = EmbeddingService(model_name=embedding_model_name)
    reranker_service = RerankerService(model_name=reranker_model_name)
    chunker = RecursiveChunker(target_tokens=500, overlap_tokens=80)

    retriever = RAGRetriever(
        chroma_client=chroma_client,
        collection_name="documents",
        embedding_service=embedding_service,
        chunker=chunker,
        reranker_service=reranker_service,
        distance_threshold=dist_threshold,
        initial_retrieval_k=initial_k,
        final_context_k=final_k
    )

    observability = ObservabilityManager(
        metrics_path=str(metrics_path),
        enabled=os.getenv("RAG_OBSERVABILITY_ENABLED", "true").lower() in ("true", "1", "yes"),
        log_questions=os.getenv("RAG_LOG_QUESTIONS", "false").lower() in ("true", "1", "yes")
    )

    return {
        "chroma_client": chroma_client,
        "embedding_service": embedding_service,
        "reranker_service": reranker_service,
        "chunker": chunker,
        "retriever": retriever,
        "observability": observability
    }


def resolve_llm_setting(name: str, default_val: str) -> str:
    """Resolve configuration from st.secrets -> environment variable -> default."""
    try:
        if name in st.secrets and str(st.secrets[name]).strip():
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    env_val = os.getenv(name)
    if env_val and env_val.strip():
        return env_val.strip()
    return default_val


def get_deepseek_api_key() -> Optional[str]:
    """Resolve DeepSeek API key from Streamlit Secrets, environment, or session state."""
    # 1. Streamlit Secrets (for Streamlit Cloud deployment)
    try:
        if "DEEPSEEK_API_KEY" in st.secrets and str(st.secrets["DEEPSEEK_API_KEY"]).strip():
            return str(st.secrets["DEEPSEEK_API_KEY"]).strip()
        if "LLM_API_KEY" in st.secrets and str(st.secrets["LLM_API_KEY"]).strip():
            return str(st.secrets["LLM_API_KEY"]).strip()
    except Exception:
        pass

    # 2. Environment Variables (.env / system)
    key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY")
    if key and key.strip():
        return key.strip()

    # 3. User Sidebar input
    return st.session_state.get("user_deepseek_api_key", "").strip() or None


# Backward-compatible aliases
get_openai_api_key = get_deepseek_api_key
get_xai_api_key = get_deepseek_api_key

# Load components
rag_components = get_rag_components()
retriever = rag_components["retriever"]
observability = rag_components["observability"]

# Provider settings
LLM_PROVIDER = resolve_llm_setting("LLM_PROVIDER", DEFAULT_LLM_PROVIDER)
LLM_BASE_URL = resolve_llm_setting("LLM_BASE_URL", DEFAULT_LLM_BASE_URL)
LLM_MODEL = resolve_llm_setting("LLM_MODEL", DEFAULT_LLM_MODEL)


# ---------------------------------------------------------
# 4. Helper Functions: Rendering
# ---------------------------------------------------------
def render_sources(sources: List[Dict[str, Any]]):
    """Render structured source cards with vector distance and reranker score."""
    if not sources:
        return

    with st.expander(f"📚 Retrieved & Re-Ranked Sources ({len(sources)} chunks)", expanded=False):
        for i, source in enumerate(sources, 1):
            if isinstance(source, dict):
                filename = source.get("filename", "Unknown Document")
                page_val = source.get("page_number")
                page_str = f"Page {page_val}" if page_val is not None else "Page: N/A"
                chunk_idx = source.get("chunk_index", 0)
                dist = source.get("distance", "N/A")
                dist_str = f"{dist:.4f}" if isinstance(dist, (int, float)) else str(dist)
                rerank_val = source.get("reranker_score")
                rerank_str = f"{rerank_val:.4f}" if isinstance(rerank_val, (int, float)) else "N/A"
                snippet = source.get("snippet", "")

                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Source {i}:** `{filename}` | **{page_str}** | **Chunk:** `#{chunk_idx}`")
                with col2:
                    st.caption(f"Dist: `{dist_str}` | Rerank: `{rerank_str}`")

                if snippet:
                    st.info(f"“{snippet}”")
            else:
                st.markdown(f"**Chunk {i}:**")
                st.markdown(f"> {source}")

            if i < len(sources):
                st.divider()


def render_rag_pipeline_info(msg: Dict[str, Any]):
    """Render RAG pipeline execution diagnostics and latency metrics."""
    req_id = msg.get("request_id")
    search_query = msg.get("search_query")
    context_found = msg.get("context_found")
    sources = msg.get("sources", [])
    latency_breakdown = msg.get("latency_breakdown", {})

    with st.expander("🔧 RAG Pipeline & Telemetry", expanded=False):
        st.markdown("**Pipeline Execution Architecture:**")
        st.code(
            f"User Question + History\n"
            f"  ↳ 1. Multi-Turn Query Reformulator ({LLM_MODEL})\n"
            f"  ↳ 2. Dense Vector Retrieval (all-MiniLM-L6-v2 in ChromaDB Top-8)\n"
            f"  ↳ 3. Cosine Distance Threshold Filter (<= 0.6)\n"
            f"  ↳ 4. Cross-Encoder Re-Ranking (ms-marco-MiniLM-L-6-v2 Top-5)\n"
            f"  ↳ 5. Grounded Context-Bound Generation ({LLM_MODEL})",
            language="text"
        )
        col_a, col_b = st.columns(2)
        with col_a:
            if req_id:
                st.markdown(f"**Request ID:** `{req_id}`")
            st.markdown(f"**Context Grounded:** `{'✅ True' if context_found else '❌ False (Fallback Used)'}`")
        with col_b:
            st.markdown(f"**Retrieved Sources:** `{len(sources)} chunks`")
            if search_query:
                st.markdown(f"**Search Query Length:** `{len(search_query)} chars`")

        if latency_breakdown:
            st.markdown("**Latency Breakdown:**")
            l_cols = st.columns(3)
            with l_cols[0]:
                st.metric("Reformulation", f"{latency_breakdown.get('reformulation_ms', 0):.0f} ms")
            with l_cols[1]:
                st.metric("Retrieval + Rerank", f"{latency_breakdown.get('retrieval_ms', 0):.0f} ms")
            with l_cols[2]:
                st.metric("DeepSeek Generation", f"{latency_breakdown.get('llm_ms', 0):.0f} ms")


# ---------------------------------------------------------
# 5. Sidebar: Controls, Document Management & Metrics
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚙️ RAG System")

    # Component Status Indicator
    st.success("🟢 **RAG Pipeline Active (In-Process)**")
    with st.expander("System Components Health", expanded=False):
        st.markdown("- **Vectorstore:** `ChromaDB PersistentClient (Active)`")
        st.markdown("- **Embedding Model:** `all-MiniLM-L6-v2 (Loaded)`")
        st.markdown("- **Cross-Encoder:** `ms-marco-MiniLM-L-6-v2 (Loaded)`")
        st.markdown(f"- **LLM Provider:** `DeepSeek ({LLM_MODEL})`")
        st.markdown(f"- **Base URL:** `{LLM_BASE_URL}`")
        st.markdown("- **Execution Mode:** `In-Process Singleton (@st.cache_resource)`")

        # Diagnostic Probe Button
        test_api_key = get_deepseek_api_key()
        if test_api_key:
            if st.button("🧪 Run DeepSeek Connection Test", key="run_probe_btn", use_container_width=True):
                with st.spinner("Executing progressive API connection probes..."):
                    probe_client = OpenAI(api_key=test_api_key, base_url=LLM_BASE_URL)
                    probe_results = run_diagnostic_probe(client=probe_client, model=LLM_MODEL)
                    all_passed = True
                    for step_name, step_info in probe_results["steps"].items():
                        if step_info["status"] == "PASS":
                            st.caption(f"✅ `{step_name}`: OK")
                        else:
                            all_passed = False
                            st.error(f"❌ `{step_name}` Failed: {step_info.get('error')}")
                    if all_passed:
                        st.success(f"🎉 All probes passed successfully with model `{LLM_MODEL}`!")

    # API Key Configuration
    api_key = get_deepseek_api_key()
    if not api_key:
        st.warning("⚠️ **DeepSeek API Key Missing**")
        user_key = st.text_input(
            "Enter DeepSeek API Key",
            type="password",
            help="Set DEEPSEEK_API_KEY in .env, Streamlit Secrets, or paste here for this session.",
            key="user_key_input"
        )
        if user_key:
            st.session_state["user_deepseek_api_key"] = user_key
            st.rerun()
    else:
        st.caption("🔑 DeepSeek API Key configured")

    st.divider()

    # Document Ingestion Section
    st.subheader("📄 Document Ingestion")
    st.caption("Upload PDF or TXT documents to index into ChromaDB.")

    uploaded_file = st.file_uploader(
        "Select Document",
        type=["pdf", "txt"],
        help="Supports PDF and TXT documents. Extracted text is chunked, embedded, and deduplicated via SHA-256."
    )

    if st.button("🚀 Index Document", use_container_width=True, type="primary"):
        if uploaded_file is not None:
            with st.spinner("Processing, parsing pages, and embedding into ChromaDB..."):
                try:
                    file_bytes = uploaded_file.getvalue()
                    filename = uploaded_file.name

                    # Save to a temporary file for parser compatibility
                    suffix = Path(filename).suffix
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(file_bytes)
                        tmp_path = Path(tmp.name)

                    try:
                        pages = parse_document(tmp_path, filename=filename)
                    finally:
                        if tmp_path.exists():
                            tmp_path.unlink()

                    # Ingest directly into ChromaDB via RAGRetriever
                    ingest_result = retriever.ingest_document(
                        file_bytes=file_bytes,
                        filename=filename,
                        pages=pages
                    )

                    is_duplicate = ingest_result.get("duplicate", False)
                    chunks_count = ingest_result.get("chunks_created", ingest_result.get("chunks_added", 0))

                    # Track in session state
                    doc_info = {
                        "name": filename,
                        "size_kb": round(len(file_bytes) / 1024, 1),
                        "chunks": chunks_count,
                        "duplicate": is_duplicate,
                        "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
                    }
                    if not any(d["name"] == filename for d in st.session_state.indexed_documents):
                        st.session_state.indexed_documents.append(doc_info)

                    if is_duplicate:
                        st.info(f"ℹ️ **Duplicate Detected:** `{filename}` is already indexed. No duplicate chunks added.")
                    else:
                        st.success(f"✅ **Success:** `{filename}` indexed ({chunks_count} chunks stored).")
                except DocumentParsingError as e:
                    st.error(f"❌ Document Parsing Error: {str(e)}")
                except Exception as e:
                    st.error(f"❌ Error during ingestion: {str(e)}")
        else:
            st.warning("Please choose a file before clicking Index.")

    # Indexed Documents List
    if st.session_state.indexed_documents:
        with st.expander(f"📁 Indexed Documents ({len(st.session_state.indexed_documents)})", expanded=False):
            for doc in st.session_state.indexed_documents:
                dup_tag = " [Duplicate]" if doc["duplicate"] else ""
                st.markdown(f"- **`{doc['name']}`** ({doc['size_kb']} KB){dup_tag}")
                st.caption(f"  Chunks: {doc['chunks']} | Added: {doc['timestamp']}")

    st.divider()

    # Telemetry & Operational Metrics
    metrics_data = observability.get_aggregated_metrics()
    if metrics_data:
        with st.expander("📊 Operational Telemetry", expanded=False):
            st.metric("Total Requests", metrics_data.get("total_requests", 0))
            st.metric("Context-Found Rate", f"{metrics_data.get('context_found_rate', 0.0):.1%}")
            st.metric("Avg Latency", f"{metrics_data.get('avg_total_latency_ms', 0.0):.0f} ms")
            st.metric("p95 Latency", f"{metrics_data.get('p95_total_latency_ms', 0.0):.0f} ms")

    # Clear Conversation
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------
# 6. Main Area: Header & Chat Interface
# ---------------------------------------------------------
st.title("📚 Document Q&A RAG Assistant")
st.markdown(
    f"**Enterprise Retrieval-Augmented Generation Platform (Powered by DeepSeek)** — "
    "Ask questions grounded strictly in your uploaded documents. "
    "Features **Conversational Memory**, **Multi-Turn Query Reformulation**, **Dense Bi-Encoder Retrieval**, **Cross-Encoder Re-Ranking**, and **DeepSeek Generation**."
)

st.divider()

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            search_query = msg.get("search_query")
            if search_query:
                with st.expander("🔍 Standalone Retrieval Query", expanded=False):
                    st.markdown("**Reformulated query used for ChromaDB vector search:**")
                    st.code(search_query, language="text")

            if msg.get("context_found") is True and msg.get("sources"):
                render_sources(msg["sources"])
            elif msg.get("context_found") is False:
                st.caption("⚠️ *Answered using general assistant fallback (no matching document context found).*")

            render_rag_pipeline_info(msg)


# ---------------------------------------------------------
# 7. Question Submission & Direct RAG Pipeline Execution
# ---------------------------------------------------------
if prompt := st.chat_input("Ask a question about your uploaded documents..."):
    current_api_key = get_deepseek_api_key()
    if not current_api_key:
        st.error("❌ DEEPSEEK_API_KEY is required to ask questions. Please configure it in Streamlit Secrets or sidebar.")
    else:
        # Build prior conversation history (excluding the current prompt)
        prior_history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
            if m.get("role") in ("user", "assistant")
        ]

        # Append user question to session state and render
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Execute in-process RAG pipeline
        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant passages, re-ranking with Cross-Encoder, and generating answer with DeepSeek..."):
                tracker = observability.start_request(question=prompt)
                try:
                    llm_client = get_llm_client(api_key=current_api_key, base_url=LLM_BASE_URL)
                except Exception as e:
                    err_msg, _ = format_llm_error(e, model=LLM_MODEL, endpoint=LLM_BASE_URL)
                    st.error(f"❌ Configuration Error: {err_msg}")
                    st.stop()

                query_reformulator = QueryReformulator(
                    llm_client=llm_client,
                    model=LLM_MODEL,
                    max_history_turns=int(os.getenv("RAG_CONVERSATION_TURNS", 5))
                )

                try:
                    # Sanitize history
                    sanitized_history = query_reformulator.sanitize_history(prior_history)

                    # Stage 1: Query Reformulation
                    t_ref_start = time.perf_counter()
                    search_query = query_reformulator.reformulate(
                        question=prompt,
                        conversation_history=sanitized_history
                    )
                    t_ref_end = time.perf_counter()
                    ref_latency_ms = (t_ref_end - t_ref_start) * 1000.0
                    was_reformulated = (search_query.strip().lower() != prompt.strip().lower())
                    tracker.record_reformulation(
                        search_query=search_query,
                        latency_ms=ref_latency_ms,
                        was_reformulated=was_reformulated
                    )

                    # Stage 2: Bi-Encoder Retrieval + Distance Filter + Cross-Encoder Reranking
                    t_ret_start = time.perf_counter()
                    retrieval_result = retriever.retrieve(
                        question=search_query,
                        top_k_candidates=int(os.getenv("RAG_INITIAL_RETRIEVAL_K", 8)),
                        max_selected_chunks=int(os.getenv("RAG_FINAL_CONTEXT_K", 5))
                    )
                    t_ret_end = time.perf_counter()
                    ret_latency_ms = (t_ret_end - t_ret_start) * 1000.0

                    context_found = retrieval_result.get("context_found", False)
                    sources = retrieval_result.get("sources", [])
                    best_dist = sources[0]["distance"] if sources else None
                    best_rerank = sources[0].get("reranker_score") if sources else None

                    tracker.record_retrieval(
                        vector_count=int(os.getenv("RAG_INITIAL_RETRIEVAL_K", 8)) if context_found else len(sources),
                        threshold_count=len(sources),
                        reranked_count=len(sources),
                        final_count=len(sources),
                        context_found=context_found,
                        best_dist=best_dist,
                        best_rerank=best_rerank,
                        retrieval_latency_ms=ret_latency_ms * 0.6,
                        rerank_latency_ms=ret_latency_ms * 0.4
                    )

                    # Stage 3: LLM Generation (Grounded vs Fallback)
                    if context_found and sources:
                        user_prompt = build_grounded_user_prompt(sources=sources, question=prompt)
                        llm_messages = [{"role": "system", "content": GROUNDED_SYSTEM_PROMPT}]
                        llm_messages.extend(sanitized_history)
                        llm_messages.append({"role": "user", "content": user_prompt})

                        t_llm_start = time.perf_counter()
                        response = llm_client.chat.completions.create(
                            model=LLM_MODEL,
                            messages=llm_messages,
                            max_tokens=500,
                            temperature=0.3
                        )
                        t_llm_end = time.perf_counter()
                        llm_latency_ms = (t_llm_end - t_llm_start) * 1000.0
                        tracker.record_llm(latency_ms=llm_latency_ms, fallback_used=False)
                    else:
                        llm_messages = [{"role": "system", "content": FALLBACK_SYSTEM_PROMPT}]
                        llm_messages.extend(sanitized_history)
                        llm_messages.append({"role": "user", "content": prompt})

                        t_llm_start = time.perf_counter()
                        response = llm_client.chat.completions.create(
                            model=LLM_MODEL,
                            messages=llm_messages,
                            max_tokens=500,
                            temperature=0.3
                        )
                        t_llm_end = time.perf_counter()
                        llm_latency_ms = (t_llm_end - t_llm_start) * 1000.0
                        tracker.record_llm(latency_ms=llm_latency_ms, fallback_used=True)

                    answer_text = response.choices[0].message.content
                    observability.log_request_metrics(tracker)

                    # UI Rendering
                    st.markdown(answer_text)

                    if search_query:
                        with st.expander("🔍 Standalone Retrieval Query", expanded=False):
                            st.markdown("**Reformulated query used for ChromaDB vector search:**")
                            st.code(search_query, language="text")

                    if context_found and sources:
                        render_sources(sources)
                    elif not context_found:
                        st.caption("⚠️ *Answered using general assistant fallback (no matching document context found).*")

                    msg_data = {
                        "role": "assistant",
                        "content": answer_text,
                        "context_found": context_found,
                        "search_query": search_query,
                        "request_id": tracker.request_id,
                        "sources": sources,
                        "latency_breakdown": {
                            "reformulation_ms": ref_latency_ms,
                            "retrieval_ms": ret_latency_ms,
                            "llm_ms": llm_latency_ms
                        }
                    }
                    render_rag_pipeline_info(msg_data)

                    # Save assistant message to session state
                    st.session_state.messages.append(msg_data)

                except Exception as e:
                    err_msg, _ = format_llm_error(e, model=LLM_MODEL, endpoint=LLM_BASE_URL)
                    tracker.record_error(f"Pipeline error: {err_msg}")
                    observability.log_request_metrics(tracker)
                    st.error(f"❌ {err_msg}")
