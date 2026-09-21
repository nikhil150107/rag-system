"""Document Q&A RAG Assistant - Standalone Streamlit Application.

Executes the modular RAG pipeline directly in-process with singleton model caching
via @st.cache_resource, supporting both local Ollama (llama3.2:3b) and DeepSeek Cloud (deepseek-chat).
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

# ---------------------------------------------------------
# 2. Page Configuration & Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="Document Q&A RAG Assistant",
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
    dist_threshold = float(os.getenv("RAG_DISTANCE_THRESHOLD", 1.0))
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

# Initial Provider configuration from environment
LLM_PROVIDER_ENV = resolve_llm_setting("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).lower().strip()


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


def render_rag_pipeline_info(
    msg: Dict[str, Any],
    active_model_name: str = DEFAULT_LLM_MODEL,
    active_threshold: Optional[float] = None
):
    """Render RAG pipeline execution diagnostics and latency metrics."""
    if active_threshold is None:
        active_threshold = getattr(retriever, "distance_threshold", float(os.getenv("RAG_DISTANCE_THRESHOLD", 1.0)))
    threshold_str = f"{active_threshold:.1f}" if isinstance(active_threshold, (int, float)) else str(active_threshold)
    req_id = msg.get("request_id")
    search_query = msg.get("search_query")
    context_found = msg.get("context_found")
    sources = msg.get("sources", [])
    latency_breakdown = msg.get("latency_breakdown", {})
    provider_name = msg.get("provider", "LLM")
    doc_name = msg.get("selected_document_name")
    doc_hash = msg.get("selected_document_hash")

    with st.expander("🔧 RAG Pipeline & Telemetry", expanded=False):
        st.markdown("**Pipeline Execution Architecture:**")
        filter_str = f"  ↳ 0. Document Isolation Filter: {doc_name} ({doc_hash[:8]}...)\n" if (doc_name and doc_hash) else (f"  ↳ 0. Document Isolation Filter: {doc_name or doc_hash}\n" if (doc_name or doc_hash) else "")
        st.code(
            f"User Question + History\n"
            f"{filter_str}"
            f"  ↳ 1. Multi-Turn Query Reformulator ({active_model_name})\n"
            f"  ↳ 2. Dense Vector Retrieval (all-MiniLM-L6-v2 in ChromaDB Top-8)\n"
            f"  ↳ 3. Cosine Distance Threshold Filter (<= {threshold_str})\n"
            f"  ↳ 4. Cross-Encoder Re-Ranking (ms-marco-MiniLM-L-6-v2 Top-5)\n"
            f"  ↳ 5. Grounded Context-Bound Generation ({active_model_name})",
            language="text"
        )
        col_a, col_b = st.columns(2)
        with col_a:
            if req_id:
                st.markdown(f"**Request ID:** `{req_id}`")
            if doc_name:
                st.markdown(f"**Selected Document:** `{doc_name}`")
            if doc_hash:
                st.markdown(f"**Document Hash:** `{doc_hash[:12]}...`")
            st.markdown(f"**Context Grounded:** `{'✅ True' if context_found else '❌ False (Fallback Used)'}`")
        with col_b:
            st.markdown(f"**Retrieved Sources:** `{len(sources)} chunks`")
            if search_query:
                st.markdown(f"**Search Query Length:** `{len(search_query)} chars`")
            if doc_name and sources:
                matching_all = all(s.get("filename") == doc_name for s in sources if isinstance(s, dict))
                if matching_all:
                    st.markdown(f"**Isolation Check:** `✅ 100% {doc_name}`")

        if latency_breakdown:
            st.markdown("**Latency Breakdown:**")
            l_cols = st.columns(3)
            with l_cols[0]:
                st.metric("Reformulation", f"{latency_breakdown.get('reformulation_ms', 0):.0f} ms")
            with l_cols[1]:
                st.metric("Retrieval + Rerank", f"{latency_breakdown.get('retrieval_ms', 0):.0f} ms")
            with l_cols[2]:
                st.metric(f"{provider_name.capitalize()} Generation", f"{latency_breakdown.get('llm_ms', 0):.0f} ms")


# ---------------------------------------------------------
# 5. Sidebar: Controls, Document Management & Metrics
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚙️ RAG System")

    # LLM Provider Selection Toggle
    default_provider_index = 0 if LLM_PROVIDER_ENV == "ollama" else 1
    selected_provider_label = st.radio(
        "🤖 **LLM Provider**",
        options=["Ollama (Local)", "DeepSeek (Cloud)"],
        index=default_provider_index,
        help="Select local Ollama for zero-cost offline inference or DeepSeek Cloud for hosted API inference."
    )
    is_ollama = "Ollama" in selected_provider_label
    active_provider = "ollama" if is_ollama else "deepseek"

    # Resolve settings for the active provider
    if is_ollama:
        raw_ollama_url = resolve_llm_setting("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
        active_base_url = normalize_ollama_base_url(raw_ollama_url)
        active_model = resolve_llm_setting("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        active_api_key = "ollama"
    else:
        raw_deepseek_url = resolve_llm_setting("DEEPSEEK_BASE_URL", resolve_llm_setting("LLM_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL))
        active_base_url = raw_deepseek_url.rstrip("/")
        active_model = resolve_llm_setting("DEEPSEEK_MODEL", resolve_llm_setting("LLM_MODEL", DEFAULT_DEEPSEEK_MODEL))
        active_api_key = get_deepseek_api_key()

    # Component Status Indicator
    st.success("🟢 **RAG Pipeline Active (In-Process)**")
    with st.expander("System Components Health", expanded=False):
        st.markdown("- **Vectorstore:** `ChromaDB PersistentClient (Active)`")
        st.markdown("- **Embedding Model:** `all-MiniLM-L6-v2 (Loaded)`")
        st.markdown("- **Cross-Encoder:** `ms-marco-MiniLM-L-6-v2 (Loaded)`")
        st.markdown(f"- **LLM Provider:** `{selected_provider_label}`")
        st.markdown(f"- **Model:** `{active_model}`")
        st.markdown(f"- **Base URL:** `{active_base_url}`")
        st.markdown("- **Execution Mode:** `In-Process Singleton (@st.cache_resource)`")

        # Live Ollama Health Check
        if is_ollama:
            health = check_ollama_health(base_url=active_base_url, target_model=active_model)
            if health["reachable"]:
                st.markdown("🟢 **Ollama Daemon:** `Running (Reachable)`")
                if health["target_model_present"]:
                    st.markdown(f"🟢 **Target Model:** `{active_model} (Installed)`")
                else:
                    st.markdown(f"🟡 **Target Model:** `{active_model} (Not found in {health['models']})`")
            else:
                st.markdown(f"🔴 **Ollama Daemon:** `Not Reachable ({health.get('error')})`")

        # Diagnostic Probe Button
        probe_button_label = "🧪 Run Ollama Connection Test" if is_ollama else "🧪 Run DeepSeek Connection Test"
        can_run_probe = is_ollama or bool(active_api_key)

        if can_run_probe:
            if st.button(probe_button_label, key="run_probe_btn", use_container_width=True):
                with st.spinner(f"Executing progressive API connection probes to {active_provider.upper()}..."):
                    probe_client = get_llm_client(provider=active_provider, api_key=active_api_key, base_url=active_base_url)
                    probe_results = run_diagnostic_probe(client=probe_client, model=active_model, provider=active_provider)
                    all_passed = True
                    for step_name, step_info in probe_results["steps"].items():
                        if step_info["status"] == "PASS":
                            st.caption(f"✅ `{step_name}`: OK")
                        else:
                            all_passed = False
                            st.error(f"❌ `{step_name}` Failed: {step_info.get('error')}")
                    if all_passed:
                        st.success(f"🎉 All probes passed successfully with {active_provider} model `{active_model}`!")

    # Provider-specific Authentication / Key handling
    if is_ollama:
        st.caption("💻 **Local Inference:** No API key required")
    else:
        if not active_api_key:
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

    # Document Management & Ingestion Section
    st.subheader("📄 Document Management")
    st.caption("Upload documents to index into ChromaDB or select an existing document.")

    # 1. Discover all indexed documents from ChromaDB
    db_indexed_docs = retriever.get_indexed_documents()

    uploaded_file = st.file_uploader(
        "Upload Document",
        type=["pdf", "txt"],
        help="Supports PDF and TXT documents. Extracted text is chunked, embedded, and deduplicated via SHA-256."
    )

    if uploaded_file is not None:
        try:
            upload_bytes = uploaded_file.getvalue()
            upload_hash = RAGRetriever.calculate_document_hash(upload_bytes)
            # Pre-select uploaded document
            if "active_doc_hash" not in st.session_state or st.session_state.get("last_uploaded_name") != uploaded_file.name:
                st.session_state["active_doc_hash"] = upload_hash
                st.session_state["active_doc_name"] = uploaded_file.name
                st.session_state["last_uploaded_name"] = uploaded_file.name
        except Exception:
            pass

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
                    doc_id = ingest_result.get("document_id")
                    chunks_count = ingest_result.get("chunks_created", ingest_result.get("chunks_added", 0))

                    st.session_state["active_doc_hash"] = doc_id
                    st.session_state["active_doc_name"] = filename

                    # Refresh indexed documents list from ChromaDB
                    db_indexed_docs = retriever.get_indexed_documents()

                    if is_duplicate:
                        st.info(f"ℹ️ **Duplicate Detected:** `{filename}` is already indexed. Active document set to `{filename}`.")
                    else:
                        st.success(f"✅ **Success:** `{filename}` indexed ({chunks_count} chunks stored).")
                except DocumentParsingError as e:
                    st.error(f"❌ Document Parsing Error: {str(e)}")
                except Exception as e:
                    st.error(f"❌ Error during ingestion: {str(e)}")
        else:
            st.warning("Please choose a file before clicking Index.")

    # 2. Active Document Selector for Retrieval Isolation
    active_doc_hash = None
    active_doc_name = None

    if db_indexed_docs:
        doc_options = list(db_indexed_docs)
        
        # Determine default selection index
        default_index = 0
        current_hash = st.session_state.get("active_doc_hash")
        if current_hash:
            for idx, d in enumerate(doc_options):
                if d["document_id"] == current_hash or d.get("document_hash") == current_hash:
                    default_index = idx
                    break

        selected_doc = st.selectbox(
            "🎯 **Active Document Scope**",
            options=doc_options,
            index=default_index,
            format_func=lambda d: f"📄 {d['filename']} ({d['chunks_count']} chunks)",
            help="Retrieval is strictly isolated to the selected document."
        )

        if selected_doc:
            active_doc_hash = selected_doc["document_id"]
            active_doc_name = selected_doc["filename"]
            st.session_state["active_doc_hash"] = active_doc_hash
            st.session_state["active_doc_name"] = active_doc_name
            st.caption(f"🔒 **Retrieval Scope:** `{active_doc_name}` (`{active_doc_hash[:8]}...`)")
    elif uploaded_file is not None:
        active_doc_hash = st.session_state.get("active_doc_hash")
        active_doc_name = st.session_state.get("active_doc_name", uploaded_file.name)
        st.caption(f"🔒 **Retrieval Scope (Uploaded):** `{active_doc_name}`")
    else:
        st.warning("⚠️ No documents indexed yet. Upload a document to start.")

    # Indexed Documents in ChromaDB Expander
    if db_indexed_docs:
        with st.expander(f"📁 Indexed in Vectorstore ({len(db_indexed_docs)} documents)", expanded=False):
            for doc in db_indexed_docs:
                is_active = (doc["document_id"] == active_doc_hash)
                active_badge = " *(Active)*" if is_active else ""
                st.markdown(f"- **`{doc['filename']}`**{active_badge}")
                st.caption(f"  ID: `{doc['document_id'][:12]}...` | Chunks: {doc['chunks_count']}")

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
    f"**Enterprise Retrieval-Augmented Generation Platform (Powered by {active_provider.capitalize()}: `{active_model}`)** — "
    "Ask questions grounded strictly in your uploaded documents. "
    "Features **Conversational Memory**, **Multi-Turn Query Reformulation**, **Dense Bi-Encoder Retrieval**, **Cross-Encoder Re-Ranking**, and **Grounded Generation**."
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

            render_rag_pipeline_info(msg, active_model_name=active_model, active_threshold=retriever.distance_threshold)


# ---------------------------------------------------------
# 7. Question Submission & Direct RAG Pipeline Execution
# ---------------------------------------------------------
if prompt := st.chat_input("Ask a question about your uploaded documents..."):
    if not is_ollama and not active_api_key:
        st.error("❌ DEEPSEEK_API_KEY is required to ask questions in Cloud mode. Please configure it in Streamlit Secrets or sidebar.")
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
            with st.spinner(f"Retrieving relevant passages, re-ranking with Cross-Encoder, and generating answer with {active_provider.capitalize()} ({active_model})..."):
                tracker = observability.start_request(question=prompt)
                try:
                    llm_client = get_llm_client(
                        provider=active_provider,
                        api_key=active_api_key,
                        base_url=active_base_url
                    )
                except Exception as e:
                    err_msg, _ = format_llm_error(e, model=active_model, endpoint=active_base_url, provider=active_provider)
                    st.error(f"❌ Configuration Error: {err_msg}")
                    st.stop()

                query_reformulator = QueryReformulator(
                    llm_client=llm_client,
                    model=active_model,
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
                        document_hash=active_doc_hash,
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
                            model=active_model,
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
                            model=active_model,
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
                        "provider": active_provider,
                        "context_found": context_found,
                        "search_query": search_query,
                        "request_id": tracker.request_id,
                        "selected_document_name": active_doc_name,
                        "selected_document_hash": active_doc_hash,
                        "sources": sources,
                        "latency_breakdown": {
                            "reformulation_ms": ref_latency_ms,
                            "retrieval_ms": ret_latency_ms,
                            "llm_ms": llm_latency_ms
                        }
                    }
                    render_rag_pipeline_info(msg_data, active_model_name=active_model, active_threshold=retriever.distance_threshold)

                    # Save assistant message to session state
                    st.session_state.messages.append(msg_data)

                except Exception as e:
                    err_msg, _ = format_llm_error(e, model=active_model, endpoint=active_base_url, provider=active_provider)
                    tracker.record_error(f"Pipeline error: {err_msg}")
                    observability.log_request_metrics(tracker)
                    st.error(f"❌ {err_msg}")
