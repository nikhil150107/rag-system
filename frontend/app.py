"""Document Q&A RAG Assistant - Streamlit Frontend."""
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import requests
import streamlit as st

# ---------------------------------------------------------
# 1. Environment & Configuration
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
ENV_PATH = ROOT_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()

FLASK_PORT = int(os.getenv("FLASK_PORT", 5000))

# Dynamic resolution of BACKEND_URL (Streamlit Cloud Secrets -> Environment Variable -> Localhost)
def resolve_backend_url() -> str:
    # 1. Streamlit Secrets (for Streamlit Cloud deployment)
    try:
        if "BACKEND_URL" in st.secrets:
            return st.secrets["BACKEND_URL"].rstrip("/")
    except Exception:
        pass

    # 2. Environment Variable
    env_url = os.getenv("BACKEND_URL")
    if env_url:
        return env_url.rstrip("/")

    # 3. Localhost fallback
    return f"http://localhost:{FLASK_PORT}"

BACKEND_URL = resolve_backend_url()

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
# 3. Helper Functions: API Calls & Rendering
# ---------------------------------------------------------
def check_backend_health() -> Optional[Dict[str, Any]]:
    """Check connection to the Flask RAG backend and retrieve component status."""
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=3)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def get_backend_metrics() -> Optional[Dict[str, Any]]:
    """Fetch aggregated operational telemetry metrics from backend."""
    try:
        resp = requests.get(f"{BACKEND_URL}/metrics", timeout=3)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


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
                rerank_val = source.get("reranker_score")
                rerank_str = f"{rerank_val:.4f}" if isinstance(rerank_val, (int, float)) else "N/A"
                snippet = source.get("snippet", "")

                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Source {i}:** `{filename}` | **{page_str}** | **Chunk:** `#{chunk_idx}`")
                with col2:
                    st.caption(f"Dist: `{dist}` | Rerank: `{rerank_str}`")

                if snippet:
                    st.info(f"“{snippet}”")
            else:
                st.markdown(f"**Chunk {i}:**")
                st.markdown(f"> {source}")

            if i < len(sources):
                st.divider()


def render_rag_pipeline_info(msg: Dict[str, Any]):
    """Render RAG pipeline stages and execution diagnostics."""
    req_id = msg.get("request_id")
    search_query = msg.get("search_query")
    context_found = msg.get("context_found")
    sources = msg.get("sources", [])

    with st.expander("🔧 RAG Pipeline & Telemetry", expanded=False):
        st.markdown("**Architecture Execution Flow:**")
        st.code(
            "User Question + History\n"
            "  ↳ 1. Multi-Turn Query Reformulator (gpt-4o-mini)\n"
            "  ↳ 2. Dense Vector Retrieval (all-MiniLM-L6-v2 in ChromaDB Top-8)\n"
            "  ↳ 3. Cosine Distance Threshold Filter (<= 0.6)\n"
            "  ↳ 4. Cross-Encoder Re-Ranking (ms-marco-MiniLM-L-6-v2 Top-5)\n"
            "  ↳ 5. Grounded Context-Bound Generation (gpt-4o-mini)",
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


# ---------------------------------------------------------
# 4. Sidebar: Connectivity, Document Management & Controls
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚙️ RAG Controls")

    # Backend Connection Status Card
    health_data = check_backend_health()
    if health_data and health_data.get("status") == "ok":
        st.success("🟢 **Backend Connected**")
        components = health_data.get("components", {})
        with st.expander("System Components Health", expanded=False):
            st.markdown(f"- **Vectorstore:** `{components.get('vectorstore', 'ok')}`")
            st.markdown(f"- **Embedding Model:** `{components.get('embedding_model', 'ok')}`")
            st.markdown(f"- **Cross-Encoder:** `{components.get('reranker', 'ok')}`")
    else:
        st.error("🔴 **Backend Disconnected**")
        st.warning(
            f"Could not reach backend at `{BACKEND_URL}`.\n\n"
            "• **Local:** Ensure Flask is running (`python backend/app.py`).\n"
            "• **Cloud:** Set `BACKEND_URL` in Streamlit Secrets."
        )

    st.divider()

    # Document Upload Section
    st.subheader("📄 Document Ingestion")
    st.caption("Upload PDF or TXT documents to index into ChromaDB vector database.")

    uploaded_file = st.file_uploader(
        "Select Document",
        type=["pdf", "txt"],
        help="Supports PDF and TXT documents. Extracted text is chunked, embedded, and deduplicated via SHA-256."
    )

    if st.button("🚀 Index Document", use_container_width=True, type="primary"):
        if uploaded_file is not None:
            with st.spinner("Processing, parsing pages, and embedding into ChromaDB..."):
                try:
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type or "application/octet-stream"
                        )
                    }
                    response = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=60)
                    if response.status_code == 200:
                        data = response.json()
                        is_duplicate = data.get("duplicate", False)
                        chunks_count = data.get("chunks_created", data.get("chunks_added", 0))

                        # Record in session state
                        doc_info = {
                            "name": uploaded_file.name,
                            "size_kb": round(len(uploaded_file.getvalue()) / 1024, 1),
                            "chunks": chunks_count,
                            "duplicate": is_duplicate,
                            "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
                        }
                        # Add if not already listed
                        if not any(d["name"] == uploaded_file.name for d in st.session_state.indexed_documents):
                            st.session_state.indexed_documents.append(doc_info)

                        if is_duplicate:
                            st.info(f"ℹ️ **Duplicate Detected:** `{uploaded_file.name}` is already indexed. No duplicate chunks added.")
                        else:
                            st.success(f"✅ **Success:** `{uploaded_file.name}` indexed ({chunks_count} chunks stored).")
                    else:
                        error_msg = response.json().get("error", "Upload failed")
                        st.error(f"❌ Upload failed: {error_msg}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Connection error: Backend server is unreachable.")
                except Exception as e:
                    st.error(f"❌ Error during upload: {str(e)}")
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
    metrics_data = get_backend_metrics()
    if metrics_data:
        with st.expander("📊 Operational Telemetry", expanded=False):
            st.metric("Total Requests", metrics_data.get("total_requests", 0))
            st.metric("Context-Found Rate", f"{metrics_data.get('context_found_rate', 0.0):.1%}")
            st.metric("Avg Latency", f"{metrics_data.get('avg_total_latency_ms', 0.0)} ms")
            st.metric("p95 Latency", f"{metrics_data.get('p95_total_latency_ms', 0.0)} ms")

    # Clear Conversation
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------
# 5. Main Area: Header & Chat Interface
# ---------------------------------------------------------
st.title("📚 Document Q&A RAG Assistant")
st.markdown(
    "**Enterprise Retrieval-Augmented Generation Platform** — Ask questions grounded strictly in your uploaded documents. "
    "Features **Conversational Memory**, **Multi-Turn Query Reformulation**, **Dense Bi-Encoder Retrieval**, **Cross-Encoder Re-Ranking**, and **Grounded LLM Generation**."
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

            render_pipeline_info(msg)


# ---------------------------------------------------------
# 6. Question Submission & Streaming/Response Handling
# ---------------------------------------------------------
if prompt := st.chat_input("Ask a question about your uploaded documents..."):
    # Build prior conversation history (excluding the current user question)
    prior_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
        if m.get("role") in ("user", "assistant")
    ]

    # Append user question to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call backend API
    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant passages, re-ranking with Cross-Encoder, and generating answer..."):
            try:
                start_req_time = time.perf_counter()
                response = requests.post(
                    f"{BACKEND_URL}/ask",
                    json={
                        "question": prompt,
                        "conversation_history": prior_history
                    },
                    timeout=60
                )
                req_latency = (time.perf_counter() - start_req_time) * 1000.0

                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "")
                    context_found = data.get("context_found", False)
                    sources = data.get("sources", [])
                    search_query = data.get("search_query")
                    req_id = data.get("request_id")

                    st.markdown(answer)

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
                        "content": answer,
                        "context_found": context_found,
                        "search_query": search_query,
                        "request_id": req_id,
                        "sources": sources
                    }
                    render_pipeline_info(msg_data)

                    # Save assistant message to state
                    st.session_state.messages.append(msg_data)
                else:
                    try:
                        err_data = response.json()
                        err_msg = err_data.get("details") or err_data.get("error") or "Unknown server error"
                    except Exception:
                        err_msg = f"Server returned HTTP {response.status_code}"
                    st.error(f"❌ Backend Error: {err_msg}")
            except requests.exceptions.ConnectionError:
                st.error(
                    f"❌ **Connection Error:** Could not connect to backend server at `{BACKEND_URL}`.\n\n"
                    "Please verify that the Flask backend is active."
                )
            except requests.exceptions.Timeout:
                st.error("⏱️ **Request Timeout:** The backend took longer than 60 seconds to respond.")
            except Exception as e:
                st.error(f"❌ **Request Failed:** {str(e)}")
