# Document Q&A RAG Assistant (Ollama & DeepSeek)

A multi-stage **Retrieval-Augmented Generation (RAG)** application with Conversational Memory, Query Reformulation, Bi-Encoder vector search, Cross-Encoder re-ranking, grounded generation via **Local Ollama** (`llama3.2:3b`) or **DeepSeek Cloud** (`deepseek-chat`), and telemetry observability.

---

## 🏛️ System Architecture

```
User Question + Conversation History
               ↓
1. Multi-Turn Query Reformulator (Ollama llama3.2:3b / DeepSeek deepseek-chat)
               ↓ [Standalone Search Query]
2. Dense Bi-Encoder Retrieval (all-MiniLM-L6-v2 in ChromaDB Top-8)
               ↓ [Top-8 Candidates]
3. Cosine Distance Threshold Filtering (<= 0.6)
               ↓ [Surviving Candidates]
4. Cross-Encoder Re-Ranking (ms-marco-MiniLM-L-6-v2 Top-5)
               ↓ [Top-5 Relevant Chunks]
5. Grounded LLM Generation (Local Ollama / DeepSeek with Document Context)
               ↓
Answer + Sources (Vector Distance & Reranker Score) + Telemetry
```

---

## 💻 Local Execution with Ollama (Zero Cost, Fully Private)

### 1. Install & Pull Model in Ollama
```powershell
# Ensure Ollama is running
ollama --version

# Pull the lightweight 3B model
ollama pull llama3.2:3b
```

### 2. Configure `.env`
```ini
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
```

### 3. Start Streamlit App
```powershell
streamlit run frontend/app.py
```
*Opens automatically at `http://localhost:8501`. Switch between Ollama (Local) and DeepSeek (Cloud) directly in the sidebar.*

---

## ☁️ Deployment on Streamlit Cloud (Standalone Architecture)

The application runs as a **standalone, self-contained Streamlit application** on **Streamlit Cloud**. The RAG pipeline (`backend/rag`) is directly executed in-process with singleton model caching (`@st.cache_resource`), completely eliminating the need for a separate Flask backend and running comfortably within Streamlit Cloud's 1 GB RAM allowance.

### Step-by-Step Deployment Instructions

1. **Push Code to GitHub**:
   - Repository: `https://github.com/nikhil150107/rag-system`
   - Main Branch: `main`

2. **Deploy on Streamlit Community Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io/) and click **New App**.
   - **Repository:** `nikhil150107/rag-system`
   - **Branch:** `main`
   - **Main file path:** `frontend/app.py`

3. **Configure Secrets**:
   - Click **Advanced Settings** $\to$ **Secrets**.
   - Add your DeepSeek API key and configuration in TOML format for Cloud deployment:
     ```toml
     DEEPSEEK_API_KEY = "your-deepseek-api-key"
     LLM_PROVIDER = "deepseek"
     DEEPSEEK_BASE_URL = "https://api.deepseek.com"
     DEEPSEEK_MODEL = "deepseek-chat"
     ```
   - *(Optional pipeline parameter overrides can also be added here if desired)*:
     ```toml
     RAG_DISTANCE_THRESHOLD = "0.6"
     RAG_INITIAL_RETRIEVAL_K = "8"
     RAG_FINAL_CONTEXT_K = "5"
     ```

4. **Click Deploy!**:
   - Streamlit Cloud will install dependencies, load and cache the transformer models in-process, and launch the web UI.

---

## 🧪 Automated Testing & Evaluation

### Run Test Suite
```powershell
pytest tests/ -v
```
*Executes all unit, integration, evaluation, and hallucination test cases using offline mocks (no live LLM / network calls required).*

---

## ⚙️ Configuration Reference

| Variable / Secret | Description | Default |
| :--- | :--- | :--- |
| `LLM_PROVIDER` | LLM Provider identifier (`ollama` or `deepseek`) | `ollama` |
| `OLLAMA_BASE_URL` | Local Ollama base URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | Local Ollama model identifier | `llama3.2:3b` |
| `DEEPSEEK_API_KEY` | DeepSeek API Key (Required for Cloud mode) | *(Optional in local mode)* |
| `DEEPSEEK_BASE_URL` | DeepSeek OpenAI-compatible API base URL | `https://api.deepseek.com` |
| `DEEPSEEK_MODEL` | DeepSeek LLM model name | `deepseek-chat` |
| `EMBEDDING_MODEL` | Bi-Encoder sentence-transformers model | `all-MiniLM-L6-v2` |
| `RAG_RERANKER_MODEL` | Hugging Face Cross-Encoder model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `RAG_DISTANCE_THRESHOLD` | Cosine distance cutoff threshold | `0.6` |
| `RAG_INITIAL_RETRIEVAL_K` | Initial candidates retrieved from ChromaDB | `8` |
| `RAG_FINAL_CONTEXT_K` | Top-$k$ chunks passed to the LLM | `5` |
| `RAG_CONVERSATION_TURNS` | Number of recent conversation turns to retain | `5` |
| `VECTOR_DB_PATH` | Local directory for ChromaDB vector storage | `./vectorstore` |
| `RAG_OBSERVABILITY_ENABLED` | Enable local JSONL telemetry logging | `true` |
| `RAG_METRICS_PATH` | File path for request metrics | `./logs/rag_metrics.jsonl` |
