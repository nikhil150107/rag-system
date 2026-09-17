# Document Q&A RAG Assistant

A multi-stage **Retrieval-Augmented Generation (RAG)** application with Conversational Memory, Query Reformulation, Bi-Encoder vector search, Cross-Encoder re-ranking, grounded generation, and telemetry observability.

---

## 🏛️ System Architecture

```
User Question + Conversation History
               ↓
1. Multi-Turn Query Reformulator (gpt-4o-mini)
               ↓ [Standalone Search Query]
2. Dense Bi-Encoder Retrieval (all-MiniLM-L6-v2 in ChromaDB Top-8)
               ↓ [Top-8 Candidates]
3. Cosine Distance Threshold Filtering (<= 0.6)
               ↓ [Surviving Candidates]
4. Cross-Encoder Re-Ranking (ms-marco-MiniLM-L-6-v2 Top-5)
               ↓ [Top-5 Relevant Chunks]
5. Grounded LLM Generation (gpt-4o-mini with Document Context)
               ↓
Answer + Sources (Vector Distance & Reranker Score) + Telemetry
```

---

## ☁️ Deployment on Streamlit Cloud (Standalone Architecture)

The application is engineered to run as a **standalone, self-contained Streamlit application** on **Streamlit Cloud**. The RAG pipeline (`backend/rag`) is directly executed in-process with singleton model caching (`@st.cache_resource`), completely eliminating the need for a separate Flask backend and running comfortably within Streamlit Cloud's 1 GB RAM allowance.

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
   - Add your OpenAI API key in TOML format:
     ```toml
     OPENAI_API_KEY = "sk-..."
     ```
   - *(Optional environment overrides can also be added here if desired)*:
     ```toml
     RAG_DISTANCE_THRESHOLD = "0.6"
     RAG_INITIAL_RETRIEVAL_K = "8"
     RAG_FINAL_CONTEXT_K = "5"
     ```

4. **Click Deploy!**:
   - Streamlit Cloud will install `requirements.txt`, load and cache the transformer models in-process, and launch the web UI.

---

## 🚀 Running Locally

### 1. Setup Virtual Environment
```powershell
# Navigate to project directory
cd C:\Users\nikhi\OneDrive\Desktop\rag-system

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure `.env`
Create `.env` in the root folder with:
```ini
OPENAI_API_KEY=your-openai-api-key-here
VECTOR_DB_PATH=./vectorstore
EMBEDDING_MODEL=all-MiniLM-L6-v2
RAG_DISTANCE_THRESHOLD=0.6
RAG_INITIAL_RETRIEVAL_K=8
RAG_FINAL_CONTEXT_K=5
RAG_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RAG_CONVERSATION_TURNS=5
RAG_OBSERVABILITY_ENABLED=true
RAG_METRICS_PATH=./logs/rag_metrics.jsonl
RAG_LOG_QUESTIONS=false
```

### 3. Start Streamlit App
```powershell
streamlit run frontend/app.py
```
*Opens automatically at `http://localhost:8501`.*

*(Optional) If you also want to run the REST API backend for automated testing or external API consumers:*
```powershell
python backend/app.py
```

---

## 🧪 Automated Testing & Evaluation

### Run Test Suite
```powershell
pytest tests/ -v
```
*Executes all 48 unit, integration, evaluation, and hallucination test cases.*

### Run Retrieval Evaluation
```powershell
python evaluation/retrieval_evaluator.py
```
*Calculates Recall@1, Recall@3, Recall@5, and MRR across Vector, Threshold, and Cross-Encoder stages.*

---

## ⚙️ Configuration Reference

| Variable / Secret | Description | Default |
| :--- | :--- | :--- |
| `OPENAI_API_KEY` | OpenAI API Key for reformulation and generation | *(Required)* |
| `EMBEDDING_MODEL` | Bi-Encoder sentence-transformers model | `all-MiniLM-L6-v2` |
| `RAG_RERANKER_MODEL` | Hugging Face Cross-Encoder model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `RAG_DISTANCE_THRESHOLD` | Cosine distance cutoff threshold | `0.6` |
| `RAG_INITIAL_RETRIEVAL_K` | Initial candidates retrieved from ChromaDB | `8` |
| `RAG_FINAL_CONTEXT_K` | Top-$k$ chunks passed to the LLM | `5` |
| `RAG_CONVERSATION_TURNS` | Number of recent conversation turns to retain | `5` |
| `VECTOR_DB_PATH` | Local directory for ChromaDB vector storage | `./vectorstore` |
| `RAG_OBSERVABILITY_ENABLED` | Enable local JSONL telemetry logging | `true` |
| `RAG_METRICS_PATH` | File path for request metrics | `./logs/rag_metrics.jsonl` |
