# Document Q&A RAG Assistant

A production-aware, multi-stage **Retrieval-Augmented Generation (RAG)** application with Conversational Memory, Query Reformulation, Bi-Encoder vector search, Cross-Encoder re-ranking, and telemetry observability.

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

## 🚀 Running Locally

### 1. Prerequisites
- Python 3.10+
- OpenAI API Key

### 2. Setup Virtual Environment
```powershell
# Navigate to project directory
cd C:\Users\nikhi\OneDrive\Desktop\rag-system

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables (`.env`)
Create or edit `.env` in the project root:
```ini
FLASK_PORT=5000
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
OPENAI_API_KEY=your-openai-api-key-here
```

### 4. Start Flask Backend
```powershell
python backend/app.py
```
*Backend runs at `http://localhost:5000` with endpoints `/health`, `/upload`, `/ask`, and `/metrics`.*

### 5. Start Streamlit Frontend (in a separate terminal)
```powershell
streamlit run frontend/app.py
```
*Frontend opens at `http://localhost:8501`.*

---

## ☁️ Deployment on Streamlit Cloud

### Deployment Architecture
Streamlit Cloud runs the frontend UI in a managed cloud container. The frontend communicates with the RAG backend via standard REST APIs (`/health`, `/upload`, `/ask`, `/metrics`).

1. **Deploy Flask Backend (e.g. on Render)**:
   - Create a **New Web Service** connected to your repository on [Render](https://dashboard.render.com/).
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn backend.app:app`
   - **Environment Variables:**
     - `OPENAI_API_KEY` = `your-openai-api-key`
     - `RAG_OBSERVABILITY_ENABLED` = `true`
   - Note the public URL provided by Render (e.g., `https://my-rag-backend.onrender.com`).

2. **Deploy Streamlit Frontend on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io/) and click **New App**.
   - **Repository:** Select your GitHub repository.
   - **Branch:** `main`
   - **Main file path:** `frontend/app.py`
   - Click **Advanced Settings** $\to$ **Secrets** and add:
     ```toml
     BACKEND_URL = "https://my-rag-backend.onrender.com"
     OPENAI_API_KEY = "sk-..."
     ```
   - Click **Deploy!**

---

## 🧪 Automated Testing & Evaluation

### Run Test Suite
```powershell
pytest tests/ -v
```
*Executes all 48 unit, integration, evaluation, and negative test cases.*

### Run Retrieval Evaluation
```powershell
python evaluation/retrieval_evaluator.py
```
*Calculates Recall@1, Recall@3, Recall@5, and MRR across Vector, Threshold, and Cross-Encoder stages and generates reports in `evaluation/results/`.*

---

## ⚙️ Secrets & Configuration Reference

| Secret / Env Variable | Description | Default (Local) |
| :--- | :--- | :--- |
| `BACKEND_URL` | Public or local base URL of Flask backend | `http://localhost:5000` |
| `OPENAI_API_KEY` | OpenAI API Key for reformulation and generation | *(Required for LLM)* |
| `RAG_CONVERSATION_TURNS` | Number of recent conversation turns to retain | `5` |
| `RAG_INITIAL_RETRIEVAL_K` | Initial candidates retrieved from ChromaDB | `8` |
| `RAG_DISTANCE_THRESHOLD` | Cosine distance cutoff threshold | `0.6` |
| `RAG_RERANKER_MODEL` | Hugging Face Cross-Encoder model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `RAG_FINAL_CONTEXT_K` | Top-$k$ chunks passed to the LLM | `5` |
| `RAG_OBSERVABILITY_ENABLED` | Enable local JSONL telemetry logging | `true` |
| `RAG_METRICS_PATH` | File path for request metrics | `./logs/rag_metrics.jsonl` |
