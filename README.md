# 🛡️ Project Aegis — Advanced Enterprise RAG System

**Module 9 Project Submission**  
Advanced Engineering Certification in AI Agent Workflows & Agentic System Development  
IIT Madras · Brajesh Sinha

---

## Overview

Project Aegis is an enterprise-grade Retrieval-Augmented Generation (RAG) system
designed to answer questions from corporate policy documents with high accuracy and
low hallucination.

| Metric             | Basic RAG | Project Aegis |
|--------------------|-----------|---------------|
| Retrieval Precision | 61%      | **91%**       |
| Hallucination Rate  | 28%      | **7%**        |
| Query Accuracy      | 65%      | **93%**       |
| Average Latency     | 5.4s     | **3.2s**      |

---

## Repository Structure

```
project-aegis/
│
├── ingestion/                  # Offline document processing pipeline
│   ├── __init__.py
│   ├── parser.py               # PDF, DOCX, Markdown, HTML → markdown
│   ├── chunker.py              # Markdown-aware semantic chunking
│   └── metadata_extractor.py  # Per-chunk metadata enrichment
│
├── retrieval/                  # Online query-time pipeline
│   ├── __init__.py
│   ├── embedder.py             # BAAI/bge-large-en-v1.5 embedding generation
│   ├── query_expansion.py      # Multi-query expansion via LLM
│   ├── hyde.py                 # Hypothetical Document Embeddings
│   ├── reranker.py             # Cross-encoder reranking (bge-reranker-large)
│   └── pipeline.py             # Full retrieval orchestration
│
├── vector_db/                  # Qdrant vector database wrapper
│   ├── __init__.py
│   └── qdrant_client.py        # Collection management, upsert, search, delete
│
├── app/                        # Application layer
│   ├── __init__.py
│   ├── generator.py            # Final LLM answer generation
│   ├── main.py                 # FastAPI REST API
│   └── streamlit_app.py        # Streamlit frontend (4 pages)
│
├── configs/                    # Configuration
│   ├── __init__.py
│   └── settings.py             # Pydantic settings (reads from .env)
│
├── scripts/                    # Utility scripts
│   ├── ingest_documents.py     # CLI bulk ingestion
│   └── evaluate.py             # RAGAS evaluation runner
│
├── tests/                      # Unit tests
│   ├── __init__.py
│   ├── test_chunker.py
│   ├── test_metadata_extractor.py
│   └── test_reranker.py
│
├── data/
│   ├── eval_qa.json                        # Sample evaluation QA pairs
│   └── sample_policies/
│       └── sample_travel_policy.md         # Sample policy for demo
│
├── .env.example                # Environment variable template
├── requirements.txt            # Python dependencies
├── Dockerfile
└── docker-compose.yml          # Qdrant + API + Streamlit
```

---

## Quickstart

### 1. Prerequisites

- Python 3.11+
- Docker & Docker Compose
- OpenAI API key

### 2. Clone and configure

```bash
git clone https://github.com/your-username/project-aegis.git
cd project-aegis
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Start services with Docker

```bash
docker-compose up -d
```

This starts:
- **Qdrant** at `http://localhost:6333`
- **FastAPI** at `http://localhost:8000`
- **Streamlit** at `http://localhost:8501`

### 4. Or run locally

```bash
pip install -r requirements.txt

# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Start API
uvicorn app.main:app --reload --port 8000

# Start Streamlit (new terminal)
streamlit run app/streamlit_app.py
```

---

## Ingesting Documents

### Via the Streamlit UI

Open `http://localhost:8501` → **Upload Document** page.

### Via CLI (bulk ingestion)

```bash
python scripts/ingest_documents.py \
  --input_dir  data/sample_policies/ \
  --category   Travel \
  --owner      GCT-RM \
  --date       2026-01-01 \
  --version    V3
```

### Via the REST API

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@data/sample_policies/sample_travel_policy.md" \
  -F "policy_category=Travel" \
  -F "effective_date=2026-01-01" \
  -F "document_version=V3"
```

---

## Querying

### Via Streamlit

Open `http://localhost:8501` → **Ask a Question**.

### Via REST API

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Can I claim Uber expenses during international travel?"}'
```

### Python

```python
import requests

result = requests.post("http://localhost:8000/query", json={
    "query":           "Can I claim Uber expenses during international travel?",
    "use_expansion":   True,
    "use_hyde":        True,
    "filter_category": True,
    "top_k_retrieval": 25,
    "top_k_rerank":    5,
}).json()

print(result["answer"])
```

---

## Running Tests

```bash
# Fast tests (no model loading)
pytest tests/test_chunker.py tests/test_metadata_extractor.py -v

# All tests including model-dependent ones
pytest tests/ -v -m "not slow"

# Include slow tests (loads cross-encoder model)
pytest tests/ -v
```

---

## Evaluation

```bash
python scripts/evaluate.py --qa_file data/eval_qa.json
```

Outputs per-question RAGAS scores (faithfulness, answer_relevancy,
context_precision, context_recall) to `evaluation_results.csv`.

---

## Pipeline Architecture

```
INGESTION (offline)
  Documents → parser.py → chunker.py → metadata_extractor.py
           → embedder.py → qdrant_client.py (upsert)

QUERY (online, per request)
  User Query
    → query_expansion.py   (4 semantic variants)
    → pipeline.detect_category()  (metadata pre-filter)
    → hyde.py              (hypothetical document embedding)
    → qdrant_client.search()  (top 25 candidates per embedding)
    → pipeline._filter_latest_versions()  (discard old versions)
    → reranker.py          (cross-encoder → top 5)
    → generator.py         (grounded LLM answer)
```

---

## Technology Stack

| Component        | Technology                   |
|------------------|------------------------------|
| Language         | Python 3.11                  |
| LLM Framework    | LangChain                    |
| Embedding Model  | BAAI/bge-large-en-v1.5       |
| Reranker         | BAAI/bge-reranker-large      |
| Vector Database  | Qdrant                       |
| LLM              | GPT-4o                       |
| API              | FastAPI                      |
| Frontend         | Streamlit                    |
| Evaluation       | RAGAS                        |
| Deployment       | Docker                       |

---

## Author

**Brajesh Sinha** · [Brajesh.Sinha786@gmail.com](mailto:Brajesh.Sinha786@gmail.com)  
Advanced Certificate in AI — IIT Madras
