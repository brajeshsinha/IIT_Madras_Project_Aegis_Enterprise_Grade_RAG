"""
app/main.py

FastAPI application exposing the Project Aegis RAG pipeline as REST endpoints.

Endpoints:
  GET  /health                  — liveness + collection stats
  POST /query                   — full RAG query (expansion + HyDE + rerank)
  POST /ingest                  — upload and ingest a policy document
  DELETE /documents/{doc_id}    — remove all chunks for a document

Run locally:
    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

from app.generator import generate_answer
from configs.settings import get_settings
from ingestion.chunker import SemanticChunker
from ingestion.metadata_extractor import extract_metadata
from ingestion.parser import parse_document
from retrieval.embedder import embed_documents
from retrieval.pipeline import retrieve
from vector_db.qdrant_client import (
    delete_by_document_id,
    ensure_collection,
    get_collection_info,
    upsert_chunks,
)

settings = get_settings()

app = FastAPI(
    title="Project Aegis – Enterprise Policy RAG",
    description="Advanced RAG system for corporate policy intelligence.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup() -> None:
    logger.info("Project Aegis API starting...")
    ensure_collection()
    logger.success("Qdrant collection ready")


# ── Schemas ──────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query:           str
    use_expansion:   bool = True
    use_hyde:        bool = True
    filter_category: bool = True
    top_k_retrieval: int  = 25
    top_k_rerank:    int  = 5


class QueryResponse(BaseModel):
    query:                str
    answer:               str
    sources:              list[dict]
    model:                str
    chunks_retrieved:     int
    chunks_after_rerank:  int


class IngestResponse(BaseModel):
    document_id:     str
    chunks_ingested: int
    message:         str


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", summary="Health check")
async def health() -> dict:
    """Returns API status and Qdrant collection stats."""
    return {
        "status":     "ok",
        "collection": get_collection_info(),
    }


@app.post("/query", response_model=QueryResponse, summary="Query policy documents")
async def query_endpoint(req: QueryRequest) -> QueryResponse:
    """
    Run the full Project Aegis retrieval pipeline and return a
    grounded answer with source attribution.

    The pipeline runs:
      1. Query expansion (multi-query)
      2. HyDE embedding
      3. Metadata pre-filter (category routing)
      4. Vector retrieval (top 25 candidates)
      5. Cross-encoder reranking (top 5)
      6. LLM answer generation
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        chunks = retrieve(
            query=req.query,
            use_expansion=req.use_expansion,
            use_hyde=req.use_hyde,
            filter_category=req.filter_category,
            top_k_retrieval=req.top_k_retrieval,
            top_k_rerank=req.top_k_rerank,
        )
        result = generate_answer(query=req.query, chunks=chunks)

        return QueryResponse(
            query=req.query,
            answer=result["answer"],
            sources=result["sources"],
            model=result["model"],
            chunks_retrieved=req.top_k_retrieval,
            chunks_after_rerank=len(chunks),
        )

    except Exception as exc:
        logger.exception(f"Query failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ingest", response_model=IngestResponse, summary="Ingest a policy document")
async def ingest_endpoint(
    file:             UploadFile = File(...),
    policy_category:  str = Form(default=""),
    policy_owner:     str = Form(default=""),
    effective_date:   str = Form(default=""),
    document_version: str = Form(default="V1"),
) -> IngestResponse:
    """
    Upload a policy document and ingest it into the vector database.
    Supported formats: PDF, DOCX, Markdown, HTML.

    If a document with the same derived document_id already exists,
    all its chunks are deleted before re-ingestion.
    """
    suffix = Path(file.filename or "doc.pdf").suffix.lower()
    if suffix not in {".pdf", ".docx", ".md", ".html", ".htm"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix!r}",
        )

    tmp = Path(f"/tmp/aegis_{file.filename}")
    try:
        tmp.write_bytes(await file.read())

        # 1. Parse
        markdown = parse_document(tmp)

        # 2. Chunk
        chunker = SemanticChunker()
        chunks  = chunker.chunk(markdown)
        if not chunks:
            raise HTTPException(status_code=422, detail="Document produced no chunks")

        # 3. Metadata
        enriched = [
            {
                "chunk": c,
                "meta":  extract_metadata(
                    chunk=c,
                    source_path=tmp,
                    policy_category=policy_category,
                    policy_owner=policy_owner,
                    effective_date=effective_date,
                    document_version=document_version,
                ),
            }
            for c in chunks
        ]

        doc_id = enriched[0]["meta"]["document_id"]

        # 4. Delete old version
        delete_by_document_id(doc_id)

        # 5. Embed
        texts   = [e["chunk"].text for e in enriched]
        vectors = embed_documents(texts)

        # 6. Upsert
        records = [
            {
                "chunk_id":  e["meta"]["chunk_id"],
                "embedding": vec,
                "metadata":  e["meta"],
                "text":      e["chunk"].text,
            }
            for e, vec in zip(enriched, vectors)
        ]
        upsert_chunks(records)

        return IngestResponse(
            document_id=doc_id,
            chunks_ingested=len(records),
            message=f"Successfully ingested '{file.filename}' as {doc_id}",
        )

    finally:
        if tmp.exists():
            tmp.unlink()


@app.delete("/documents/{document_id}", summary="Delete a document")
async def delete_document(document_id: str) -> dict:
    """Remove all Qdrant chunks belonging to a given document_id."""
    delete_by_document_id(document_id)
    return {"message": f"Document '{document_id}' deleted from the index"}
