"""
vector_db/qdrant_client.py

Thin, typed wrapper around the Qdrant Python SDK.

Responsibilities:
  • Collection creation with payload indexes for fast metadata filtering
  • Batch upsert with automatic chunking into 200-point batches
  • Filtered approximate nearest-neighbour search
  • Document-level deletion (used before re-ingestion)

All other modules import from here — the Qdrant SDK is never accessed directly.

Usage::
    from vector_db.qdrant_client import ensure_collection, upsert_chunks, search

    ensure_collection()

    upsert_chunks([
        {
            "chunk_id":  "uuid-...",
            "embedding": [0.1, 0.2, ...],   # 1024-dim float list
            "metadata":  {...},
            "text":      "Policy clause text...",
        }
    ])

    results = search(
        query_vector=[0.1, 0.2, ...],
        top_k=25,
        filters={"policy_category": "Travel"},
    )
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from configs.settings import get_settings

settings = get_settings()

# BAAI/bge-large-en-v1.5 produces 1024-dim vectors
VECTOR_DIM = 1024

# Payload fields to index for fast metadata filtering
_INDEXED_FIELDS = (
    "policy_category",
    "policy_owner",
    "effective_date",
    "document_id",
    "document_version",
)


# ── Client singleton ────────────────────────────────────────────────────────

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        logger.info(
            f"Qdrant client connected: "
            f"{settings.qdrant_host}:{settings.qdrant_port}"
        )
    return _client


# ── Collection management ───────────────────────────────────────────────────

def ensure_collection(
    collection_name: str = settings.qdrant_collection,
) -> None:
    """
    Create the collection if it does not already exist, then create
    keyword payload indexes on all metadata fields used for filtering.
    Safe to call multiple times — idempotent.
    """
    client   = get_client()
    existing = {c.name for c in client.get_collections().collections}

    if collection_name in existing:
        logger.info(f"Collection '{collection_name}' already exists")
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=qm.VectorParams(
            size=VECTOR_DIM,
            distance=qm.Distance.COSINE,
        ),
    )

    for field in _INDEXED_FIELDS:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field,
            field_schema=qm.PayloadSchemaType.KEYWORD,
        )

    logger.success(
        f"Collection '{collection_name}' created "
        f"with indexes on {_INDEXED_FIELDS}"
    )


# ── Upsert ──────────────────────────────────────────────────────────────────

def upsert_chunks(
    records:         list[dict[str, Any]],
    collection_name: str = settings.qdrant_collection,
) -> int:
    """
    Upsert a list of chunk records into Qdrant.

    Each record must contain:
        chunk_id  : str           — used as the Qdrant point ID
        embedding : list[float]   — 1024-dim normalised vector
        metadata  : dict          — stored as payload (filterable)
        text      : str           — stored in payload as "chunk_text"

    Returns the total number of points upserted.

    Example::
        upsert_chunks([
            {
                "chunk_id":  "abc-123",
                "embedding": [0.01, ...],
                "metadata":  {"policy_category": "Travel", ...},
                "text":      "Employees may claim taxi expenses...",
            }
        ])
    """
    client = get_client()
    points = []

    for rec in records:
        payload = dict(rec["metadata"])
        payload["chunk_text"] = rec["text"]   # retrieve text alongside metadata
        points.append(
            qm.PointStruct(
                id=rec["chunk_id"],
                vector=rec["embedding"],
                payload=payload,
            )
        )

    if not points:
        logger.warning("upsert_chunks: nothing to upsert")
        return 0

    total = 0
    batch_size = 200
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=collection_name, points=batch)
        total += len(batch)
        logger.debug(
            f"Upserted batch {i // batch_size + 1}: {len(batch)} points"
        )

    logger.success(f"Upserted {total} points into '{collection_name}'")
    return total


# ── Search ──────────────────────────────────────────────────────────────────

def search(
    query_vector:    list[float],
    top_k:           int = settings.retrieval_top_k,
    filters:         dict[str, str] | None = None,
    collection_name: str = settings.qdrant_collection,
) -> list[dict[str, Any]]:
    """
    Approximate nearest-neighbour search with optional metadata filtering.

    Args:
        query_vector: Normalised 1024-dim float vector.
        top_k:        Number of candidates to return.
        filters:      Optional dict of exact-match metadata filters, e.g.
                      {"policy_category": "HR"}. All conditions are AND-ed.
        collection_name: Target Qdrant collection.

    Returns:
        List of dicts::

            {
                "score":      float,   # cosine similarity (higher = closer)
                "chunk_text": str,
                "metadata":   dict,    # everything else from the payload
            }

    Example::
        results = search(
            query_vector=embed_query("Can I claim Uber?"),
            top_k=25,
            filters={"policy_category": "Travel"},
        )
    """
    client = get_client()

    qdrant_filter = None
    if filters:
        qdrant_filter = qm.Filter(must=[
            qm.FieldCondition(key=k, match=qm.MatchValue(value=v))
            for k, v in filters.items()
        ])

    hits = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k,
        query_filter=qdrant_filter,
        with_payload=True,
    )

    results = []
    for hit in hits:
        payload    = dict(hit.payload or {})
        chunk_text = payload.pop("chunk_text", "")
        results.append({
            "score":      hit.score,
            "chunk_text": chunk_text,
            "metadata":   payload,
        })

    logger.debug(
        f"Search: {len(results)} hits  "
        f"(top_k={top_k}, filters={filters})"
    )
    return results


# ── Deletion ────────────────────────────────────────────────────────────────

def delete_by_document_id(
    document_id:     str,
    collection_name: str = settings.qdrant_collection,
) -> None:
    """
    Delete all Qdrant points belonging to a given document_id.
    Called before re-ingesting an updated version of the same document.

    Example::
        delete_by_document_id("TRV-POL-2026-V2")
    """
    client = get_client()
    client.delete(
        collection_name=collection_name,
        points_selector=qm.FilterSelector(
            filter=qm.Filter(must=[
                qm.FieldCondition(
                    key="document_id",
                    match=qm.MatchValue(value=document_id),
                )
            ])
        ),
    )
    logger.info(f"Deleted all chunks for document_id='{document_id}'")


def get_collection_info(
    collection_name: str = settings.qdrant_collection,
) -> dict:
    """Return basic stats about the collection."""
    client = get_client()
    info   = client.get_collection(collection_name)
    return {
        "name":         collection_name,
        "points_count": info.points_count,
        "status":       str(info.status),
    }
