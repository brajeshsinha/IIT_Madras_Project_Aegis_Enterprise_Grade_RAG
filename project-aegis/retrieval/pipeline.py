"""
retrieval/pipeline.py

Orchestrates the complete query-time retrieval pipeline:

  Step 1  Query expansion       → generate 4 semantic variants
  Step 2  Category detection    → infer metadata filter from query
  Step 3  HyDE embedding        → embed a hypothetical policy excerpt
  Step 4  Vector retrieval      → search Qdrant with all embeddings
  Step 5  Deduplication         → merge results across query variants
  Step 6  Version post-filter   → discard superseded document versions
  Step 7  Cross-encoder rerank  → score each candidate against the query
  Step 8  Return top-K          → pass to final LLM for answer generation

Usage::
    from retrieval.pipeline import retrieve

    chunks = retrieve("Can I claim Uber expenses during international travel?")
    # Returns list of top-5 reranked chunk dicts ready for answer generation
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date

from loguru import logger

from configs.settings import get_settings
from retrieval.embedder import embed_query
from retrieval.hyde import generate_hyde_embedding
from retrieval.query_expansion import expand_query
from retrieval.reranker import rerank
from vector_db.qdrant_client import search

settings = get_settings()


# ── Category router ─────────────────────────────────────────────────────────

_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Travel":      ["travel", "flight", "hotel", "taxi", "uber", "per diem",
                    "reimbursement", "international", "trip", "visa", "cab"],
    "HR":          ["leave", "maternity", "paternity", "employee", "appraisal",
                    "grievance", "resignation", "hiring", "conduct", "attendance"],
    "Legal":       ["compliance", "gdpr", "audit", "regulation", "contract",
                    "nda", "confidentiality", "litigation"],
    "Finance":     ["budget", "invoice", "payment", "expense", "cost",
                    "capex", "opex", "approval"],
    "Procurement": ["vendor", "supplier", "rfp", "tender", "purchase order"],
    "Security":    ["password", "access", "breach", "cybersecurity", "vpn", "mfa"],
    "Insurance":   ["insurance", "claim", "premium", "coverage"],
}


def detect_category(query: str) -> str | None:
    """
    Return the most likely policy category for a query, or None if ambiguous.
    None means the vector search runs across all categories (no filter).
    """
    lower  = query.lower()
    scores = {
        cat: sum(1 for kw in kws if kw in lower)
        for cat, kws in _CATEGORY_KEYWORDS.items()
    }
    best, best_score = max(scores.items(), key=lambda x: x[1])
    if best_score >= 1:
        logger.info(f"Category detected: {best} (score={best_score})")
        return best
    logger.info("Category ambiguous — unfiltered search")
    return None


# ── Version post-filter ─────────────────────────────────────────────────────

def _filter_latest_versions(candidates: list[dict]) -> list[dict]:
    """
    For each document_id in the candidate set, discard chunks whose
    effective_date is older than the maximum date seen for that document.

    This prevents outdated policy versions from appearing in answers.

    Example:
        TRV-POL-2026 V1 (2023) → discarded
        TRV-POL-2026 V2 (2024) → discarded
        TRV-POL-2026 V3 (2026) → kept
    """
    def parse_date(c: dict) -> date:
        raw = c.get("metadata", {}).get("effective_date", "")
        try:
            return date.fromisoformat(raw)
        except (ValueError, TypeError):
            return date.min

    groups: dict[str, list[dict]] = defaultdict(list)
    no_id:  list[dict] = []

    for c in candidates:
        doc_id = c.get("metadata", {}).get("document_id")
        if doc_id:
            groups[doc_id].append(c)
        else:
            no_id.append(c)

    kept: list[dict] = []
    for doc_id, chunks in groups.items():
        latest = max(parse_date(c) for c in chunks)
        kept.extend(c for c in chunks if parse_date(c) == latest)

    total_before = sum(len(v) for v in groups.values()) + len(no_id)
    total_after  = len(kept) + len(no_id)
    logger.info(f"Version filter: {total_before} → {total_after} chunks")
    return kept + no_id


# ── Main pipeline ────────────────────────────────────────────────────────────

def retrieve(
    query:           str,
    use_expansion:   bool = True,
    use_hyde:        bool = True,
    filter_category: bool = True,
    top_k_retrieval: int | None = None,
    top_k_rerank:    int | None = None,
) -> list[dict]:
    """
    Run the full retrieval pipeline and return reranked chunks.

    Args:
        query:           The user's original question.
        use_expansion:   Generate multi-query variants (default: True).
        use_hyde:        Use HyDE embedding (default: True).
        filter_category: Apply metadata pre-filter (default: True).
        top_k_retrieval: Candidates to retrieve per embedding vector.
        top_k_rerank:    Final chunks to keep after reranking.

    Returns:
        List of top-K chunk dicts::

            {
                "chunk_text":   str,
                "metadata":     dict,
                "score":        float,   # vector similarity
                "rerank_score": float,   # cross-encoder score
            }
    """
    retrieval_k = top_k_retrieval or settings.retrieval_top_k
    rerank_k    = top_k_rerank    or settings.reranker_top_k

    # ── Step 1: Query expansion ──────────────────────────────────────────────
    queries = expand_query(query, n=4) if use_expansion else [query]
    logger.info(f"Pipeline: {len(queries)} query variant(s)")

    # ── Step 2: Metadata filter ──────────────────────────────────────────────
    filters: dict[str, str] | None = None
    if filter_category:
        cat = detect_category(query)
        if cat:
            filters = {"policy_category": cat}

    # ── Step 3: Build embedding vectors ─────────────────────────────────────
    embeddings: list[list[float]] = [embed_query(q) for q in queries]
    if use_hyde:
        embeddings.append(generate_hyde_embedding(query))

    # ── Step 4 + 5: Retrieve and deduplicate ─────────────────────────────────
    seen: set[str] = set()
    candidates: list[dict] = []

    for vec in embeddings:
        for result in search(query_vector=vec, top_k=retrieval_k, filters=filters):
            cid = result["metadata"].get("chunk_id", result["chunk_text"][:64])
            if cid not in seen:
                seen.add(cid)
                candidates.append(result)

    logger.info(f"Unique candidates after dedup: {len(candidates)}")

    # ── Step 6: Version post-filter ─────────────────────────────────────────
    candidates = _filter_latest_versions(candidates)

    # ── Step 7: Rerank ───────────────────────────────────────────────────────
    top = rerank(query=query, candidates=candidates, top_k=rerank_k)

    logger.success(f"Retrieval complete: {len(top)} reranked chunks returned")
    return top
