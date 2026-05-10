"""
retrieval/reranker.py

Cross-encoder reranking using BAAI/bge-reranker-large.

Why reranking?
  Vector search (bi-encoder) retrieves chunks that are broadly similar to
  the query, but it scores them based on embedding distance — not logical
  relevance to the specific question. A bi-encoder encodes the query and
  chunk independently and never sees them together.

  A cross-encoder takes [query, chunk] as a joint input. It can reason
  about how well this specific chunk actually answers this specific question.
  This makes it significantly more accurate, but too slow for full-corpus
  search — so we apply it only to the 25 candidates the vector search
  already returned.

Workflow:
  1. Vector search returns top 25 candidates.
  2. Cross-encoder scores each (query, chunk) pair → float in [0, 1].
  3. Candidates sorted by reranker score descending.
  4. Top 5 retained for final LLM answer generation.

Usage::
    from retrieval.reranker import rerank, compute_score

    top5 = rerank(query="Can I claim Uber?", candidates=candidates, top_k=5)
    score = compute_score(query="...", chunk_text="...")
"""

from __future__ import annotations

from functools import lru_cache

from loguru import logger
from sentence_transformers import CrossEncoder

from configs.settings import get_settings

settings = get_settings()


# ── Model singleton ─────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load() -> CrossEncoder:
    logger.info(f"Loading reranker: {settings.reranker_model}")
    model = CrossEncoder(settings.reranker_model, max_length=512)
    logger.success("Reranker model ready")
    return model


# ── Public API ───────────────────────────────────────────────────────────────

def rerank(
    query:      str,
    candidates: list[dict],
    top_k:      int | None = None,
) -> list[dict]:
    """
    Rerank candidate chunks against the query using a cross-encoder.

    Args:
        query:      The user's original question.
        candidates: List of dicts from vector search, each containing
                    at minimum {"chunk_text": str, "metadata": dict, "score": float}.
        top_k:      How many to return after reranking.
                    Defaults to settings.reranker_top_k (5).

    Returns:
        top_k candidates sorted by rerank_score descending.
        Each dict gains a "rerank_score" field (float, higher = more relevant).

    Example::
        top5 = rerank(
            query="Can I claim Uber during international travel?",
            candidates=retrieved_25_chunks,
            top_k=5,
        )
    """
    k = top_k or settings.reranker_top_k

    if not candidates:
        logger.warning("rerank() called with empty candidates list")
        return []

    model  = _load()
    pairs  = [[query, c["chunk_text"]] for c in candidates]

    logger.info(f"Reranking {len(pairs)} candidates → keeping top {k}")
    raw_scores = model.predict(pairs, show_progress_bar=False)

    scored = [
        {**cand, "rerank_score": float(score)}
        for cand, score in zip(candidates, raw_scores)
    ]
    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    top = scored[:k]

    logger.info(
        "Reranker scores: "
        + ", ".join(f"{r['rerank_score']:.3f}" for r in top)
    )
    return top


def compute_score(query: str, chunk_text: str) -> float:
    """
    Score a single (query, chunk) pair.

    Example::
        score = compute_score(
            query="Can I expense a taxi?",
            chunk_text="Employees may claim taxi expenses for approved travel...",
        )
        # 0.912
    """
    model = _load()
    score = model.predict([[query, chunk_text]], show_progress_bar=False)
    return float(score[0])
