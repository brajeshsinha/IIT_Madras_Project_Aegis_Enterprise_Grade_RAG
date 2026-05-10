"""
tests/test_reranker.py

Unit tests for the reranker module.
Run: pytest tests/test_reranker.py -v

Note: these tests load the cross-encoder model, so they require
sentence-transformers and a GPU/CPU environment with enough memory.
Mark with @pytest.mark.slow if you want to skip in CI.
"""

import pytest

from retrieval.reranker import rerank, compute_score


# ── Fixtures ─────────────────────────────────────────────────────────────────

QUERY = "Can I claim Uber expenses during international business travel?"

CANDIDATES = [
    {
        "chunk_text": "Employees may claim Uber, taxi, and ride-sharing expenses "
                      "for approved international business travel, subject to receipt submission.",
        "metadata": {"document_id": "TRV-POL-2026-V3", "policy_category": "Travel"},
        "score": 0.85,
    },
    {
        "chunk_text": "The company cafeteria is open Monday to Friday, 8am to 6pm.",
        "metadata": {"document_id": "HR-POL-2026-V1", "policy_category": "HR"},
        "score": 0.60,
    },
    {
        "chunk_text": "All travel bookings must be made via the approved travel portal "
                      "at least 7 days in advance.",
        "metadata": {"document_id": "TRV-POL-2026-V3", "policy_category": "Travel"},
        "score": 0.75,
    },
]


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_rerank_returns_correct_count():
    result = rerank(QUERY, CANDIDATES, top_k=2)
    assert len(result) == 2


@pytest.mark.slow
def test_rerank_adds_rerank_score():
    result = rerank(QUERY, CANDIDATES, top_k=3)
    for item in result:
        assert "rerank_score" in item
        assert isinstance(item["rerank_score"], float)


@pytest.mark.slow
def test_rerank_sorted_descending():
    result = rerank(QUERY, CANDIDATES, top_k=3)
    scores = [r["rerank_score"] for r in result]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.slow
def test_rerank_uber_chunk_ranks_highest():
    """The Uber expense chunk should rank first for this query."""
    result = rerank(QUERY, CANDIDATES, top_k=3)
    assert "Uber" in result[0]["chunk_text"] or "ride-sharing" in result[0]["chunk_text"]


@pytest.mark.slow
def test_compute_score_returns_float():
    score = compute_score(QUERY, CANDIDATES[0]["chunk_text"])
    assert isinstance(score, float)


@pytest.mark.slow
def test_compute_score_range():
    score = compute_score(QUERY, CANDIDATES[0]["chunk_text"])
    # Cross-encoder scores are not strictly bounded, but relevant pairs
    # should score positively
    assert score > -10.0


def test_rerank_empty_candidates():
    result = rerank(QUERY, [], top_k=5)
    assert result == []
