"""
retrieval/embedder.py

Embedding generation using BAAI/bge-large-en-v1.5 via sentence-transformers.

Two separate functions are intentional:
  • embed_query()     — prefixes the BGE instruction for query-side encoding
  • embed_documents() — no prefix, batch-optimised for ingestion

This asymmetry is recommended by the BGE authors and improves retrieval
precision by aligning query and document embedding spaces.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from configs.settings import get_settings

settings = get_settings()

# BGE instruction prefix for query embedding (not used on document side)
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ── Model singleton ─────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    logger.info(f"Loading embedding model: {settings.embedding_model}")
    model = SentenceTransformer(settings.embedding_model)
    dim = model.get_sentence_embedding_dimension()
    logger.success(f"Embedding model ready  (dim={dim})")
    return model


# ── Public API ───────────────────────────────────────────────────────────────

def embed_query(text: str) -> list[float]:
    """
    Embed a single query string using the BGE query instruction prefix.
    Returns a normalised float vector.

    Example::
        vec = embed_query("Can I claim Uber during business travel?")
    """
    model  = _load_model()
    vector = model.encode(_QUERY_PREFIX + text, normalize_embeddings=True)
    return vector.tolist()


def embed_documents(texts: list[str], batch_size: int | None = None) -> list[list[float]]:
    """
    Embed a list of document chunks in batches (no query prefix).
    Returns one normalised float vector per input text.

    Example::
        vectors = embed_documents(["Policy clause 1...", "Policy clause 2..."])
    """
    bs    = batch_size or settings.embedding_batch_size
    model = _load_model()

    logger.info(f"Embedding {len(texts)} documents  (batch_size={bs})")
    all_vecs: list[np.ndarray] = []

    for start in range(0, len(texts), bs):
        batch = texts[start : start + bs]
        vecs  = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_vecs.append(vecs)
        logger.debug(f"  batch {start // bs + 1}/{(len(texts) - 1) // bs + 1} done")

    return np.vstack(all_vecs).tolist()


def embedding_dim() -> int:
    """Return the vector dimensionality of the loaded model."""
    return _load_model().get_sentence_embedding_dimension()
