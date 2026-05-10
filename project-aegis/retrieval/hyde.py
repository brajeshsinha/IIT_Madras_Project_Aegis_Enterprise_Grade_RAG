"""
retrieval/hyde.py

Hypothetical Document Embeddings (HyDE).

Reference: Gao et al., "Precise Zero-Shot Dense Retrieval without
           Relevance Labels" (2022)  https://arxiv.org/abs/2212.10496

Problem:
  User questions and policy clauses sit in different parts of the embedding
  space. A question is syntactically very different from the formal policy
  text it relates to, even when they are semantically equivalent.

  Example:
      Query:  "Can I claim Uber rides during business travel?"
      Policy: "Employees may claim ride-sharing expenses for approved
               international business travel, subject to receipt submission."

  These embed to different vectors. Standard retrieval may miss the clause.

Solution:
  1. Ask the LLM to generate a hypothetical policy clause that would answer
     the question.
  2. Embed the hypothetical clause (not the question).
  3. The hypothetical clause is close in embedding space to the real policy
     text, so retrieval succeeds.

Usage::
    vec      = generate_hyde_embedding("Can I claim Uber?")   # for retrieval
    doc_text = generate_hypothetical_doc("Can I claim Uber?")  # for debugging
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI
from loguru import logger

from configs.settings import get_settings
from retrieval.embedder import embed_query

settings = get_settings()


# ── Prompt ───────────────────────────────────────────────────────────────────

_PROMPT = """\
You are an enterprise policy expert.

Write a short excerpt from a corporate policy document that would directly
answer the employee's question below.

Instructions:
  - Write in the style of formal policy text (authoritative, third-person, precise).
  - Do NOT start with "According to policy" or similar preambles.
  - Write the policy clause itself, as if reading from the document.
  - 2–4 sentences maximum.

Employee question: {query}
"""


# ── Public API ───────────────────────────────────────────────────────────────

def generate_hyde_embedding(query: str) -> list[float]:
    """
    Generate a hypothetical policy document excerpt for the query
    and return its embedding vector.

    Falls back to a standard query embedding if the LLM call fails.

    Example::
        vec = generate_hyde_embedding("Can I expense a taxi?")
    """
    doc = _generate(query)
    logger.debug(f"HyDE doc (truncated): {doc[:100]}...")
    return embed_query(doc)


def generate_hypothetical_doc(query: str) -> str:
    """
    Return the hypothetical policy text as a string.
    Useful for the Pipeline Debugger page in Streamlit.

    Example::
        text = generate_hypothetical_doc("Can I expense a taxi?")
        # "Employees may claim taxi expenses for approved business travel..."
    """
    return _generate(query)


# ── Internals ────────────────────────────────────────────────────────────────

def _generate(query: str) -> str:
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0.2,
        openai_api_key=settings.openai_api_key,
        max_tokens=200,
    )
    try:
        response = llm.invoke(_PROMPT.format(query=query))
        doc = response.content.strip()
        logger.info("HyDE: hypothetical document generated successfully")
        return doc
    except Exception as exc:
        logger.warning(f"HyDE generation failed ({exc}) — using raw query as fallback")
        return query
