"""
app/generator.py

Final answer generation using the top-K reranked chunks as context.

Design decisions:
  • System prompt enforces strict grounding — no outside knowledge.
  • Source attribution is required — LLM must cite document_id and section.
  • If context is empty, a polite "not found" message is returned without
    invoking the LLM (saves tokens and avoids hallucination).
  • Context window is kept tight (top-5 chunks only) to avoid the
    "lost in the middle" problem and minimise token cost.

Usage::
    from app.generator import generate_answer

    result = generate_answer(query="Can I claim Uber?", chunks=top5_chunks)
    print(result["answer"])
    print(result["sources"])
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI
from loguru import logger

from configs.settings import get_settings

settings = get_settings()


# ── Prompt templates ────────────────────────────────────────────────────────

_SYSTEM = """\
You are an enterprise policy assistant for a large organisation.

Answer employee questions accurately and concisely using ONLY the policy
context provided below. Do not use any outside knowledge.

Rules:
  1. Base your entire answer on the provided context excerpts only.
  2. If the answer is not present in the context, respond exactly:
     "No policy information found for this query. Please contact your HR
      or relevant department for assistance."
  3. Mention the policy document name, version, and section where possible.
  4. Reproduce exact numerical values (amounts, dates, durations) from the
     context — never round, estimate, or paraphrase numbers.
  5. Write clearly and professionally — employees rely on this for compliance.
"""

_USER_TEMPLATE = """\
Employee question: {query}

Policy context:
{context}

Answer:
"""


# ── Public API ───────────────────────────────────────────────────────────────

def generate_answer(query: str, chunks: list[dict]) -> dict:
    """
    Generate a grounded answer from the top-K reranked chunks.

    Args:
        query:  The original user question.
        chunks: Reranked chunks from retrieval/pipeline.py, each::

                    {
                        "chunk_text":   str,
                        "metadata":     dict,
                        "rerank_score": float,
                    }

    Returns::

        {
            "answer":  str,         # final grounded answer
            "sources": list[dict],  # metadata dicts for each chunk used
            "model":   str,         # model name used
        }
    """
    if not chunks:
        return {
            "answer": (
                "No policy information found for this query. "
                "Please contact your HR or relevant department for assistance."
            ),
            "sources": [],
            "model": settings.openai_model,
        }

    context = _build_context(chunks)
    user_msg = _USER_TEMPLATE.format(query=query, context=context)

    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0.1,   # low temperature for factual consistency
        openai_api_key=settings.openai_api_key,
    )

    logger.info(f"Generating answer  (model={settings.openai_model})")
    response = llm.invoke([
        {"role": "system", "content": _SYSTEM},
        {"role": "user",   "content": user_msg},
    ])

    answer  = response.content.strip()
    sources = [c["metadata"] for c in chunks]

    logger.success(f"Answer generated: {len(answer)} chars, {len(sources)} sources")
    return {
        "answer":  answer,
        "sources": sources,
        "model":   settings.openai_model,
    }


# ── Internals ────────────────────────────────────────────────────────────────

def _build_context(chunks: list[dict]) -> str:
    """
    Format the reranked chunks into a numbered context block for the prompt.

    Each entry includes:
      [N] Document: <doc_id>  |  Section: <h1> > <h2>  |  Relevance: <score>
      <chunk text>
    """
    parts: list[str] = []

    for i, chunk in enumerate(chunks, start=1):
        meta    = chunk.get("metadata", {})
        doc_id  = meta.get("document_id", "Unknown")
        h1      = meta.get("h1_header", "")
        h2      = meta.get("h2_header", "")
        section = " > ".join(filter(None, [h1, h2])) or "—"
        score   = chunk.get("rerank_score", 0.0)

        header = f"[{i}] Document: {doc_id}  |  Section: {section}  |  Relevance: {score:.2f}"
        parts.append(f"{header}\n{chunk['chunk_text']}")

    return "\n\n---\n\n".join(parts)
