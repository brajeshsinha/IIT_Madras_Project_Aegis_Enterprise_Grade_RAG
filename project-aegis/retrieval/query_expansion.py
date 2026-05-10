"""
retrieval/query_expansion.py

Multi-query expansion using an LLM.

Problem: Users phrase questions using vocabulary that may not match policy
document wording. A query like "Can I expense a taxi?" might never appear
verbatim in the policy — but "ground transportation reimbursement" will.

Solution: Ask the LLM to generate N semantically equivalent rephrasings
of the original query. Retrieval runs against all variants; results are
merged and deduplicated.

Example:
    Original:   "Can I expense a taxi?"
    Variants:   ["Ground transportation reimbursement policy",
                 "Cab fare expense rules",
                 "Uber and ride-sharing reimbursement",
                 "Approved modes of local transport"]
"""

from __future__ import annotations

import json
import re

from langchain_openai import ChatOpenAI
from loguru import logger

from configs.settings import get_settings

settings = get_settings()


# ── Prompt ───────────────────────────────────────────────────────────────────

_PROMPT = """\
You are an enterprise search assistant.

Generate {n} alternative phrasings of the user's query below.
Each variant must:
  - Express the same intent as the original
  - Use different vocabulary (synonyms, formal language, related terms)
  - Be a self-contained question or phrase

Respond ONLY with a JSON array of strings. No explanation. No markdown fences.

User query: {query}
"""


# ── Public API ───────────────────────────────────────────────────────────────

def expand_query(query: str, n: int = 4) -> list[str]:
    """
    Return [original] + up to n LLM-generated variants.
    Falls back to [original] if the LLM call fails.

    Example::
        variants = expand_query("Can I expense a taxi?", n=4)
        # ["Can I expense a taxi?",
        #  "Ground transportation reimbursement",
        #  "Cab fare policy",
        #  "Uber reimbursement rules",
        #  "Ride-sharing expense policy"]
    """
    llm    = _llm()
    prompt = _PROMPT.format(query=query, n=n)

    try:
        response = llm.invoke(prompt)
        variants = _parse(response.content)

        # Deduplicate preserving order; always keep original first
        seen   = {query.lower()}
        result = [query]
        for v in variants:
            v = v.strip()
            if v and v.lower() not in seen:
                result.append(v)
                seen.add(v.lower())

        logger.info(f"Query expansion: {len(result)} variants for {query!r:.50}")
        return result

    except Exception as exc:
        logger.warning(f"Query expansion failed ({exc}) — using original only")
        return [query]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model,
        temperature=0.3,
        openai_api_key=settings.openai_api_key,
    )


def _parse(text: str) -> list[str]:
    """Extract a JSON array from the LLM response, tolerating markdown fences."""
    cleaned = re.sub(r"```(?:json)?|```", "", text).strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return [str(i) for i in parsed]
    except json.JSONDecodeError:
        pass

    # Fallback: treat each non-empty line as a variant
    return [
        re.sub(r'^\d+\.\s*|^[-•]\s*|^"(.+)"$', r"\1", l).strip()
        for l in text.splitlines()
        if l.strip()
    ]
