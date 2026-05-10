"""
ingestion/metadata_extractor.py

Extracts and enriches structured metadata for each chunk before it is
stored in Qdrant.

Metadata serves two purposes:
  1. Pre-filtering at query time — scope vector search to the right category.
  2. Post-filtering — discard chunks from superseded document versions.

Metadata fields stored per chunk:
  chunk_id          Unique UUID for this chunk point in Qdrant
  document_id       Stable ID derived from filename + version (e.g. TRV-POL-2026-V3)
  document_version  e.g. "V3"
  source_file       Original filename
  policy_category   HR / Travel / Legal / Finance / Procurement / Security / Insurance / General
  policy_owner      Department code (e.g. GCT-RM, HR-OPS)
  effective_date    ISO date string "YYYY-MM-DD"
  h1_header         Top-level section the chunk belongs to
  h2_header         Subsection
  h3_header         Nested subsection
  chunk_index       Position of chunk in the document
  has_table         True if the chunk contains a markdown table
  has_numeric_data  True if the chunk contains currency / percentages / numbers
  token_count       Estimated token count
  indexed_at        UTC timestamp of ingestion
"""

from __future__ import annotations

import re
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from ingestion.chunker import Chunk


# ── Public entry-point ──────────────────────────────────────────────────────

def extract_metadata(
    chunk:             Chunk,
    source_path:       str | Path,
    policy_category:   str = "",
    policy_owner:      str = "",
    effective_date:    str = "",
    document_version:  str = "V1",
) -> dict[str, Any]:
    """
    Build a complete metadata payload for a single chunk.

    If policy_category is not supplied, auto-detection runs over the
    chunk text and headers.
    """
    path    = Path(source_path)
    doc_id  = _derive_document_id(path, document_version)

    if not policy_category:
        policy_category = _detect_category(chunk.text, chunk.h1_header, chunk.h2_header)

    payload: dict[str, Any] = {
        # ── Identification ────────────────────────────────────────────────
        "chunk_id":         str(uuid.uuid4()),
        "document_id":      doc_id,
        "document_version": document_version,
        "source_file":      path.name,

        # ── Classification ────────────────────────────────────────────────
        "policy_category":  policy_category,
        "policy_owner":     policy_owner or _infer_owner(policy_category),
        "effective_date":   effective_date or date.today().isoformat(),

        # ── Structure ─────────────────────────────────────────────────────
        "h1_header":   chunk.h1_header,
        "h2_header":   chunk.h2_header,
        "h3_header":   chunk.h3_header,
        "chunk_index": chunk.chunk_index,

        # ── Content signals ───────────────────────────────────────────────
        "has_table":        _has_table(chunk.text),
        "has_numeric_data": _has_numbers(chunk.text),
        "token_count":      chunk.token_count,

        # ── Audit ─────────────────────────────────────────────────────────
        "indexed_at": datetime.utcnow().isoformat(),
    }

    logger.debug(
        f"Metadata: chunk={chunk.chunk_index} "
        f"doc={doc_id} cat={policy_category}"
    )
    return payload


# ── Document ID ─────────────────────────────────────────────────────────────

def _derive_document_id(path: Path, version: str) -> str:
    """
    Derive a stable, human-readable document ID from the filename.

    Examples:
        corporate_travel_policy.pdf  → TRV-POL-2026-V3
        hr_leave_policy.docx         → HR-POL-2026-V1
        unknown.md                   → POL-2026-V1
    """
    stem = path.stem.upper().replace(" ", "_").replace("-", "_")

    _PREFIXES = {
        "TRAVEL":      "TRV",
        "LEAVE":       "LV",
        "HR":          "HR",
        "FINANCE":     "FIN",
        "LEGAL":       "LGL",
        "PROCUREMENT": "PRC",
        "SECURITY":    "SEC",
        "INSURANCE":   "INS",
    }

    prefix = "POL"
    for keyword, code in _PREFIXES.items():
        if keyword in stem:
            prefix = f"{code}-POL"
            break

    year = datetime.utcnow().year
    return f"{prefix}-{year}-{version}"


# ── Category auto-detection ─────────────────────────────────────────────────

_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Travel": [
        "travel", "per diem", "reimbursement", "flight", "hotel",
        "taxi", "uber", "visa", "passport", "international", "trip",
        "ground transportation", "cab fare",
    ],
    "HR": [
        "maternity", "paternity", "leave", "employee", "hiring",
        "resignation", "performance", "appraisal", "grievance",
        "workplace", "conduct", "attendance",
    ],
    "Legal": [
        "compliance", "gdpr", "audit", "regulation", "liability",
        "contract", "intellectual property", "nda", "confidentiality",
        "litigation",
    ],
    "Finance": [
        "budget", "invoice", "payment", "expense", "cost centre",
        "purchase", "approval", "capex", "opex", "reconciliation",
    ],
    "Procurement": [
        "vendor", "supplier", "rfp", "rfq", "tender",
        "purchase order", "sourcing", "bidding",
    ],
    "Security": [
        "password", "access control", "data breach", "cybersecurity",
        "incident", "firewall", "encryption", "vpn", "mfa",
    ],
    "Insurance": [
        "insurance", "claim", "premium", "coverage",
        "beneficiary", "indemnity", "deductible",
    ],
}


def _detect_category(text: str, h1: str, h2: str) -> str:
    corpus = (text + " " + h1 + " " + h2).lower()
    scores = {
        cat: sum(1 for kw in keywords if kw in corpus)
        for cat, keywords in _CATEGORY_KEYWORDS.items()
    }
    best, score = max(scores.items(), key=lambda x: x[1])
    return best if score >= 1 else "General"


_OWNER_MAP: dict[str, str] = {
    "Travel":      "GCT-RM",
    "HR":          "HR-OPS",
    "Legal":       "LEGAL",
    "Finance":     "FIN-CTRL",
    "Procurement": "SCM",
    "Security":    "INFOSEC",
    "Insurance":   "RISK-MGT",
    "General":     "CORP",
}


def _infer_owner(category: str) -> str:
    return _OWNER_MAP.get(category, "CORP")


def _has_table(text: str) -> bool:
    return bool(re.search(r"^\|.+\|$", text, re.MULTILINE))


def _has_numbers(text: str) -> bool:
    return bool(re.search(r"[\$€£¥]\s*\d+|\d+\s*%|\b\d{2,}\b", text))
