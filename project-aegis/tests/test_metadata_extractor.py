"""
tests/test_metadata_extractor.py

Unit tests for the metadata extractor.
Run: pytest tests/test_metadata_extractor.py -v
"""

from pathlib import Path

import pytest

from ingestion.chunker import Chunk
from ingestion.metadata_extractor import extract_metadata, _detect_category


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_chunk(text: str, h1: str = "", h2: str = "") -> Chunk:
    return Chunk(text=text, h1_header=h1, h2_header=h2, chunk_index=0, token_count=len(text) // 4)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_extract_metadata_returns_dict():
    chunk = _make_chunk("Employees may claim taxi expenses during approved travel.")
    meta  = extract_metadata(chunk=chunk, source_path="travel_policy.pdf")
    assert isinstance(meta, dict)


def test_required_keys_present():
    chunk    = _make_chunk("Some policy text.")
    meta     = extract_metadata(chunk=chunk, source_path="hr_policy.docx")
    required = {
        "chunk_id", "document_id", "document_version", "source_file",
        "policy_category", "policy_owner", "effective_date",
        "h1_header", "h2_header", "chunk_index",
        "has_table", "has_numeric_data", "token_count", "indexed_at",
    }
    assert required.issubset(meta.keys())


def test_category_auto_detect_travel():
    chunk = _make_chunk(
        "Employees travelling internationally must retain hotel and taxi receipts.",
        h1="Corporate Travel Policy"
    )
    meta = extract_metadata(chunk=chunk, source_path="policy.pdf")
    assert meta["policy_category"] == "Travel"


def test_category_auto_detect_hr():
    chunk = _make_chunk(
        "Maternity leave entitlement is 26 weeks at full pay.",
        h1="HR Policy"
    )
    meta = extract_metadata(chunk=chunk, source_path="hr.pdf")
    assert meta["policy_category"] == "HR"


def test_category_override():
    chunk = _make_chunk("Some text.")
    meta  = extract_metadata(chunk=chunk, source_path="policy.pdf", policy_category="Legal")
    assert meta["policy_category"] == "Legal"


def test_version_in_document_id():
    chunk = _make_chunk("Some text.")
    meta  = extract_metadata(chunk=chunk, source_path="policy.pdf", document_version="V3")
    assert "V3" in meta["document_id"]


def test_has_table_true():
    table_text = "| Country | Amount |\n| --- | --- |\n| USA | $120 |"
    chunk = _make_chunk(table_text)
    meta  = extract_metadata(chunk=chunk, source_path="policy.pdf")
    assert meta["has_table"] is True


def test_has_table_false():
    chunk = _make_chunk("No tables here, just plain prose.")
    meta  = extract_metadata(chunk=chunk, source_path="policy.pdf")
    assert meta["has_table"] is False


def test_has_numeric_data_currency():
    chunk = _make_chunk("Per diem is $120 per day.")
    meta  = extract_metadata(chunk=chunk, source_path="policy.pdf")
    assert meta["has_numeric_data"] is True


def test_detect_category_fallback():
    result = _detect_category("general statement without keywords", "", "")
    assert result == "General"
