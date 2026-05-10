"""
tests/test_chunker.py

Unit tests for the SemanticChunker.
Run: pytest tests/test_chunker.py -v
"""

import pytest

from ingestion.chunker import SemanticChunker, Chunk


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def chunker():
    return SemanticChunker(max_tokens=200, overlap_pct=0.10)


SAMPLE_MD = """\
# Corporate Travel Policy

## Domestic Travel

Employees travelling within the country may book economy class flights.
All bookings must be made through the approved travel portal.

## International Travel

### Passport Requirements

Employees must hold a passport valid for at least six months beyond
the travel return date.

### Per Diem Allowances

| Country | Daily Allowance |
| ---     | ---             |
| USA     | $120            |
| UK      | £95             |
| Germany | €100            |

## Expense Submission

All receipts must be submitted within 30 days of returning.
"""


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_produces_chunks(chunker):
    chunks = chunker.chunk(SAMPLE_MD)
    assert len(chunks) > 0, "Should produce at least one chunk"


def test_chunks_are_chunk_instances(chunker):
    chunks = chunker.chunk(SAMPLE_MD)
    for c in chunks:
        assert isinstance(c, Chunk)


def test_h1_header_captured(chunker):
    chunks = chunker.chunk(SAMPLE_MD)
    h1s = {c.h1_header for c in chunks}
    assert "Corporate Travel Policy" in h1s


def test_h2_header_captured(chunker):
    chunks = chunker.chunk(SAMPLE_MD)
    h2s = {c.h2_header for c in chunks}
    assert "International Travel" in h2s


def test_table_preserved_as_single_chunk(chunker):
    """A small table should not be split across chunks."""
    table_md = """\
## Per Diem

| Country | Amount |
| ---     | ---    |
| USA     | $120   |
| UK      | £95    |
"""
    chunks = chunker.chunk(table_md)
    table_chunks = [c for c in chunks if "USA" in c.text and "UK" in c.text]
    assert len(table_chunks) >= 1, "Table rows should appear together in at least one chunk"


def test_chunk_indices_sequential(chunker):
    chunks = chunker.chunk(SAMPLE_MD)
    for expected_i, chunk in enumerate(chunks):
        assert chunk.chunk_index == expected_i


def test_token_count_populated(chunker):
    chunks = chunker.chunk(SAMPLE_MD)
    for c in chunks:
        assert c.token_count > 0


def test_overlap_adds_context(chunker):
    """Each chunk after the first should start with words from the previous chunk."""
    long_md = "# Policy\n\n" + " ".join([f"sentence{i}." for i in range(200)])
    chunks = chunker.chunk(long_md)
    if len(chunks) >= 2:
        # The second chunk's text should contain some words from the first
        prev_words = set(chunks[0].text.split()[-20:])
        next_words = set(chunks[1].text.split()[:20])
        assert prev_words & next_words, "Overlap should share words between consecutive chunks"


def test_empty_document(chunker):
    chunks = chunker.chunk("")
    assert chunks == []


def test_single_line_document(chunker):
    chunks = chunker.chunk("# Title\n\nA single sentence.")
    assert len(chunks) >= 1
    assert "single sentence" in chunks[-1].text
