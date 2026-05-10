"""
scripts/ingest_documents.py

CLI tool to bulk-ingest all policy documents from a directory.

Usage:
    python scripts/ingest_documents.py \\
        --input_dir  data/sample_policies/ \\
        --category   Travel \\
        --owner      GCT-RM \\
        --date       2026-01-01 \\
        --version    V3

All files with extensions .pdf / .docx / .md / .html in the directory are
parsed, chunked, embedded, and upserted into Qdrant.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from configs.settings import get_settings
from ingestion.chunker import SemanticChunker
from ingestion.metadata_extractor import extract_metadata
from ingestion.parser import parse_document
from retrieval.embedder import embed_documents
from vector_db.qdrant_client import (
    delete_by_document_id,
    ensure_collection,
    upsert_chunks,
)

settings  = get_settings()
SUPPORTED = {".pdf", ".docx", ".md", ".html", ".htm"}


def ingest_file(
    path:             Path,
    policy_category:  str,
    policy_owner:     str,
    effective_date:   str,
    document_version: str,
) -> int:
    """Ingest one document. Returns the number of chunks upserted."""
    logger.info(f"Processing: {path.name}")

    # 1. Parse to markdown
    markdown = parse_document(path)

    # 2. Semantic chunk
    chunker = SemanticChunker()
    chunks  = chunker.chunk(markdown)

    if not chunks:
        logger.warning(f"  No chunks produced for {path.name} — skipping")
        return 0

    # 3. Extract metadata
    enriched = [
        {
            "chunk": c,
            "meta":  extract_metadata(
                chunk=c,
                source_path=path,
                policy_category=policy_category,
                policy_owner=policy_owner,
                effective_date=effective_date,
                document_version=document_version,
            ),
        }
        for c in chunks
    ]

    doc_id = enriched[0]["meta"]["document_id"]

    # 4. Remove old version if present
    delete_by_document_id(doc_id)

    # 5. Embed (batched)
    texts   = [e["chunk"].text for e in enriched]
    vectors = embed_documents(texts)

    # 6. Upsert to Qdrant
    records = [
        {
            "chunk_id":  e["meta"]["chunk_id"],
            "embedding": vec,
            "metadata":  e["meta"],
            "text":      e["chunk"].text,
        }
        for e, vec in zip(enriched, vectors)
    ]
    upsert_chunks(records)

    logger.success(f"  {path.name} → {len(records)} chunks  (doc_id={doc_id})")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bulk-ingest policy documents into Qdrant"
    )
    parser.add_argument("--input_dir", required=True,
                        help="Directory containing policy documents")
    parser.add_argument("--category",  default="",
                        help="Policy category (e.g. Travel, HR). Auto-detected if omitted.")
    parser.add_argument("--owner",     default="",
                        help="Department owner code (e.g. GCT-RM)")
    parser.add_argument("--date",      default="",
                        help="Effective date YYYY-MM-DD")
    parser.add_argument("--version",   default="V1",
                        help="Document version string (e.g. V3)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"Directory not found: {input_dir}")

    files = [f for f in input_dir.iterdir() if f.suffix.lower() in SUPPORTED]
    if not files:
        raise SystemExit(f"No supported files found in {input_dir}")

    ensure_collection()
    logger.info(f"Found {len(files)} document(s) to ingest")

    total_chunks = 0
    for f in sorted(files):
        total_chunks += ingest_file(
            path=f,
            policy_category=args.category,
            policy_owner=args.owner,
            effective_date=args.date,
            document_version=args.version,
        )

    logger.success(
        f"Done — {len(files)} file(s), {total_chunks} total chunks ingested"
    )


if __name__ == "__main__":
    main()
