"""
ingestion/chunker.py

Markdown-aware semantic chunker for Project Aegis.

Design principles:
  1. Split at markdown header boundaries (#, ##, ###) to respect
     the logical structure of policy documents.
  2. Detect and protect complete markdown tables — never split mid-row.
     If a table exceeds max_tokens, split row-by-row but prepend the
     header row to every resulting sub-chunk.
  3. Apply a configurable word-level overlap between consecutive chunks
     so that sentences spanning a boundary are still retrievable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Generator

from loguru import logger

from configs.settings import get_settings

settings = get_settings()

# ── Data model ──────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    text:       str
    h1_header:  str = ""
    h2_header:  str = ""
    h3_header:  str = ""
    chunk_index: int = 0
    token_count: int = 0


# ── Chunker ─────────────────────────────────────────────────────────────────

class SemanticChunker:
    """
    Splits a markdown document into semantically coherent chunks.

    Usage::

        chunker = SemanticChunker()
        chunks  = chunker.chunk(markdown_text)
    """

    # Matches a complete markdown table block (header + separator + data rows)
    _TABLE_RE = re.compile(
        r"((?:\|[^\n]+\|\n)+(?:\|[-| :]+\|\n)(?:\|[^\n]+\|\n)*)",
        re.MULTILINE,
    )
    # Matches any markdown heading line
    _HEADER_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

    def __init__(
        self,
        max_tokens:   int | None   = None,
        overlap_pct:  float | None = None,
    ) -> None:
        self.max_tokens  = max_tokens  or settings.max_chunk_tokens
        self.overlap_pct = overlap_pct or settings.chunk_overlap_pct

    # ── Public API ───────────────────────────────────────────────────────────

    def chunk(self, markdown_text: str) -> list[Chunk]:
        """
        Return a flat, numbered list of Chunk objects.

        Pipeline:
          parse headers → split sections → preserve tables →
          apply overlap → assign indices
        """
        sections = list(self._split_by_headers(markdown_text))
        raw_chunks: list[Chunk] = []

        for section in sections:
            raw_chunks.extend(self._split_section(section))

        chunks = self._apply_overlap(raw_chunks)

        for i, c in enumerate(chunks):
            c.chunk_index  = i
            c.token_count  = self._tokens(c.text)

        logger.info(f"Chunked document into {len(chunks)} segments")
        return chunks

    # ── Header splitting ─────────────────────────────────────────────────────

    def _split_by_headers(self, text: str) -> Generator[dict, None, None]:
        """
        Emit one section dict per heading encountered, tracking h1/h2/h3 context.
        Lines before the first heading are emitted as a preamble section.
        """
        current: dict = {"h1": "", "h2": "", "h3": "", "lines": []}

        for line in text.splitlines():
            m = self._HEADER_RE.match(line)
            if m:
                if current["lines"]:
                    yield dict(current)
                level = len(m.group(1))
                title = m.group(2).strip()
                if level == 1:
                    current = {"h1": title, "h2": "",            "h3": "", "lines": [line]}
                elif level == 2:
                    current = {"h1": current["h1"], "h2": title, "h3": "", "lines": [line]}
                else:
                    current = {"h1": current["h1"], "h2": current["h2"], "h3": title, "lines": [line]}
            else:
                current["lines"].append(line)

        if current["lines"]:
            yield dict(current)

    # ── Section splitting ────────────────────────────────────────────────────

    def _split_section(self, section: dict) -> list[Chunk]:
        """
        Further split one section into chunks, separating tables from prose.
        Tables are handled atomically; prose is split at sentence boundaries.
        """
        body = "\n".join(section["lines"])
        h1, h2, h3 = section["h1"], section["h2"], section["h3"]

        parts = self._TABLE_RE.split(body)
        result: list[Chunk] = []
        prose_buf: list[str] = []

        for part in parts:
            if self._TABLE_RE.fullmatch(part):
                if prose_buf:
                    result.extend(self._split_prose("\n".join(prose_buf), h1, h2, h3))
                    prose_buf = []
                result.extend(self._split_table(part, h1, h2, h3))
            else:
                prose_buf.append(part)

        if prose_buf:
            result.extend(self._split_prose("\n".join(prose_buf), h1, h2, h3))

        return result

    def _split_prose(self, text: str, h1: str, h2: str, h3: str) -> list[Chunk]:
        """
        Split prose at sentence boundaries when it exceeds max_tokens.
        Single oversized sentences are yielded as-is.
        """
        text = text.strip()
        if not text:
            return []

        if self._tokens(text) <= self.max_tokens:
            return [Chunk(text=text, h1_header=h1, h2_header=h2, h3_header=h3)]

        sentences = re.split(r"(?<=[.!?])\s+", text)
        buffer:  list[str] = []
        chunks:  list[Chunk] = []

        for sent in sentences:
            buffer.append(sent)
            if self._tokens(" ".join(buffer)) > self.max_tokens:
                if len(buffer) > 1:
                    chunks.append(Chunk(
                        text=" ".join(buffer[:-1]),
                        h1_header=h1, h2_header=h2, h3_header=h3,
                    ))
                    buffer = [buffer[-1]]
                else:
                    chunks.append(Chunk(
                        text=buffer[0],
                        h1_header=h1, h2_header=h2, h3_header=h3,
                    ))
                    buffer = []

        if buffer:
            chunks.append(Chunk(
                text=" ".join(buffer),
                h1_header=h1, h2_header=h2, h3_header=h3,
            ))
        return chunks

    def _split_table(self, table_md: str, h1: str, h2: str, h3: str) -> list[Chunk]:
        """
        Keep the table whole if it fits within max_tokens.
        Otherwise split row-by-row, prepending header+separator to every chunk
        so column context is never lost.

        Example sub-chunk structure:
            | Country | Per Diem |
            | ---     | ---      |
            | USA     | $120     |
            | UK      | £95      |
        """
        table_md = table_md.strip()
        if self._tokens(table_md) <= self.max_tokens:
            return [Chunk(text=table_md, h1_header=h1, h2_header=h2, h3_header=h3)]

        lines = table_md.splitlines()
        if len(lines) < 3:
            return [Chunk(text=table_md, h1_header=h1, h2_header=h2, h3_header=h3)]

        header_row    = lines[0]
        separator_row = lines[1]
        data_rows     = lines[2:]

        buffer: list[str] = []
        chunks: list[Chunk] = []

        for row in data_rows:
            candidate = "\n".join([header_row, separator_row] + buffer + [row])
            if self._tokens(candidate) > self.max_tokens and buffer:
                chunk_text = "\n".join([header_row, separator_row] + buffer)
                chunks.append(Chunk(text=chunk_text, h1_header=h1, h2_header=h2, h3_header=h3))
                buffer = [row]
            else:
                buffer.append(row)

        if buffer:
            chunk_text = "\n".join([header_row, separator_row] + buffer)
            chunks.append(Chunk(text=chunk_text, h1_header=h1, h2_header=h2, h3_header=h3))

        return chunks

    # ── Overlap ──────────────────────────────────────────────────────────────

    def _apply_overlap(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Prepend the last overlap_pct words of chunk[i-1] to chunk[i].

        This preserves sentence continuity across boundaries. Example:
            Chunk A ends:   "...Employees traveling internationally"
            Chunk B starts: "traveling internationally must obtain a visa..."
        """
        if len(chunks) < 2:
            return chunks

        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_words    = chunks[i - 1].text.split()
            n_overlap     = max(1, int(len(prev_words) * self.overlap_pct))
            overlap_text  = " ".join(prev_words[-n_overlap:])
            merged_text   = (overlap_text + " " + chunks[i].text).strip()
            result.append(Chunk(
                text=merged_text,
                h1_header=chunks[i].h1_header,
                h2_header=chunks[i].h2_header,
                h3_header=chunks[i].h3_header,
            ))
        return result

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _tokens(text: str) -> int:
        """Rough token estimate: ~4 chars per token (GPT tokenisation)."""
        return len(text) // 4
