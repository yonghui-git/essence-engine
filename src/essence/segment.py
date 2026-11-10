"""Deterministic segmentation of a source body into excerpts.

Excerpt ids (``<source id>#exc_<n>``) must be stable for the same text so that
provenance links in notes survive a reindex. The rules: split on blank lines
(paragraphs) and on chat message headings; merge short paragraphs; split very
long ones on sentence boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_CHARS = 1200
MIN_CHARS = 200
_HEADING = re.compile(r"^#{1,6} ")
_SENTENCE_END = re.compile(r"(?<=[.!?。！？])\s+")


@dataclass(frozen=True)
class Excerpt:
    n: int
    start: int
    end: int
    text: str

    def ref(self, source_id: str) -> str:
        return f"{source_id}#exc_{self.n}"


def _blocks(body: str) -> list[tuple[int, int]]:
    """(start, end) spans of paragraphs; a heading line always starts a new block."""
    spans: list[tuple[int, int]] = []
    start = None
    pos = 0
    for line in body.splitlines(keepends=True):
        stripped = line.strip()
        if not stripped:
            if start is not None:
                spans.append((start, pos))
                start = None
        elif _HEADING.match(line) and start is not None:
            spans.append((start, pos))
            start = pos
        elif start is None:
            start = pos
        pos += len(line)
    if start is not None:
        spans.append((start, pos))
    return spans


def _split_long(body: str, start: int, end: int) -> list[tuple[int, int]]:
    if end - start <= MAX_CHARS:
        return [(start, end)]
    out: list[tuple[int, int]] = []
    chunk_start = start
    last_cut = start
    for m in _SENTENCE_END.finditer(body, start, end):
        if m.end() - chunk_start > MAX_CHARS and last_cut > chunk_start:
            out.append((chunk_start, last_cut))
            chunk_start = last_cut
        last_cut = m.end()
    # Hard-split anything still too long (no sentence breaks).
    while end - chunk_start > MAX_CHARS:
        out.append((chunk_start, chunk_start + MAX_CHARS))
        chunk_start += MAX_CHARS
    out.append((chunk_start, end))
    return out


def segment(body: str) -> list[Excerpt]:
    spans: list[tuple[int, int]] = []
    for s, e in _blocks(body):
        spans.extend(_split_long(body, s, e))

    merged: list[tuple[int, int]] = []
    for s, e in spans:
        starts_heading = bool(_HEADING.match(body[s:e]))
        if merged and not starts_heading:
            ps, pe = merged[-1]
            if (pe - ps) < MIN_CHARS and (e - ps) <= MAX_CHARS:
                merged[-1] = (ps, e)
                continue
        merged.append((s, e))

    excerpts = []
    for n, (s, e) in enumerate(merged, start=1):
        text = body[s:e].strip()
        if text:
            excerpts.append(Excerpt(len(excerpts) + 1, s, e, text))
    return excerpts
