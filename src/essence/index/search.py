"""Full-text search over notes (FTS5 + BM25). Vectors and RRF fusion arrive in week 4."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

_TOKEN = re.compile(r"\w+", re.UNICODE)


@dataclass
class SearchHit:
    note_id: str
    type: str
    status: str
    title: str
    path: str
    snippet: str
    score: float


def to_fts_query(text: str) -> str:
    """Turn free text into a safe FTS5 query: every word must match, last word as a prefix.

    Quoting each token stops user input from being read as FTS5 syntax
    (AND/OR/NEAR, column filters, unbalanced quotes).
    """
    tokens = _TOKEN.findall(text)
    if not tokens:
        return ""
    parts = [f'"{t}"' for t in tokens[:-1]] + [f'"{tokens[-1]}"*']
    return " ".join(parts)


def search(
    conn: sqlite3.Connection,
    text: str,
    *,
    limit: int = 10,
    types: list[str] | None = None,
    include_inactive: bool = False,
    log: bool = True,
) -> list[SearchHit]:
    query = to_fts_query(text)
    if not query:
        return []
    sql = [
        "SELECT n.id, n.type, n.status, n.title, n.path,",
        " snippet(notes_fts, 2, '[', ']', ' ... ', 12) AS snip,",
        " bm25(notes_fts, 0.0, 5.0, 1.0, 2.0) AS score",
        " FROM notes_fts JOIN notes n ON n.id = notes_fts.note_id",
        " WHERE notes_fts MATCH ?",
    ]
    params: list = [query]
    if not include_inactive:
        sql.append(" AND n.status = 'active'")
    if types:
        sql.append(f" AND n.type IN ({', '.join('?' * len(types))})")
        params += types
    sql.append(" ORDER BY score LIMIT ?")
    params.append(limit)

    hits = [
        SearchHit(r["id"], r["type"], r["status"], r["title"], r["path"], r["snip"], r["score"])
        for r in conn.execute("".join(sql), params)
    ]
    if log:
        with conn:
            conn.execute(
                "INSERT INTO events (at, kind, detail) VALUES (?, 'search', ?)",
                (datetime.now(timezone.utc).isoformat(timespec="seconds"), f"hits={len(hits)}"),
            )
    return hits
