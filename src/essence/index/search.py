"""Search over notes.

- `search`: full-text (FTS5 + BM25), titles weighted highest.
- `hybrid_search`: full-text + vector similarity + 1-hop link expansion,
  fused with reciprocal rank fusion (RRF). Falls back to full-text + links
  when no embedder is available.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

_TOKEN = re.compile(r"\w+", re.UNICODE)
RRF_K = 60


@dataclass
class SearchHit:
    note_id: str
    type: str
    status: str
    title: str
    path: str
    snippet: str
    score: float
    via: str = "text"  # text | vector | link | text+vector ...


def to_fts_query(text: str, mode: str = "all") -> str:
    """Turn free text into a safe FTS5 query.

    mode="all": every word must match, the last as a prefix (interactive search).
    mode="any": any word may match (retrieval of related notes from long text).
    Quoting each token stops user input from being read as FTS5 syntax.
    """
    tokens = _TOKEN.findall(text)
    if not tokens:
        return ""
    if mode == "any":
        uniq = list(dict.fromkeys(t.lower() for t in tokens if len(t) > 2))[:64]
        return " OR ".join(f'"{t}"' for t in uniq)
    parts = [f'"{t}"' for t in tokens[:-1]] + [f'"{tokens[-1]}"*']
    return " ".join(parts)


def _log(conn: sqlite3.Connection, kind: str, detail: str) -> None:
    with conn:
        conn.execute(
            "INSERT INTO events (at, kind, detail) VALUES (?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"), kind, detail),
        )


def _fts(conn, query, *, limit, types, include_inactive):
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
    return conn.execute("".join(sql), params).fetchall()


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
    hits = [
        SearchHit(r["id"], r["type"], r["status"], r["title"], r["path"], r["snip"], r["score"])
        for r in _fts(conn, query, limit=limit, types=types, include_inactive=include_inactive)
    ]
    if log:
        _log(conn, "search", f"hits={len(hits)}")
    return hits


def _neighbors(conn, ids: list[str]) -> list[str]:
    if not ids:
        return []
    marks = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT src, dst FROM edges WHERE (src IN ({marks}) OR dst IN ({marks})) AND kind != 'derived_from'",
        [*ids, *ids],
    ).fetchall()
    top = set(ids)
    out: list[str] = []
    for src, dst in rows:
        # A note linked to a top hit gets a vote, even if it is itself a top hit.
        for other, anchor in ((dst, src), (src, dst)):
            if anchor in top and other != anchor and other not in out:
                out.append(other)
    return out


def hybrid_search(
    conn: sqlite3.Connection,
    text: str,
    *,
    embedder=None,
    limit: int = 10,
    types: list[str] | None = None,
    include_inactive: bool = False,
    mode: str = "all",
    log: bool = True,
    hop: bool = True,
) -> list[SearchHit]:
    ranked: dict[str, list[int]] = {}
    via: dict[str, set[str]] = {}
    snippets: dict[str, str] = {}

    query = to_fts_query(text, mode)
    if query:
        rows = _fts(conn, query, limit=50, types=types, include_inactive=include_inactive)
        if not rows and mode == "all":
            rows = _fts(conn, to_fts_query(text, "any"), limit=50, types=types, include_inactive=include_inactive)
        for rank, r in enumerate(rows):
            ranked.setdefault(r["id"], []).append(rank)
            via.setdefault(r["id"], set()).add("text")
            snippets[r["id"]] = r["snip"]

    if embedder is not None and text.strip():
        from essence.index.vectors import vector_search

        for rank, (nid, _sim) in enumerate(
            vector_search(conn, embedder, text, limit=50, include_inactive=include_inactive, types=types)
        ):
            ranked.setdefault(nid, []).append(rank)
            via.setdefault(nid, set()).add("vector")

    scores = {nid: sum(1.0 / (RRF_K + r + 1) for r in ranks) for nid, ranks in ranked.items()}

    if hop and scores:
        top = [nid for nid, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:5]]
        for rank, nid in enumerate(_neighbors(conn, top)):
            # A linked note counts as half a vote.
            scores[nid] = scores.get(nid, 0.0) + 0.5 / (RRF_K + rank + 1)
            via.setdefault(nid, set()).add("link")

    if not scores:
        if log:
            _log(conn, "search", "hits=0")
        return []

    ids = list(scores)
    marks = ",".join("?" * len(ids))
    sql = f"SELECT n.id, n.type, n.status, n.title, n.path, f.body FROM notes n JOIN notes_fts f ON f.note_id = n.id WHERE n.id IN ({marks})"
    params: list = list(ids)
    if not include_inactive:
        sql += " AND n.status = 'active'"
    if types:
        sql += f" AND n.type IN ({','.join('?' * len(types))})"
        params += types
    rows = {r["id"]: r for r in conn.execute(sql, params)}

    hits = []
    for nid in sorted(rows, key=lambda i: -scores[i])[:limit]:
        r = rows[nid]
        snippet = snippets.get(nid) or " ".join((r["body"] or "").split())[:160]
        hits.append(
            SearchHit(nid, r["type"], r["status"], r["title"], r["path"], snippet, scores[nid], "+".join(sorted(via[nid])))
        )
    if log:
        _log(conn, "search", f"hits={len(hits)}")
    return hits
