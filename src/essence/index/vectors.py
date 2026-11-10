"""Note embeddings: a local embedder, an embedding cache, and brute-force cosine search.

A personal knowledge base holds thousands of notes, not millions, so an exact
scan over float32 vectors with numpy is fast (milliseconds) and needs no
extension. The `VectorStore` boundary keeps sqlite-vec / LanceDB swappable.
"""

from __future__ import annotations

import hashlib
import sqlite3
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    model_name: str

    def embed(self, texts: list[str]) -> list[np.ndarray]: ...


class FastEmbedder:
    """Local ONNX embeddings via fastembed (model downloaded once, then offline)."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:
                raise RuntimeError("vector search needs fastembed: pip install 'essence-engine[vectors]'") from exc
            import os
            from pathlib import Path

            # A stable cache (fastembed defaults to the temp dir, which gets cleaned).
            cache = Path(os.environ.get("ESSENCE_MODEL_CACHE", Path.home() / ".cache" / "essence-engine" / "models"))
            cache.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            self._model = TextEmbedding(model_name=self.model_name, cache_dir=str(cache))
        return self._model

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        return [np.asarray(v, dtype=np.float32) for v in self._load().embed(texts)]


def make_embedder(config) -> Embedder | None:
    if config.embed_provider == "fastembed":
        try:
            import fastembed  # noqa: F401
        except ImportError:
            return None
        return FastEmbedder(config.embed_model)
    return None


def note_text(title: str, applies_when: str | None, body: str) -> str:
    parts = [title]
    if applies_when:
        parts.append(f"Applies when: {applies_when}")
    parts.append(body[:4000])
    return "\n".join(parts)


def _key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n else v


def embed_texts(conn: sqlite3.Connection, embedder: Embedder, texts: list[str]) -> list[np.ndarray]:
    """Embed with the cache: only texts never seen with this model are computed."""
    keys = [_key(t) for t in texts]
    found: dict[str, np.ndarray] = {}
    for i in range(0, len(keys), 500):
        chunk = keys[i : i + 500]
        rows = conn.execute(
            f"SELECT hash, vector FROM embed_cache WHERE model = ? AND hash IN ({','.join('?' * len(chunk))})",
            [embedder.model_name, *chunk],
        ).fetchall()
        found.update({r[0]: np.frombuffer(r[1], dtype=np.float32) for r in rows})
    missing = [(k, t) for k, t in zip(keys, texts) if k not in found]
    if missing:
        unique = dict(missing)
        vectors = embedder.embed(list(unique.values()))
        with conn:
            for k, v in zip(unique.keys(), vectors):
                v = _normalize(v.astype(np.float32))
                found[k] = v
                conn.execute(
                    "INSERT OR REPLACE INTO embed_cache (hash, model, dim, vector) VALUES (?, ?, ?, ?)",
                    (k, embedder.model_name, int(v.shape[0]), v.tobytes()),
                )
    return [found[k] for k in keys]


def embed_notes(conn: sqlite3.Connection, embedder: Embedder) -> int:
    """Attach a vector to every indexed note. Returns how many notes have vectors."""
    rows = conn.execute(
        "SELECT n.id, n.title, f.applies_when, f.body FROM notes n JOIN notes_fts f ON f.note_id = n.id"
    ).fetchall()
    if not rows:
        return 0
    texts = [note_text(r["title"], r["applies_when"], r["body"]) for r in rows]
    embed_texts(conn, embedder, texts)
    with conn:
        conn.execute("DELETE FROM note_vectors")
        conn.executemany(
            "INSERT INTO note_vectors (note_id, hash, model) VALUES (?, ?, ?)",
            [(r["id"], _key(t), embedder.model_name) for r, t in zip(rows, texts)],
        )
    return len(rows)


def vector_search(
    conn: sqlite3.Connection,
    embedder: Embedder,
    query: str,
    *,
    limit: int = 50,
    include_inactive: bool = False,
    types: list[str] | None = None,
) -> list[tuple[str, float]]:
    """(note id, cosine similarity), best first."""
    sql = (
        "SELECT v.note_id, c.vector FROM note_vectors v"
        " JOIN embed_cache c ON c.hash = v.hash AND c.model = v.model"
        " JOIN notes n ON n.id = v.note_id WHERE v.model = ?"
    )
    params: list = [embedder.model_name]
    if not include_inactive:
        sql += " AND n.status = 'active'"
    if types:
        sql += f" AND n.type IN ({','.join('?' * len(types))})"
        params += types
    rows = conn.execute(sql, params).fetchall()
    if not rows:
        return []
    ids = [r[0] for r in rows]
    matrix = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    q = embed_texts(conn, embedder, [query])[0]
    sims = matrix @ q
    order = np.argsort(-sims)[:limit]
    return [(ids[i], float(sims[i])) for i in order]


def similar_pairs(conn: sqlite3.Connection, model: str, threshold: float) -> list[tuple[str, str, float]]:
    """Pairs of active notes of the same type whose cosine similarity is >= threshold."""
    rows = conn.execute(
        "SELECT v.note_id, n.type, c.vector FROM note_vectors v"
        " JOIN embed_cache c ON c.hash = v.hash AND c.model = v.model"
        " JOIN notes n ON n.id = v.note_id WHERE v.model = ? AND n.status = 'active' ORDER BY v.note_id",
        (model,),
    ).fetchall()
    out: list[tuple[str, str, float]] = []
    by_type: dict[str, list[tuple[str, np.ndarray]]] = {}
    for r in rows:
        by_type.setdefault(r[1], []).append((r[0], np.frombuffer(r[2], dtype=np.float32)))
    for items in by_type.values():
        if len(items) < 2:
            continue
        ids = [i for i, _ in items]
        m = np.vstack([v for _, v in items])
        sims = m @ m.T
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                if sims[a, b] >= threshold:
                    out.append((ids[a], ids[b], float(sims[a, b])))
    return sorted(out, key=lambda t: -t[2])
