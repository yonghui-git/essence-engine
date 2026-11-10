"""SQLite index: schema and migrations.

Two kinds of tables live here:

- DERIVED tables are rebuilt from the vault by ``reindex`` and can be dropped
  at any time (sources, excerpts, notes, note_versions, edges, FTS tables).
- OPERATIONAL tables hold state that is not in the vault (inbox suggestions,
  reuse events, the outbound LLM call log). ``reindex`` never touches them.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

DERIVED_TABLES = ["note_vectors", "edges", "note_versions", "excerpts", "notes", "sources"]
DERIVED_FTS = ["notes_fts", "excerpts_fts"]

_MIGRATIONS: list[str] = [
    # v1
    """
    CREATE TABLE sources (
        id          TEXT PRIMARY KEY,
        title       TEXT NOT NULL,
        source_kind TEXT,
        origin_ref  TEXT,
        path        TEXT NOT NULL,
        captured    TEXT,
        never_cloud INTEGER NOT NULL DEFAULT 0,
        hash        TEXT NOT NULL
    );

    CREATE TABLE excerpts (
        id         TEXT PRIMARY KEY,      -- src_<id>#exc_<n>
        source_id  TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
        start_offset INTEGER,
        end_offset   INTEGER,
        text       TEXT NOT NULL
    );

    CREATE TABLE notes (
        id          TEXT PRIMARY KEY,
        type        TEXT NOT NULL,
        status      TEXT NOT NULL,
        title       TEXT NOT NULL,
        path        TEXT NOT NULL,
        version     INTEGER NOT NULL,
        created_by  TEXT NOT NULL,
        created     TEXT NOT NULL,
        updated     TEXT,
        valid_from  TEXT,
        valid_to    TEXT,
        review_on   TEXT,
        hash        TEXT NOT NULL
    );
    CREATE INDEX notes_type_status ON notes(type, status);

    CREATE TABLE note_versions (
        note_id  TEXT NOT NULL,
        version  INTEGER NOT NULL,
        status   TEXT NOT NULL,
        title    TEXT NOT NULL,
        path     TEXT NOT NULL,
        updated  TEXT,
        PRIMARY KEY (note_id, version)
    );

    CREATE TABLE edges (
        src   TEXT NOT NULL,
        dst   TEXT NOT NULL,
        kind  TEXT NOT NULL,  -- links | derived_from | supports | contradicts | refines
        PRIMARY KEY (src, dst, kind)
    );
    CREATE INDEX edges_dst ON edges(dst);

    CREATE VIRTUAL TABLE notes_fts USING fts5(
        note_id UNINDEXED, title, body, applies_when,
        tokenize = 'unicode61 remove_diacritics 2'
    );
    CREATE VIRTUAL TABLE excerpts_fts USING fts5(
        excerpt_id UNINDEXED, text,
        tokenize = 'unicode61 remove_diacritics 2'
    );

    -- Operational (not rebuilt from the vault)
    CREATE TABLE suggestions (
        id          TEXT PRIMARY KEY,
        op          TEXT NOT NULL,          -- add | update | link | invalidate | noop
        relation    TEXT,                   -- new | confirms | refines | contradicts | extends
        target_id   TEXT,
        payload     TEXT NOT NULL,          -- JSON
        status      TEXT NOT NULL DEFAULT 'pending',  -- pending | accepted | edited | rejected
        created     TEXT NOT NULL,
        reviewed    TEXT,
        edit_distance INTEGER
    );

    CREATE TABLE events (
        id      INTEGER PRIMARY KEY AUTOINCREMENT,
        at      TEXT NOT NULL,
        kind    TEXT NOT NULL,  -- search | mcp_read | digest_open | accept | edit | reject | reindex
        note_id TEXT,
        detail  TEXT
    );

    CREATE TABLE llm_calls (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        at            TEXT NOT NULL,
        provider      TEXT NOT NULL,
        model         TEXT NOT NULL,
        purpose       TEXT NOT NULL,  -- triage | extract | reconcile | consolidate | digest
        input_tokens  INTEGER,
        output_tokens INTEGER,
        cost_usd      REAL,
        source_ids    TEXT,           -- JSON list
        redactions    INTEGER
    );
    """,
    # v2: distillation progress and an embedding cache (both operational: costly to recompute)
    """
    CREATE TABLE distill_state (
        source_id    TEXT PRIMARY KEY,
        content_hash TEXT NOT NULL,
        status       TEXT NOT NULL,   -- done | skipped | failed | never_cloud | budget
        triage       TEXT,
        candidates   INTEGER,
        suggestions  INTEGER,
        error        TEXT,
        updated      TEXT NOT NULL
    );

    CREATE TABLE embed_cache (
        hash   TEXT NOT NULL,
        model  TEXT NOT NULL,
        dim    INTEGER NOT NULL,
        vector BLOB NOT NULL,
        PRIMARY KEY (hash, model)
    );

    CREATE TABLE note_vectors (     -- derived: note id -> cache key, rebuilt with the index
        note_id TEXT PRIMARY KEY,
        hash    TEXT NOT NULL,
        model   TEXT NOT NULL
    );

    ALTER TABLE suggestions ADD COLUMN source_id TEXT;
    ALTER TABLE suggestions ADD COLUMN inbox_path TEXT;
    ALTER TABLE suggestions ADD COLUMN reason TEXT;
    """,
]


class IndexKeyError(RuntimeError):
    pass


def _sqlcipher():
    try:
        from sqlcipher3 import dbapi2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise IndexKeyError(
            "index encryption needs SQLCipher: pip install 'essence-engine[encrypt]'"
        ) from exc
    return dbapi2


def connect(path: Path, key: str | None = None) -> sqlite3.Connection:
    """Open (and migrate) the index. With `key`, the file is SQLCipher-encrypted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if key:
        dbapi = _sqlcipher()
        conn = dbapi.connect(str(path))
        conn.row_factory = dbapi.Row
        conn.execute("PRAGMA key = " + _quote(key))
        try:
            conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
        except dbapi.DatabaseError as exc:
            conn.close()
            raise IndexKeyError(
                f"cannot open {path.name}: wrong ESSENCE_DB_KEY, or the file is not encrypted "
                "(delete it and run `essence reindex` to rebuild)"
            ) from exc
    else:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
        except sqlite3.DatabaseError as exc:
            conn.close()
            raise IndexKeyError(
                f"cannot open {path.name}: it looks encrypted; set ESSENCE_DB_KEY and [index] encrypt = true"
            ) from exc
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    migrate(conn)
    return conn


def _quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def is_encrypted(path: Path) -> bool:
    """A plain SQLite file starts with a fixed header; an encrypted one doesn't."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    with path.open("rb") as f:
        return f.read(16) != b"SQLite format 3\x00"


@contextmanager
def open_index(path: Path, key: str | None = None) -> Iterator[sqlite3.Connection]:
    """Connect, and always close (sqlite3's own context manager only commits)."""
    conn = connect(path, key)
    try:
        yield conn
    finally:
        conn.close()


def schema_version(conn: sqlite3.Connection) -> int:
    return conn.execute("PRAGMA user_version").fetchone()[0]


def migrate(conn: sqlite3.Connection) -> None:
    current = schema_version(conn)
    for version, script in enumerate(_MIGRATIONS[current:], start=current + 1):
        with conn:
            conn.executescript(script)
            conn.execute(f"PRAGMA user_version = {version}")
