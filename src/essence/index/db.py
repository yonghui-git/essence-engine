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

DERIVED_TABLES = ["edges", "note_versions", "excerpts", "notes", "sources"]
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
]


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    migrate(conn)
    return conn


@contextmanager
def open_index(path: Path) -> Iterator[sqlite3.Connection]:
    """Connect, and always close (sqlite3's own context manager only commits)."""
    conn = connect(path)
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
