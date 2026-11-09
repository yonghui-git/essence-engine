"""Rebuild the derived index tables from the vault."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from essence.index.db import DERIVED_FTS, DERIVED_TABLES
from essence.vault.io import NoteParseError, read_note
from essence.vault.layout import Vault
from essence.vault.model import Note, NoteType


@dataclass
class ReindexReport:
    sources: int = 0
    notes: int = 0
    versions: int = 0
    edges: int = 0
    errors: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    dangling: list[str] = field(default_factory=list)


def _iso(value) -> str | None:
    return value.isoformat() if value is not None else None


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def reindex(vault: Vault, conn: sqlite3.Connection) -> ReindexReport:
    """Drop and rebuild every derived table from the Markdown vault.

    Unparseable files are reported, not fatal. Operational tables are untouched.
    """
    report = ReindexReport()
    seen: dict[str, Path] = {}
    parsed: list[tuple[Path, Note]] = []

    for path in vault.note_files():
        try:
            note = read_note(path)
        except NoteParseError as exc:
            report.errors.append(str(exc))
            continue
        if note.id in seen:
            report.duplicates.append(f"{note.id}: {_rel(vault, seen[note.id])} and {_rel(vault, path)}")
            continue
        seen[note.id] = path
        parsed.append((path, note))

    with conn:
        for table in DERIVED_FTS + DERIVED_TABLES:
            conn.execute(f"DELETE FROM {table}")

        for path, note in parsed:
            rel = _rel(vault, path)
            if note.type is NoteType.SOURCE:
                conn.execute(
                    "INSERT INTO sources (id, title, source_kind, origin_ref, path, captured, never_cloud, hash)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (note.id, note.title, note.source_kind, note.origin_ref, rel,
                     _iso(note.captured), int(bool(note.never_cloud)), _hash(path)),
                )
                report.sources += 1
                continue

            conn.execute(
                "INSERT INTO notes (id, type, status, title, path, version, created_by, created, updated,"
                " valid_from, valid_to, review_on, hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (note.id, note.type.value, note.status.value, note.title, rel, note.version,
                 note.created_by.value, _iso(note.created), _iso(note.updated), _iso(note.valid_from),
                 _iso(note.valid_to), _iso(note.review_on), _hash(path)),
            )
            conn.execute(
                "INSERT INTO notes_fts (note_id, title, body, applies_when) VALUES (?, ?, ?, ?)",
                (note.id, note.title, note.body, note.applies_when or ""),
            )
            report.notes += 1

            edges = {(note.id, dst, "links") for dst in note.links}
            edges |= {(note.id, ref.split("#", 1)[0], "derived_from") for ref in note.sources}
            conn.executemany("INSERT OR IGNORE INTO edges (src, dst, kind) VALUES (?, ?, ?)", sorted(edges))
            report.edges += len(edges)
            report.dangling += [f"{s} -> {d} ({k})" for s, d, k in sorted(edges) if d not in seen]

        for path in vault.history_files():
            try:
                old = read_note(path)
            except NoteParseError as exc:
                report.errors.append(str(exc))
                continue
            conn.execute(
                "INSERT OR REPLACE INTO note_versions (note_id, version, status, title, path, updated)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (old.id, old.version, old.status.value, old.title, _rel(vault, path), _iso(old.updated)),
            )
            report.versions += 1

        conn.execute(
            "INSERT INTO events (at, kind, detail) VALUES (?, 'reindex', ?)",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"),
             f"sources={report.sources} notes={report.notes} errors={len(report.errors)}"),
        )
    return report


def _rel(vault: Vault, path: Path) -> str:
    return path.resolve().relative_to(vault.root).as_posix()
