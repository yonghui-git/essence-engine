"""Maintenance operations: propagating deletion, index encryption, metrics and cost reports."""

from __future__ import annotations

import shutil
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from essence.distill.inbox import create_suggestion, pending
from essence.llm.cost import cost_usd
from essence.vault.io import NoteParseError, read_note, write_note
from essence.vault.layout import Vault
from essence.vault.model import NoteType, Origin


# --- deletion --------------------------------------------------------------------------------


@dataclass
class DeleteReport:
    source_id: str
    raw_files: list[str] = field(default_factory=list)
    notes_updated: list[str] = field(default_factory=list)
    orphans: list[str] = field(default_factory=list)
    suggestions_removed: list[str] = field(default_factory=list)


def delete_source(vault: Vault, conn: sqlite3.Connection, source_id: str) -> DeleteReport:
    """Delete a source everywhere it reached.

    - the raw file and its original copy are removed;
    - every note citing it loses those citations (in all versions under _history);
    - an AI-made note left with no sources moves to the inbox as an `orphan`
      (accept = keep it, reject = delete it);
    - pending suggestions derived from it are discarded;
    - its distillation record is dropped. The outbound-call log keeps only ids.
    """
    report = DeleteReport(source_id)
    prefix = f"{source_id}#"

    for path in vault.note_files():
        try:
            note = read_note(path)
        except NoteParseError:
            continue
        if note.id == source_id and note.type is NoteType.SOURCE:
            original = note.extra.get("original")
            if original:
                (vault.root / original).unlink(missing_ok=True)
            path.unlink()
            report.raw_files.append(vault.relpath(path.parent) + "/" + path.name)
            continue
        kept = [s for s in note.sources if s != source_id and not s.startswith(prefix)]
        if len(kept) == len(note.sources):
            continue
        note = note.model_copy(update={"sources": kept})
        if not kept and note.created_by is not Origin.USER:
            create_suggestion(
                vault, conn, op="orphan", note=note, target_id=note.id,
                reason=f"Its only source ({source_id}) was deleted. Keep the note, or reject to delete it.",
            )
            path.unlink()
            report.orphans.append(note.id)
        else:
            write_note(note, path)
            report.notes_updated.append(note.id)

    for path in vault.history_files():
        try:
            old = read_note(path)
        except NoteParseError:
            continue
        kept = [s for s in old.sources if s != source_id and not s.startswith(prefix)]
        if len(kept) != len(old.sources):
            write_note(old.model_copy(update={"sources": kept}), path)

    items, _ = pending(vault)
    for sug in items:
        if sug.source_id == source_id:
            sug.path.unlink(missing_ok=True)
            report.suggestions_removed.append(sug.id)
    with conn:
        if report.suggestions_removed:
            conn.executemany(
                "UPDATE suggestions SET status = 'discarded', payload = '{}' WHERE id = ?",
                [(i,) for i in report.suggestions_removed],
            )
        conn.execute("UPDATE suggestions SET payload = '{}' WHERE source_id = ?", (source_id,))
        conn.execute("DELETE FROM distill_state WHERE source_id = ?", (source_id,))
        conn.execute(
            "INSERT INTO events (at, kind, detail) VALUES (?, 'delete', ?)",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"), source_id),
        )
    return report


# --- encryption ------------------------------------------------------------------------------


def encrypt_index(path: Path, key: str) -> None:
    """Convert a plain index.db into a SQLCipher-encrypted one in place (operational tables kept)."""
    from sqlcipher3 import dbapi2  # type: ignore[import-not-found]

    from essence.index.db import _quote, is_encrypted

    if is_encrypted(path):
        raise ValueError("index is already encrypted")
    tmp = path.with_name(path.name + ".enc")
    tmp.unlink(missing_ok=True)
    src = dbapi2.connect(str(path))
    try:
        src.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        src.execute(f"ATTACH DATABASE {_quote(str(tmp))} AS enc KEY {_quote(key)}")
        src.execute("SELECT sqlcipher_export('enc')")
        version = src.execute("PRAGMA main.user_version").fetchone()[0]
        src.execute(f"PRAGMA enc.user_version = {int(version)}")
        src.execute("DETACH DATABASE enc")
    finally:
        src.close()
    for suffix in ("-wal", "-shm"):
        path.with_name(path.name + suffix).unlink(missing_ok=True)
    backup = path.with_name(path.name + ".plain-backup")
    shutil.move(path, backup)
    shutil.move(tmp, path)
    backup.unlink()


# --- metrics ---------------------------------------------------------------------------------


@dataclass
class Metrics:
    reviewed: int
    accepted: int
    edited: int
    rejected: int
    pending: int
    mean_edit_chars: float
    active_days_28: int
    reuse_events_7: int
    notes_active: int
    notes_superseded: int
    sources: int

    def share(self, n: int) -> str:
        return f"{100 * n / self.reviewed:.0f}%" if self.reviewed else "-"


def metrics(conn: sqlite3.Connection) -> Metrics:
    by = dict(conn.execute("SELECT status, COUNT(*) FROM suggestions GROUP BY status").fetchall())
    reviewed = sum(by.get(s, 0) for s in ("accepted", "edited", "rejected"))
    mean_edit = conn.execute("SELECT AVG(edit_distance) FROM suggestions WHERE status = 'edited'").fetchone()[0] or 0.0
    since28 = (date.today() - timedelta(days=27)).isoformat()
    since7 = (date.today() - timedelta(days=6)).isoformat()
    active_days = conn.execute(
        "SELECT COUNT(DISTINCT substr(at, 1, 10)) FROM events WHERE substr(at, 1, 10) >= ?"
        " AND kind IN ('search', 'mcp_search', 'mcp_read', 'accept', 'edit', 'reject', 'digest_open')",
        (since28,),
    ).fetchone()[0]
    reuse = conn.execute(
        "SELECT COUNT(*) FROM events WHERE substr(at, 1, 10) >= ? AND kind IN ('search', 'mcp_search', 'mcp_read', 'digest_open')",
        (since7,),
    ).fetchone()[0]
    status = dict(conn.execute("SELECT status, COUNT(*) FROM notes GROUP BY status").fetchall())
    return Metrics(
        reviewed=reviewed,
        accepted=by.get("accepted", 0),
        edited=by.get("edited", 0),
        rejected=by.get("rejected", 0),
        pending=by.get("pending", 0),
        mean_edit_chars=float(mean_edit),
        active_days_28=active_days,
        reuse_events_7=reuse,
        notes_active=status.get("active", 0),
        notes_superseded=status.get("superseded", 0),
        sources=conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0],
    )


@dataclass
class CostReport:
    rows: list[tuple[str, str, int, int, int, float]]  # purpose, model, calls, in, out, usd
    total_usd: float
    days: int
    sources_distilled: int

    @property
    def per_item(self) -> float:
        return self.total_usd / self.sources_distilled if self.sources_distilled else 0.0

    def monthly_at(self, items_per_day: int = 30) -> float:
        return self.per_item * items_per_day * 30


def cost_report(conn: sqlite3.Connection, days: int = 30) -> CostReport:
    since = (date.today() - timedelta(days=days - 1)).isoformat()
    rows = conn.execute(
        "SELECT purpose, model, COUNT(*), SUM(input_tokens), SUM(output_tokens), SUM(cost_usd)"
        " FROM llm_calls WHERE substr(at, 1, 10) >= ? GROUP BY purpose, model ORDER BY SUM(cost_usd) DESC",
        (since,),
    ).fetchall()
    out = [(r[0], r[1], r[2], r[3] or 0, r[4] or 0, r[5] or 0.0) for r in rows]
    distilled = conn.execute(
        "SELECT COUNT(*) FROM distill_state WHERE status = 'done' AND substr(updated, 1, 10) >= ?", (since,)
    ).fetchone()[0]
    return CostReport(out, sum(r[5] for r in out), days, distilled)


def estimate(model: str, input_tokens: int = 12_000, output_tokens: int = 2_000) -> float:
    return cost_usd(model, input_tokens, output_tokens)
