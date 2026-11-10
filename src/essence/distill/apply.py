"""Accept or reject inbox suggestions. The only path by which AI output becomes canonical."""

from __future__ import annotations

import shutil
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone

from essence.distill.inbox import Suggestion, clean_note, origin_after_review
from essence.vault.io import write_note
from essence.vault.layout import Vault
from essence.vault.model import Note, NoteStatus, Origin
from essence.vault.repo import NoteRepo


class ApplyError(RuntimeError):
    pass


@dataclass
class ApplyResult:
    suggestion_id: str
    status: str  # accepted | edited | rejected
    changed: list[str]  # note ids written


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _merge(a: list[str], b: list[str]) -> list[str]:
    return list(dict.fromkeys([*a, *b]))


def archive_version(vault: Vault, note: Note) -> None:
    """Keep the current version under notes/_history/<id>/v<N>.md before changing it."""
    path = vault.history_path(note.id, note.version)
    if not path.exists():
        write_note(note, path)


def _record(conn: sqlite3.Connection, sug: Suggestion, status: str, edit_distance: int, note_id: str | None) -> None:
    now = _now().isoformat()
    with conn:
        conn.execute(
            "UPDATE suggestions SET status = ?, reviewed = ?, edit_distance = ? WHERE id = ?",
            (status, now, edit_distance, sug.id),
        )
        conn.execute(
            "INSERT INTO events (at, kind, note_id, detail) VALUES (?, ?, ?, ?)",
            (now, {"accepted": "accept", "edited": "edit", "rejected": "reject"}[status], note_id, f"{sug.op} {sug.id}"),
        )


def _target(repo: NoteRepo, sug: Suggestion, required: bool = True):
    if not sug.target_id:
        if required:
            raise ApplyError(f"{sug.op} suggestion {sug.id} has no target")
        return None
    hit = repo.get(sug.target_id)
    if hit is None and required:
        raise ApplyError(f"target note {sug.target_id} no longer exists")
    return hit


def accept(vault: Vault, conn: sqlite3.Connection, sug: Suggestion, repo: NoteRepo | None = None) -> ApplyResult:
    repo = repo or NoteRepo(vault)
    row = conn.execute("SELECT payload FROM suggestions WHERE id = ?", (sug.id,)).fetchone()
    origin, distance = origin_after_review(row["payload"] if row else None, sug.note)
    proposal = clean_note(sug.note)
    now = _now()
    changed: list[str] = []

    if sug.op == "add":
        note = proposal.model_copy(
            update={"status": NoteStatus.ACTIVE, "created_by": origin, "links": _merge(proposal.links, sug.link_ids)}
        )
        path = vault.path_for(note)
        write_note(note, path)
        repo.remember(note.id, path)
        changed.append(note.id)

    elif sug.op in ("update", "merge"):
        path, current = _target(repo, sug)
        archive_version(vault, current)
        updated = current.model_copy(
            update={
                "title": proposal.title,
                "body": proposal.body,
                "applies_when": proposal.applies_when,
                "not_when": proposal.not_when,
                "sources": _merge(current.sources, proposal.sources),
                "links": _merge(current.links, [*proposal.links, *(sug.link_ids if sug.op == "update" else [])]),
                "version": current.version + 1,
                "updated": now,
                "change_reason": sug.reason or None,
                "status": NoteStatus.ACTIVE,
            }
        )
        write_note(updated, path)
        changed.append(updated.id)
        if sug.op == "merge":
            for other_id in sug.link_ids:
                hit = repo.get(other_id)
                if hit is None:
                    continue
                opath, other = hit
                archive_version(vault, other)
                write_note(
                    other.model_copy(
                        update={
                            "status": NoteStatus.SUPERSEDED,
                            "superseded_by": updated.id,
                            "valid_to": date.today(),
                            "version": other.version + 1,
                            "updated": now,
                            "change_reason": f"merged into {updated.id}",
                        }
                    ),
                    opath,
                )
                changed.append(other.id)

    elif sug.op == "confirm":
        path, current = _target(repo, sug)
        write_note(current.model_copy(update={"sources": _merge(current.sources, proposal.sources), "updated": now}), path)
        changed.append(current.id)

    elif sug.op == "supersede":
        tpath, old = _target(repo, sug)
        new = proposal.model_copy(
            update={
                "status": NoteStatus.ACTIVE,
                "created_by": origin,
                "links": _merge(proposal.links, [old.id, *sug.link_ids]),
                "change_reason": sug.reason or None,
                "valid_from": proposal.valid_from or date.today(),
            }
        )
        npath = vault.path_for(new)
        write_note(new, npath)
        repo.remember(new.id, npath)
        archive_version(vault, old)
        write_note(
            old.model_copy(
                update={
                    "status": NoteStatus.SUPERSEDED,
                    "superseded_by": new.id,
                    "valid_to": date.today(),
                    "version": old.version + 1,
                    "updated": now,
                    "change_reason": sug.reason or f"superseded by {new.id}",
                }
            ),
            tpath,
        )
        changed += [new.id, old.id]

    elif sug.op == "link":
        path, current = _target(repo, sug)
        write_note(current.model_copy(update={"links": _merge(current.links, sug.link_ids), "updated": now}), path)
        changed.append(current.id)

    elif sug.op == "orphan":
        # Keep a note whose sources were deleted: restore it from the inbox to notes/.
        note = proposal.model_copy(update={"status": NoteStatus.ACTIVE, "updated": now})
        path = vault.path_for(note)
        write_note(note, path)
        repo.remember(note.id, path)
        changed.append(note.id)

    else:
        raise ApplyError(f"unknown op {sug.op}")

    sug.path.unlink(missing_ok=True)
    status = "edited" if origin is Origin.AI_EDITED and sug.op != "orphan" else "accepted"
    _record(conn, sug, status, distance, changed[0] if changed else None)
    return ApplyResult(sug.id, status, changed)


def reject(vault: Vault, conn: sqlite3.Connection, sug: Suggestion) -> ApplyResult:
    changed: list[str] = []
    if sug.op == "orphan":
        # Rejecting an orphan deletes the note for good, with its history.
        history = vault.history / sug.note.id
        if history.is_dir():
            shutil.rmtree(history)
        changed.append(sug.note.id)
    sug.path.unlink(missing_ok=True)
    _record(conn, sug, "rejected", 0, None)
    return ApplyResult(sug.id, "rejected", changed)
