"""Decision support: decisions carry an expected outcome and a review date. When the date
comes, the person records what actually happened and the lesson; the earlier version is kept,
so the gap between expectation and outcome stays visible."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone

from essence.distill.apply import archive_version
from essence.vault.io import write_note
from essence.vault.layout import Vault
from essence.vault.model import NoteStatus, NoteType
from essence.vault.repo import NoteRepo


class DecisionError(ValueError):
    pass


@dataclass
class DecisionRow:
    id: str
    title: str
    review_on: str | None
    state: str  # due | upcoming | reviewed | open


def list_decisions(conn: sqlite3.Connection, vault: Vault, *, today: date | None = None) -> list[DecisionRow]:
    """Active decisions, those due for review first."""
    today = today or date.today()
    repo = NoteRepo(vault)
    rows = conn.execute(
        "SELECT id, title, review_on FROM notes WHERE type = 'decision' AND status = 'active'"
        " ORDER BY review_on IS NULL, review_on, title"
    ).fetchall()
    out = []
    for r in rows:
        found = repo.get(r["id"])
        reviewed = bool(found and found[1].actual_outcome)
        if r["review_on"] and r["review_on"] <= today.isoformat():
            state = "due"
        elif r["review_on"]:
            state = "upcoming"
        else:
            state = "reviewed" if reviewed else "open"
        out.append(DecisionRow(r["id"], r["title"], r["review_on"], state))
    order = {"due": 0, "upcoming": 1, "open": 2, "reviewed": 3}
    return sorted(out, key=lambda d: (order[d.state], d.review_on or "9999"))


def record_outcome(
    vault: Vault,
    note_id: str,
    *,
    actual: str,
    lesson: str = "",
    next_review: date | None = None,
    repo: NoteRepo | None = None,
):
    """Write the actual outcome and lesson as a new version of the decision."""
    repo = repo or NoteRepo(vault)
    found = repo.get(note_id)
    if not found:
        raise DecisionError(f"no note {note_id}")
    path, note = found
    if note.type is not NoteType.DECISION:
        raise DecisionError(f"{note_id} is a {note.type.value}, not a decision")
    if note.status is not NoteStatus.ACTIVE:
        raise DecisionError(f"{note_id} is {note.status.value}; record outcomes on the active decision")
    if not actual.strip():
        raise DecisionError("the actual outcome is required")
    archive_version(vault, note)
    updated = note.model_copy(update={
        "actual_outcome": actual.strip(),
        "lesson": lesson.strip() or note.lesson,
        "review_on": next_review,
        "version": note.version + 1,
        "updated": datetime.now(timezone.utc).replace(microsecond=0),
        "change_reason": "outcome recorded",
    })
    write_note(updated, path)
    return updated
