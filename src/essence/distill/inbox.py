"""The review inbox: suggested changes, one editable Markdown file each.

An inbox file is a normal note file whose frontmatter also carries a
`suggestion` block (op, target, reason, source). People can review in the CLI
or by editing the files directly (in Obsidian, say); the file on disk is
what gets applied. The `suggestions` table mirrors each item so accept / edit
/ reject rates can be measured.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from essence.ids import new_id
from essence.vault.io import NoteParseError, read_note, write_note
from essence.vault.layout import Vault, slugify
from essence.vault.model import Note, NoteStatus, Origin

OPS = ("add", "update", "confirm", "supersede", "link", "merge", "orphan")


@dataclass
class Suggestion:
    id: str
    op: str
    note: Note
    path: Path
    target_id: str | None
    link_ids: list[str]
    relation: str | None
    reason: str
    source_id: str | None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _payload(note: Note) -> str:
    return json.dumps(
        {"type": note.type.value, "title": note.title, "body": note.body, "applies_when": note.applies_when},
        ensure_ascii=False,
    )


def create_suggestion(
    vault: Vault,
    conn: sqlite3.Connection,
    *,
    op: str,
    note: Note,
    target_id: str | None = None,
    link_ids: list[str] | None = None,
    relation: str | None = None,
    reason: str = "",
    source_id: str | None = None,
) -> Suggestion:
    if op not in OPS:
        raise ValueError(f"unknown suggestion op {op!r}")
    sug_id = new_id("sug")
    meta = {
        "id": sug_id,
        "op": op,
        "relation": relation,
        "target": target_id,
        "links": list(link_ids or []),
        "reason": reason,
        "source": source_id,
        "created": _now(),
    }
    proposal = note.model_copy(update={"status": NoteStatus.SUGGESTED, "extra": {**note.extra, "suggestion": meta}})
    short = sug_id[-6:].lower()
    path = vault.inbox / f"{op}-{slugify(note.title, 50)}-{short}.md"
    write_note(proposal, path)
    with conn:
        conn.execute(
            "INSERT INTO suggestions (id, op, relation, target_id, payload, status, created, source_id, inbox_path, reason)"
            " VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?)",
            (sug_id, op, relation, target_id, _payload(note), meta["created"], source_id, vault.relpath(path), reason),
        )
    return Suggestion(sug_id, op, proposal, path, target_id, meta["links"], relation, reason, source_id)


def read_suggestion(path: Path) -> Suggestion:
    note = read_note(path)
    meta = note.extra.get("suggestion")
    if not isinstance(meta, dict) or "id" not in meta or meta.get("op") not in OPS:
        raise NoteParseError("inbox file has no valid `suggestion` block", path)
    return Suggestion(
        id=str(meta["id"]),
        op=meta["op"],
        note=note,
        path=path,
        target_id=meta.get("target") or None,
        link_ids=[str(x) for x in meta.get("links") or []],
        relation=meta.get("relation"),
        reason=meta.get("reason") or "",
        source_id=meta.get("source") or None,
    )


def pending(vault: Vault) -> tuple[list[Suggestion], list[str]]:
    """All inbox items (oldest first) and any files that could not be read."""
    items, errors = [], []
    for path in sorted(vault.inbox.glob("*.md")) if vault.inbox.is_dir() else []:
        try:
            items.append(read_suggestion(path))
        except NoteParseError as exc:
            errors.append(str(exc))
    items.sort(key=lambda s: s.note.extra.get("suggestion", {}).get("created") or "")
    return items, errors


def find(vault: Vault, ref: str) -> Suggestion | None:
    """Find by suggestion id, a unique id suffix, a file name, or a path such as `inbox/<file>.md`."""
    items, _ = pending(vault)
    ref_l = ref.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].lower()  # `essence inbox` prints paths
    exact = [s for s in items if s.id.lower() == ref_l or s.path.name.lower() == ref_l]
    if exact:
        return exact[0]
    partial = [s for s in items if s.id.lower().endswith(ref_l) or ref_l in s.path.name.lower()]
    return partial[0] if len(partial) == 1 else None


def clean_note(note: Note) -> Note:
    """Strip the suggestion block from a note about to become canonical."""
    extra = {k: v for k, v in note.extra.items() if k != "suggestion"}
    return note.model_copy(update={"extra": extra})


def origin_after_review(original_payload: str | None, note: Note) -> tuple[Origin, int]:
    """Was the proposal edited by the person? Returns (origin, approximate characters changed)."""
    from difflib import SequenceMatcher

    if not original_payload:
        return Origin.AI, 0
    try:
        orig = json.loads(original_payload)
    except json.JSONDecodeError:
        return Origin.AI, 0
    before = f"{orig.get('title', '')}\n{orig.get('applies_when') or ''}\n{orig.get('body', '')}".strip()
    after = f"{note.title}\n{note.applies_when or ''}\n{note.body}".strip()
    if before == after:
        return Origin.AI, 0
    ratio = SequenceMatcher(None, before, after).ratio()
    return Origin.AI_EDITED, max(1, round((1 - ratio) * max(len(before), len(after))))
