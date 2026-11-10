"""Shared capture machinery: turn a captured item into an immutable raw Source note.

Re-importing the same item (same kind + origin) is idempotent: unchanged
content is skipped; changed content (a chat that grew) rewrites the raw file
under the same source id so provenance links stay valid.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from essence.config import Config
from essence.vault.io import NoteParseError, read_note, write_note
from essence.vault.layout import Vault, slugify
from essence.vault.model import Note, NoteType, Origin


@dataclass
class CapturedItem:
    source_kind: str  # chat | web | doc
    origin_ref: str  # stable identity: conversation id, URL, file path
    title: str
    body: str
    captured: datetime | None = None
    subfolder: str = ""  # e.g. "chatgpt" under raw/chats/
    extra: dict = field(default_factory=dict)
    original_file: Path | None = None  # copied into raw/_originals/

    def content_hash(self) -> str:
        return hashlib.sha256(self.body.encode("utf-8")).hexdigest()[:16]


@dataclass
class ImportResult:
    new: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def merge(self, other: ImportResult) -> None:
        self.new += other.new
        self.updated += other.updated
        self.unchanged += other.unchanged
        self.errors += other.errors

    def summary(self) -> str:
        return (
            f"{len(self.new)} new, {len(self.updated)} updated, "
            f"{len(self.unchanged)} unchanged, {len(self.errors)} errors"
        )


_FOLDER = {"chat": "chats", "web": "web", "doc": "docs"}


class SourceStore:
    """Index of existing raw sources by (kind, origin_ref), built once per import run."""

    def __init__(self, vault: Vault, config: Config | None = None):
        self.vault = vault
        self.config = config or Config()
        self._by_origin: dict[tuple[str, str], tuple[Path, Note]] = {}
        for path in vault.note_files():
            if not path.is_relative_to(vault.raw):
                continue
            try:
                note = read_note(path)
            except NoteParseError:
                continue
            if note.type is NoteType.SOURCE and note.origin_ref:
                self._by_origin[(note.source_kind or "", note.origin_ref)] = (path, note)

    def find(self, kind: str, origin_ref: str) -> Note | None:
        hit = self._by_origin.get((kind, origin_ref))
        return hit[1] if hit else None

    def store(self, item: CapturedItem) -> tuple[str, Note]:
        """Write the item. Returns (status, note) with status new | updated | unchanged."""
        if not item.body.strip():
            raise ValueError(f"empty content: {item.origin_ref}")
        digest = item.content_hash()
        key = (item.source_kind, item.origin_ref)
        existing = self._by_origin.get(key)
        now = datetime.now(timezone.utc).replace(microsecond=0)

        if existing:
            path, old = existing
            if old.extra.get("content_hash") == digest:
                return "unchanged", old
            note = old.model_copy(
                update={
                    "title": item.title or old.title,
                    "body": item.body,
                    "updated": now,
                    "extra": {**old.extra, **item.extra, "content_hash": digest},
                }
            )
            write_note(note, path)
            self._by_origin[key] = (path, note)
            return "updated", note

        note = Note(
            type=NoteType.SOURCE,
            title=item.title or item.origin_ref,
            source_kind=item.source_kind,
            origin_ref=item.origin_ref,
            captured=item.captured or now,
            created_by=Origin.USER,
            never_cloud=True if self.config.is_never_cloud(item.origin_ref, item.title) else None,
            body=item.body,
            extra={**item.extra, "content_hash": digest},
        )
        folder = self.vault.raw / _FOLDER[item.source_kind]
        if item.subfolder:
            folder = folder / item.subfolder
        short = note.id.split("_", 1)[-1][-6:].lower()
        path = folder / f"{slugify(note.title)}-{short}.md"
        if item.original_file is not None:
            dest = self.vault.raw / "_originals" / f"{short}-{item.original_file.name}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item.original_file, dest)
            note.extra["original"] = dest.relative_to(self.vault.root).as_posix()
        write_note(note, path)
        self._by_origin[key] = (path, note)
        return "new", note

    def store_all(self, items) -> ImportResult:
        result = ImportResult()
        for item in items:
            if isinstance(item, Exception):
                result.errors.append(str(item))
                continue
            try:
                status, note = self.store(item)
            except (ValueError, OSError) as exc:
                result.errors.append(f"{item.origin_ref}: {exc}")
                continue
            getattr(result, status).append(note.id)
        return result
