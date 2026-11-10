"""Find notes by id on disk."""

from __future__ import annotations

from pathlib import Path

from essence.vault.io import NoteParseError, read_note
from essence.vault.layout import Vault
from essence.vault.model import Note


class NoteRepo:
    def __init__(self, vault: Vault):
        self.vault = vault
        self._paths: dict[str, Path] | None = None

    def _scan(self) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        for path in self.vault.note_files():
            try:
                paths.setdefault(read_note(path).id, path)
            except NoteParseError:
                continue
        return paths

    def refresh(self) -> None:
        self._paths = None

    def path(self, note_id: str) -> Path | None:
        if self._paths is None:
            self._paths = self._scan()
        p = self._paths.get(note_id)
        if p is not None and not p.exists():
            self._paths = self._scan()
            p = self._paths.get(note_id)
        return p

    def get(self, note_id: str) -> tuple[Path, Note] | None:
        p = self.path(note_id)
        if p is None:
            return None
        return p, read_note(p)

    def remember(self, note_id: str, path: Path) -> None:
        if self._paths is None:
            self._paths = self._scan()
        self._paths[note_id] = path

    def forget(self, note_id: str) -> None:
        if self._paths is not None:
            self._paths.pop(note_id, None)
