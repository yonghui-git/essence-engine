"""Vault directory layout and initialization."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from essence.vault.model import KNOWLEDGE_TYPES, Note, NoteType

SCHEMA_TEMPLATE = """\
# Knowledge schema

This file tells the distillation pipeline what to keep and how to shape it.
Edit it to fit how you think; the AI reads it before every extraction.

## What counts as essence

Keep what you will reuse: principles and mental models, cause and effect,
decisions with reasons and outcomes, lessons from experience, open questions.
Leave out one-off facts, anything a search can find again, and verbatim text.
Details stay in the sources as evidence. The rule is reusability, not brevity.

## Note types

- **Concept**: an idea, defined once and reused.
- **Claim**: a statement, tied to its source excerpt. Has a confidence.
- **Insight**: your synthesis across sources. Versioned when it changes.
- **Decision**: what you chose and why, what you expected, and when to review it.
- **Question**: something still open.

Every note says when it applies (`applies_when`) and, if useful, when it does not (`not_when`).

## How new input relates to what you know

For each input, ask "what is new here relative to what I already know?" and tag
each candidate as one of: new, confirms, refines, contradicts, extends.
Contradicted notes are marked superseded or invalid, never deleted.
"""

GITIGNORE = ".essence/\n"

_DIRS = [
    "raw/chats",
    "raw/web",
    "raw/docs",
    *[t.folder for t in KNOWLEDGE_TYPES],
    "notes/_history",
    "inbox",
    "digests",
    ".essence",
]


@dataclass(frozen=True)
class Vault:
    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def notes(self) -> Path:
        return self.root / "notes"

    @property
    def history(self) -> Path:
        return self.root / "notes" / "_history"

    @property
    def inbox(self) -> Path:
        return self.root / "inbox"

    @property
    def state(self) -> Path:
        return self.root / ".essence"

    @property
    def index_path(self) -> Path:
        return self.state / "index.db"

    @property
    def schema_path(self) -> Path:
        return self.root / "schema.md"

    @property
    def config_path(self) -> Path:
        return self.state / "config.toml"

    @property
    def digests(self) -> Path:
        return self.root / "digests"

    def config(self):
        from essence.config import load_config

        return load_config(self.config_path)

    def relpath(self, path: Path) -> str:
        return path.resolve().relative_to(self.root).as_posix()

    def is_initialized(self) -> bool:
        return self.state.is_dir() and self.notes.is_dir()

    def path_for(self, note: Note) -> Path:
        """Default file path for a new note: <folder>/<slug>-<short id>.md."""
        short = note.id.split("_", 1)[-1][-6:].lower()
        return self.root / note.type.folder / f"{slugify(note.title)}-{short}.md"

    def history_path(self, note_id: str, version: int) -> Path:
        return self.history / note_id / f"v{version}.md"

    def note_files(self) -> list[Path]:
        """Every canonical note file (sources in raw/, knowledge in notes/), excluding history."""
        files: list[Path] = []
        if self.raw.is_dir():
            originals = self.raw / "_originals"
            files += sorted(p for p in self.raw.rglob("*.md") if not p.is_relative_to(originals))
        for t in KNOWLEDGE_TYPES:
            folder = self.root / t.folder
            if folder.is_dir():
                files += sorted(folder.rglob("*.md"))
        return files

    def history_files(self) -> list[Path]:
        return sorted(self.history.rglob("v*.md")) if self.history.is_dir() else []


def init_vault(root: Path) -> tuple[Vault, list[Path]]:
    """Create the vault layout. Idempotent: never overwrites existing files.

    Returns the vault and the paths that were created.
    """
    vault = Vault(root.resolve())
    created: list[Path] = []
    for rel in _DIRS:
        d = vault.root / rel
        if not d.exists():
            d.mkdir(parents=True)
            created.append(d)
    from essence.config import DEFAULT_CONFIG

    for path, content in [
        (vault.schema_path, SCHEMA_TEMPLATE),
        (vault.root / ".gitignore", GITIGNORE),
        (vault.config_path, DEFAULT_CONFIG),
    ]:
        if not path.exists():
            path.write_text(content, encoding="utf-8", newline="\n")
            created.append(path)
    return vault, created


_CJK_SLUG = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]")


def slugify(text: str, max_len: int = 60) -> str:
    """File-name slug: accents stripped from Latin text; Chinese, Japanese and Korean kept as is."""
    out: list[str] = []
    for ch in text:
        if _CJK_SLUG.match(ch):
            out.append(ch)
            continue
        base = unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode().lower()
        out.append(base if base.isalnum() else "-")
    slug = re.sub(r"-+", "-", "".join(out)).strip("-")
    if _CJK_SLUG.search(slug):
        max_len = min(max_len, 32)  # each CJK character carries a word; keep paths short
    return slug[:max_len].rstrip("-") or "note"


def note_type_from_str(value: str) -> NoteType:
    try:
        return NoteType(value.lower())
    except ValueError as exc:
        options = ", ".join(t.value for t in KNOWLEDGE_TYPES)
        raise ValueError(f"unknown note type {value!r}; expected one of: {options}") from exc
