"""Read and write notes as Markdown with YAML frontmatter."""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from essence.vault.model import Note

_FRONTMATTER = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n?(.*)\Z", re.DOTALL)
_KNOWN = set(Note.model_fields) - {"body", "extra"}


class NoteParseError(ValueError):
    def __init__(self, message: str, path: Path | None = None):
        self.path = path
        super().__init__(f"{path}: {message}" if path else message)


def _plain(value: Any) -> Any:
    """Turn YAML-native dates in unknown keys into strings so they survive round-trips."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


def parse_note(text: str, path: Path | None = None) -> Note:
    match = _FRONTMATTER.match(text.lstrip("﻿"))
    if not match:
        raise NoteParseError("missing YAML frontmatter", path)
    raw_meta, body = match.groups()
    try:
        meta = yaml.safe_load(raw_meta) or {}
    except yaml.YAMLError as exc:
        raise NoteParseError(f"invalid YAML: {exc}", path) from exc
    if not isinstance(meta, dict):
        raise NoteParseError("frontmatter is not a mapping", path)

    known = {k: v for k, v in meta.items() if k in _KNOWN}
    extra = {k: _plain(v) for k, v in meta.items() if k not in _KNOWN}
    try:
        return Note(**known, body=body.replace("\r\n", "\n"), extra=extra)
    except ValidationError as exc:
        raise NoteParseError(str(exc), path) from exc


def render_note(note: Note) -> str:
    meta = yaml.safe_dump(
        note.frontmatter(), sort_keys=False, allow_unicode=True, default_flow_style=False, width=1000
    )
    body = note.body
    if body and not body.endswith("\n"):
        body += "\n"
    return f"---\n{meta}---\n{body}"


def read_note(path: Path) -> Note:
    return parse_note(path.read_text(encoding="utf-8"), path)


def write_note(note: Note, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(render_note(note), encoding="utf-8", newline="\n")
    tmp.replace(path)
