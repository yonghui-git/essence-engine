"""Watch a web-clip or document folder and import what appears or changes.

Polling, not OS file events: it works the same on Windows, macOS, Linux and synced folders
(Dropbox, iCloud, Syncthing), needs no dependency, and a personal folder is small. A file is
imported only once it has stopped changing for one interval, so a half-written clip from the
Obsidian Web Clipper or a sync client is not read early. Re-imports are cheap and safe: the
source store recognises unchanged content.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from essence.capture.base import ImportResult, SourceStore
from essence.capture.files import DOC_SUFFIXES, iter_documents, parse_web_clip

WATCH_KINDS = ("web", "docs")
Signature = tuple[int, int]  # (mtime_ns, size)


def _suffixes(kind: str) -> set[str]:
    if kind == "web":
        return {".md"}
    if kind == "docs":
        return DOC_SUFFIXES
    raise ValueError(f"can only watch {' or '.join(WATCH_KINDS)} folders, not {kind!r}")


def snapshot(folder: Path, kind: str, skip: Path | None = None) -> dict[Path, Signature]:
    """Signature of every importable file under `folder` (hidden folders and `skip` excluded)."""
    suffixes = _suffixes(kind)
    out: dict[Path, Signature] = {}
    for p in folder.rglob("*"):
        rel = p.relative_to(folder).parts
        if any(part.startswith(".") for part in rel) or p.suffix.lower() not in suffixes or not p.is_file():
            continue
        if skip is not None and (p == skip or skip in p.parents):
            continue
        try:
            st = p.stat()
        except OSError:
            continue  # deleted between listing and stat
        out[p] = (st.st_mtime_ns, st.st_size)
    return out


@dataclass
class FolderWatcher:
    store: SourceStore
    kind: str
    folder: Path
    use_docling: bool = True
    seen: dict[Path, Signature] = field(default_factory=dict)  # imported at this signature
    _last: dict[Path, Signature] = field(default_factory=dict)  # signature at the previous poll

    def __post_init__(self) -> None:
        _suffixes(self.kind)  # validate early
        if not self.folder.is_dir():
            raise FileNotFoundError(self.folder)
        # Never import the vault into itself if the folder contains it.
        self._skip = self.store.vault.root if self.store.vault.root.is_relative_to(self.folder) else None

    def poll(self) -> tuple[ImportResult, list[Path]]:
        """Import files that are new or changed and have been stable since the last poll."""
        current = snapshot(self.folder, self.kind, self._skip)
        ready = [p for p, sig in current.items() if self._last.get(p) == sig and self.seen.get(p) != sig]
        self._last = current
        result = ImportResult()
        for path in sorted(ready):
            items = iter_documents(path, self.use_docling) if self.kind == "docs" else [self._clip(path)]
            result.merge(self.store.store_all(items))
            self.seen[path] = current[path]
        return result, ready

    @staticmethod
    def _clip(path: Path):
        try:
            return parse_web_clip(path)
        except (OSError, UnicodeDecodeError) as exc:
            return ValueError(f"{path.name}: {exc}")


def watch(
    watcher: FolderWatcher,
    *,
    interval: float = 30.0,
    on_batch: Callable[[ImportResult, list[Path]], None] | None = None,
    polls: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Poll until interrupted (or `polls` times, for tests)."""
    n = 0
    while polls is None or n < polls:
        result, files = watcher.poll()
        if files and on_batch:
            on_batch(result, files)
        n += 1
        if polls is None or n < polls:
            sleep(interval)
