import os

import pytest

from essence.capture.base import SourceStore
from essence.capture.watch import FolderWatcher, watch
from essence.config import Config

CLIP = "---\ntitle: {t}\nsource: https://example.com/{t}\n---\n\n{body}\n"


def _write(path, title, body="Spaced repetition beats rereading for long-term memory."):
    path.write_text(CLIP.format(t=title, body=body), encoding="utf-8")


def _bump(path):
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))


def test_imports_only_after_a_file_settles_and_only_once(vault, tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    _write(clips / "a.md", "a")
    w = FolderWatcher(SourceStore(vault, Config()), "web", clips)
    result, files = w.poll()
    assert files == []  # first sight: not yet known to be stable
    result, files = w.poll()
    assert [f.name for f in files] == ["a.md"] and len(result.new) == 1
    assert w.poll()[1] == []  # unchanged: nothing to do

    _write(clips / "a.md", "a", "Spaced repetition beats rereading, and testing beats both.")
    _bump(clips / "a.md")
    assert w.poll()[1] == []  # changed since last poll: wait
    result, files = w.poll()
    assert len(result.updated) == 1  # same clip URL -> same source, updated


def test_skips_hidden_folders_wrong_types_and_the_vault(vault, tmp_path):
    (tmp_path / ".obsidian").mkdir()
    _write(tmp_path / ".obsidian" / "x.md", "hidden")
    (tmp_path / "notes.txt").write_text("not a clip", encoding="utf-8")
    _write(tmp_path / "real.md", "real")
    w = FolderWatcher(SourceStore(vault, Config()), "web", tmp_path)  # the vault lives under tmp_path
    batches = []
    watch(w, polls=2, sleep=lambda s: None, on_batch=lambda r, f: batches.append([p.name for p in f]))
    assert batches == [["real.md"]]


def test_rejects_unwatchable_kinds(vault, tmp_path):
    with pytest.raises(ValueError):
        FolderWatcher(SourceStore(vault, Config()), "chatgpt", tmp_path)
    with pytest.raises(FileNotFoundError):
        FolderWatcher(SourceStore(vault, Config()), "web", tmp_path / "missing")
