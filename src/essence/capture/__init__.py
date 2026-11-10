from pathlib import Path

from essence.capture.base import CapturedItem, ImportResult, SourceStore
from essence.capture.chats import detect_provider, parse_chatgpt, parse_claude
from essence.capture.files import iter_documents, iter_web_clips

__all__ = ["CapturedItem", "ImportResult", "SourceStore", "import_path"]

KINDS = ("chatgpt", "claude", "chat", "web", "docs")


def import_path(store: SourceStore, kind: str, path: Path, use_docling: bool = True) -> ImportResult:
    """Import from `path`. kind: chatgpt | claude | chat (auto-detect) | web | docs."""
    if not path.exists():
        raise FileNotFoundError(path)
    if kind == "chat":
        kind = detect_provider(path)
    if kind == "chatgpt":
        items = parse_chatgpt(path)
    elif kind == "claude":
        items = parse_claude(path)
    elif kind == "web":
        items = iter_web_clips(path) if path.is_dir() else iter_web_clips_file(path)
    elif kind == "docs":
        items = iter_documents(path, use_docling)
    else:
        raise ValueError(f"unknown import kind {kind!r}; expected one of {', '.join(KINDS)}")
    return store.store_all(items)


def iter_web_clips_file(path: Path):
    from essence.capture.files import parse_web_clip

    yield parse_web_clip(path)
