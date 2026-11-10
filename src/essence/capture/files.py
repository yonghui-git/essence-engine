"""Importers for web clips (Obsidian Web Clipper output) and documents."""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import yaml

from essence.capture.base import CapturedItem

_FM = re.compile(r"\A﻿?---\r?\n(.*?)\r?\n---\r?\n?(.*)\Z", re.DOTALL)

DOC_SUFFIXES = {".pdf", ".docx", ".md", ".markdown", ".txt", ".html", ".htm", ".pptx", ".xlsx"}


def _mtime(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).replace(microsecond=0)


def parse_web_clip(path: Path) -> CapturedItem:
    text = path.read_text(encoding="utf-8")
    meta: dict = {}
    body = text
    m = _FM.match(text)
    if m:
        try:
            loaded = yaml.safe_load(m.group(1)) or {}
            meta = loaded if isinstance(loaded, dict) else {}
        except yaml.YAMLError:
            meta = {}
        body = m.group(2)
    url = meta.get("source") or meta.get("url") or meta.get("link")
    extra = {k: str(meta[k]) for k in ("author", "published", "description", "site") if meta.get(k)}
    if isinstance(meta.get("tags"), list):
        extra["tags"] = [str(t) for t in meta["tags"]]
    return CapturedItem(
        source_kind="web",
        origin_ref=str(url) if url else f"file:{path.name}",
        title=str(meta.get("title") or path.stem),
        body=body.replace("\r\n", "\n"),
        captured=_mtime(path),
        extra=extra,
    )


def iter_web_clips(folder: Path) -> Iterator[CapturedItem | Exception]:
    for path in sorted(folder.rglob("*.md")):
        try:
            yield parse_web_clip(path)
        except (OSError, UnicodeDecodeError) as exc:
            yield ValueError(f"{path.name}: {exc}")


def _docling_markdown(path: Path) -> str | None:
    try:
        from docling.document_converter import DocumentConverter  # type: ignore[import-not-found]
    except ImportError:
        return None
    result = DocumentConverter().convert(str(path))
    return result.document.export_to_markdown()


def _fallback_markdown(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".md", ".markdown", ".txt"):
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ValueError("PDF import needs `pip install essence-engine[docs]` (docling) or pypdf") from exc
        reader = PdfReader(str(path))
        pages = [(p.extract_text() or "").strip() for p in reader.pages]
        return "\n\n".join(f"<!-- page {i} -->\n{t}" for i, t in enumerate(pages, 1) if t)
    if suffix == ".docx":
        try:
            import docx  # python-docx
        except ImportError as exc:
            raise ValueError("DOCX import needs docling or python-docx") from exc
        d = docx.Document(str(path))
        lines = []
        for p in d.paragraphs:
            if not p.text.strip():
                continue
            style = (p.style.name or "").lower() if p.style is not None else ""
            level = re.search(r"heading (\d)", style)
            lines.append(f"{'#' * int(level.group(1))} {p.text}" if level else p.text)
        return "\n\n".join(lines)
    if suffix in (".html", ".htm"):
        html = path.read_text(encoding="utf-8", errors="replace")
        html = re.sub(r"(?is)<(script|style).*?</\1>", "", html)
        return re.sub(r"\n{3,}", "\n\n", re.sub(r"<[^>]+>", "", html)).strip()
    raise ValueError(f"unsupported document type {suffix} (install docling for pptx/xlsx)")


def parse_document(path: Path, use_docling: bool = True) -> CapturedItem:
    markdown = _docling_markdown(path) if use_docling and path.suffix.lower() not in (".md", ".txt") else None
    parser = "docling" if markdown is not None else "builtin"
    if markdown is None:
        markdown = _fallback_markdown(path)
    title = path.stem.replace("_", " ").replace("-", " ").strip() or path.name
    first_heading = re.search(r"^# (.+)$", markdown, re.M)
    if first_heading:
        title = first_heading.group(1).strip()
    return CapturedItem(
        source_kind="doc",
        origin_ref=f"file:{path.resolve().as_posix()}",
        title=title,
        body=markdown,
        captured=_mtime(path),
        extra={"file_name": path.name, "parser": parser},
        original_file=path,
    )


def iter_documents(target: Path, use_docling: bool = True) -> Iterator[CapturedItem | Exception]:
    paths = [target] if target.is_file() else sorted(p for p in target.rglob("*") if p.suffix.lower() in DOC_SUFFIXES)
    for path in paths:
        try:
            yield parse_document(path, use_docling)
        except Exception as exc:  # noqa: BLE001 - report and continue
            yield ValueError(f"{path.name}: {exc}")
