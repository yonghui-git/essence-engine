"""Importers for official AI chat exports (ChatGPT and Claude).

Neither vendor documents its export schema, so both parsers are tolerant:
unknown fields are ignored, missing ones fall back, and a conversation that
cannot be parsed is reported instead of aborting the import.
Exports are the permitted route; nothing here scrapes a web UI.
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from essence.capture.base import CapturedItem


def _load_conversations(path: Path) -> list[dict[str, Any]]:
    """Accept an export .zip, an extracted folder, or conversations.json itself."""
    if path.is_file() and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if n.rsplit("/", 1)[-1] == "conversations.json"]
            if not names:
                raise ValueError(f"no conversations.json in {path.name}")
            data = json.loads(zf.read(names[0]).decode("utf-8"))
    else:
        file = path / "conversations.json" if path.is_dir() else path
        data = json.loads(file.read_text(encoding="utf-8"))
    if isinstance(data, dict):  # some variants wrap the list
        data = data.get("conversations") or data.get("items") or []
    if not isinstance(data, list):
        raise ValueError("conversations.json is not a list")
    return data


def _ts(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc).replace(microsecond=0)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).replace(microsecond=0)
    return None


def render_transcript(messages: list[tuple[str, str, datetime | None]]) -> str:
    parts = []
    for role, text, when in messages:
        stamp = f" · {when.strftime('%Y-%m-%d %H:%M')}" if when else ""
        parts.append(f"## {role.capitalize()}{stamp}\n\n{text.strip()}\n")
    return "\n".join(parts)


# --- ChatGPT -------------------------------------------------------------------------------


def _chatgpt_text(message: dict[str, Any]) -> str:
    content = message.get("content") or {}
    ctype = content.get("content_type", "text")
    if ctype not in ("text", "multimodal_text"):
        return ""  # code execution, tool output, browsing results, etc.
    parts = content.get("parts") or []
    texts = [p for p in parts if isinstance(p, str)]
    texts += [p.get("text", "") for p in parts if isinstance(p, dict) and isinstance(p.get("text"), str)]
    return "\n\n".join(t for t in texts if t.strip())


def _chatgpt_branch(conv: dict[str, Any]) -> list[dict[str, Any]]:
    """Messages on the active branch, oldest first (the export stores a tree)."""
    mapping: dict[str, Any] = conv.get("mapping") or {}
    node_id = conv.get("current_node")
    if not node_id or node_id not in mapping:
        # Fall back to the deepest leaf.
        leaves = [k for k, v in mapping.items() if not v.get("children")]
        node_id = leaves[-1] if leaves else None
    chain = []
    seen = set()
    while node_id and node_id in mapping and node_id not in seen:
        seen.add(node_id)
        node = mapping[node_id]
        if node.get("message"):
            chain.append(node["message"])
        node_id = node.get("parent")
    return list(reversed(chain))


def parse_chatgpt(path: Path) -> Iterator[CapturedItem | Exception]:
    for conv in _load_conversations(path):
        try:
            conv_id = conv.get("conversation_id") or conv.get("id")
            if not conv_id:
                raise ValueError("conversation without id")
            messages = []
            for msg in _chatgpt_branch(conv):
                role = (msg.get("author") or {}).get("role")
                meta = msg.get("metadata") or {}
                if role not in ("user", "assistant") or meta.get("is_visually_hidden_from_conversation"):
                    continue
                text = _chatgpt_text(msg)
                if text:
                    messages.append((role, text, _ts(msg.get("create_time"))))
            if not messages:
                continue
            yield CapturedItem(
                source_kind="chat",
                origin_ref=f"chatgpt:{conv_id}",
                title=(conv.get("title") or "Untitled ChatGPT conversation").strip(),
                body=render_transcript(messages),
                captured=_ts(conv.get("update_time")) or _ts(conv.get("create_time")),
                subfolder="chatgpt",
                extra={"provider": "chatgpt", "messages": len(messages)},
            )
        except Exception as exc:  # noqa: BLE001 - one bad conversation must not stop the import
            yield ValueError(f"chatgpt conversation {conv.get('id', '?')}: {exc}")


# --- Claude --------------------------------------------------------------------------------


def _claude_text(msg: dict[str, Any]) -> str:
    blocks = msg.get("content")
    if isinstance(blocks, list) and blocks:
        texts = [b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"]
        joined = "\n\n".join(t for t in texts if t and t.strip())
        if joined:
            return joined
    text = msg.get("text")
    return text if isinstance(text, str) else ""


def parse_claude(path: Path) -> Iterator[CapturedItem | Exception]:
    for conv in _load_conversations(path):
        try:
            conv_id = conv.get("uuid") or conv.get("id")
            if not conv_id:
                raise ValueError("conversation without uuid")
            messages = []
            for msg in conv.get("chat_messages") or []:
                sender = msg.get("sender") or msg.get("role")
                role = {"human": "user", "user": "user", "assistant": "assistant"}.get(sender)
                if not role:
                    continue
                text = _claude_text(msg)
                if text.strip():
                    messages.append((role, text, _ts(msg.get("created_at"))))
            if not messages:
                continue
            yield CapturedItem(
                source_kind="chat",
                origin_ref=f"claude:{conv_id}",
                title=(conv.get("name") or "Untitled Claude conversation").strip(),
                body=render_transcript(messages),
                captured=_ts(conv.get("updated_at")) or _ts(conv.get("created_at")),
                subfolder="claude",
                extra={"provider": "claude", "messages": len(messages)},
            )
        except Exception as exc:  # noqa: BLE001
            yield ValueError(f"claude conversation {conv.get('uuid', '?')}: {exc}")


def detect_provider(path: Path) -> str:
    """Guess chatgpt vs claude from the first conversation's shape."""
    convs = _load_conversations(path)
    for conv in convs[:5]:
        if "mapping" in conv:
            return "chatgpt"
        if "chat_messages" in conv:
            return "claude"
    raise ValueError("could not tell whether this is a ChatGPT or a Claude export")
