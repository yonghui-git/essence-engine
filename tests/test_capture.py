import json
import zipfile

import pytest

from essence.capture import import_path
from essence.capture.base import CapturedItem, SourceStore
from essence.config import Config
from essence.index import open_index, reindex
from essence.privacy.redact import redact
from essence.segment import MAX_CHARS, segment
from essence.vault.io import read_note


def _chatgpt_export(tmp_path, extra_msg=False):
    mapping = {
        "root": {"id": "root", "message": None, "parent": None, "children": ["u1"]},
        "u1": {"id": "u1", "parent": "root", "children": ["a1", "a1b"], "message": {
            "author": {"role": "user"}, "create_time": 1762700000.0,
            "content": {"content_type": "text", "parts": ["What is an evergreen note?"]}}},
        "a1b": {"id": "a1b", "parent": "u1", "children": [], "message": {
            "author": {"role": "assistant"}, "content": {"content_type": "text", "parts": ["ABANDONED BRANCH"]}}},
        "a1": {"id": "a1", "parent": "u1", "children": ["t1"], "message": {
            "author": {"role": "assistant"}, "create_time": 1762700010.0,
            "content": {"content_type": "text", "parts": ["A note organized by concept, written to evolve."]}}},
        "t1": {"id": "t1", "parent": "a1", "children": [], "message": {
            "author": {"role": "tool"}, "content": {"content_type": "code", "text": "print(1)"}}},
    }
    current = "t1"
    if extra_msg:
        mapping["t1"]["children"] = ["u2"]
        mapping["u2"] = {"id": "u2", "parent": "t1", "children": [], "message": {
            "author": {"role": "user"}, "content": {"content_type": "text", "parts": ["And a Zettelkasten?"]}}}
        current = "u2"
    convs = [
        {"id": "conv-1", "title": "Evergreen notes", "create_time": 1762700000.0, "update_time": 1762700100.0,
         "mapping": mapping, "current_node": current},
        {"id": "conv-empty", "title": "Nothing", "mapping": {}, "current_node": None},
    ]
    path = tmp_path / "chatgpt.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("conversations.json", json.dumps(convs))
        zf.writestr("user.json", "{}")
    return path


def test_chatgpt_import_follows_active_branch_and_is_idempotent(vault, tmp_path):
    store = SourceStore(vault)
    res = import_path(store, "chat", _chatgpt_export(tmp_path))
    assert len(res.new) == 1 and not res.errors
    path = next((vault.raw / "chats" / "chatgpt").glob("*.md"))
    note = read_note(path)
    assert note.origin_ref == "chatgpt:conv-1" and note.source_kind == "chat"
    assert "What is an evergreen note?" in note.body and "organized by concept" in note.body
    assert "ABANDONED BRANCH" not in note.body and "print(1)" not in note.body

    again = import_path(SourceStore(vault), "chatgpt", _chatgpt_export(tmp_path))
    assert again.unchanged == res.new

    grown = import_path(SourceStore(vault), "chatgpt", _chatgpt_export(tmp_path, extra_msg=True))
    assert grown.updated == res.new  # same id, new content
    assert "Zettelkasten" in read_note(path).body


def test_claude_import(vault, tmp_path):
    convs = [{
        "uuid": "c-1", "name": "Local-first storage", "created_at": "2026-11-01T10:00:00Z",
        "updated_at": "2026-11-01T10:05:00Z",
        "chat_messages": [
            {"sender": "human", "text": "SQLite or Postgres?", "content": [{"type": "text", "text": "SQLite or Postgres?"}]},
            {"sender": "assistant", "text": "", "content": [{"type": "text", "text": "SQLite for one user."},
                                                            {"type": "tool_use", "name": "x"}]},
        ],
    }]
    folder = tmp_path / "claude-export"
    folder.mkdir()
    (folder / "conversations.json").write_text(json.dumps(convs), encoding="utf-8")
    res = import_path(SourceStore(vault), "chat", folder)
    assert len(res.new) == 1
    note = read_note(next((vault.raw / "chats" / "claude").glob("*.md")))
    assert note.origin_ref == "claude:c-1"
    assert "## User" in note.body and "SQLite for one user." in note.body


def test_bad_conversation_is_reported_not_fatal(vault, tmp_path):
    p = tmp_path / "conversations.json"
    p.write_text(json.dumps([{"chat_messages": []}, {"uuid": "ok", "chat_messages": [{"sender": "human", "text": "hello there"}]}]))
    res = import_path(SourceStore(vault), "claude", p)
    assert len(res.new) == 1 and len(res.errors) == 1


def test_web_clips_and_documents(vault, tmp_path):
    clips = tmp_path / "Clippings"
    clips.mkdir()
    (clips / "page.md").write_text(
        "---\ntitle: Evergreen notes\nsource: https://notes.example.com/evergreen\nauthor: A. M.\ntags: [pkm]\n---\nBody text.\n",
        encoding="utf-8",
    )
    res = import_path(SourceStore(vault), "web", clips)
    note = read_note(next((vault.raw / "web").glob("*.md")))
    assert res.new and note.origin_ref == "https://notes.example.com/evergreen"
    assert note.extra["author"] == "A. M." and note.body.strip() == "Body text."

    import docx

    d = docx.Document()
    d.add_heading("Quarterly plan", level=1)
    d.add_paragraph("We will start with consultants.")
    docs = tmp_path / "docs"
    docs.mkdir()
    d.save(docs / "plan.docx")
    (docs / "notes.txt").write_text("plain text document", encoding="utf-8")
    res = import_path(SourceStore(vault), "docs", docs, use_docling=False)
    assert len(res.new) == 2, res.errors
    doc_note = next(read_note(p) for p in (vault.raw / "docs").glob("*.md") if "quarterly" in p.name)
    assert doc_note.title == "Quarterly plan" and "# Quarterly plan" in doc_note.body
    assert (vault.root / doc_note.extra["original"]).is_file()
    with open_index(vault.index_path) as conn:
        report = reindex(vault, conn)  # originals under raw/_originals must not be parsed as notes
    assert report.errors == [] and report.sources == 3


def test_never_cloud_patterns(vault):
    store = SourceStore(vault, Config(never_cloud_patterns=["intranet.acme"]))
    _, a = store.store(CapturedItem("web", "https://intranet.acme/x", "Payroll", "secret plans " * 30))
    _, b = store.store(CapturedItem("web", "https://example.com", "Public", "public text " * 30))
    assert a.never_cloud is True and b.never_cloud is None


def test_segment_is_stable_and_bounded():
    body = "\n\n".join(f"Paragraph {i}. " + "word " * (40 + i * 30) for i in range(12)) + "\n\n## User\n\nshort"
    first, second = segment(body), segment(body)
    assert first == second
    assert all(len(e.text) <= MAX_CHARS for e in first)
    assert [e.n for e in first] == list(range(1, len(first) + 1))
    assert any(e.text.startswith("## User") for e in first)  # headings start excerpts
    assert body[first[0].start:first[0].end].strip() == first[0].text


@pytest.mark.parametrize(
    "text, kind",
    [
        ("mail me at jane.doe@example.com", "EMAIL"),
        ("key sk-ant-api03-abcdefghijklmnopqrstuvwxyz123456", "API_KEY"),
        ("token ghp_abcdefghijklmnopqrstuvwxyz0123", "API_KEY"),
        ("IBAN NL91 ABNA 0417 1643 00 please", "IBAN"),
        ("card 4111 1111 1111 1111", "CARD"),
        ("call +31 6 1234 5678", "PHONE"),
        ("password: hunter2hunter2", "SECRET"),
    ],
)
def test_redaction(text, kind):
    r = redact(text)
    assert f"[{kind}_1]" in r.text and r.counts.get(kind) == 1


def test_redaction_leaves_ordinary_numbers():
    text = "In 2026 we had 700 million users and version 3.12.10; meeting on 2026-11-10."
    assert redact(text).text == text


def test_redaction_is_consistent():
    r = redact("a@b.com wrote to c@d.com, then a@b.com again")
    assert r.text == "[EMAIL_1] wrote to [EMAIL_2], then [EMAIL_1] again"
