import asyncio
import json

import pytest

from essence.app import App
from essence.capture.base import CapturedItem, SourceStore
from essence.distill.inbox import create_suggestion, pending
from essence.distill.apply import accept, reject
from essence.evaluation.run import load_golden, run_eval, score_item
from essence.index import open_index, reindex
from essence.index.db import IndexKeyError, connect, is_encrypted
from essence.llm.base import LLMClient, tokens_today
from essence.llm.cost import cost_usd, price
from essence.mcp_server.server import BearerAuth, KnowledgeTools, build_server
from essence.ops import cost_report, delete_source, encrypt_index, metrics
from essence.vault.io import write_note
from essence.vault.model import Note, NoteType, Origin
from essence.vault.repo import NoteRepo
from tests.helpers import FakeProvider, cand


def _src(vault, origin="x", body="Some captured text " * 30):
    _, n = SourceStore(vault).store(CapturedItem("web", origin, f"Source {origin}", body))
    return n


def test_delete_propagates_to_notes_history_and_inbox(vault):
    s1, s2 = _src(vault, "one"), _src(vault, "two")
    only = Note(type=NoteType.INSIGHT, title="Only from s1", body="b", sources=[f"{s1.id}#exc_1"], created_by=Origin.AI)
    both = Note(type=NoteType.CLAIM, title="From both", body="b", sources=[f"{s1.id}#exc_1", f"{s2.id}#exc_1"], created_by=Origin.AI)
    mine = Note(type=NoteType.INSIGHT, title="Mine", body="b", sources=[f"{s1.id}#exc_2"], created_by=Origin.USER)
    for n in (only, both, mine):
        write_note(n, vault.path_for(n))
    write_note(both.model_copy(update={"version": 1}), vault.history_path(both.id, 1))
    with open_index(vault.index_path) as conn:
        reindex(vault, conn)
        pend = create_suggestion(vault, conn, op="add", note=Note(type=NoteType.CLAIM, title="p", body="q"), source_id=s1.id)
        rep = delete_source(vault, conn, s1.id)
        assert rep.orphans == [only.id]
        assert set(rep.notes_updated) == {both.id, mine.id}
        assert rep.suggestions_removed == [pend.id]
        assert len(rep.raw_files) == 1
        reindex(vault, conn)
        assert conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM excerpts WHERE source_id = ?", (s1.id,)).fetchone()[0] == 0
        repo = NoteRepo(vault)
        assert repo.get(both.id)[1].sources == [f"{s2.id}#exc_1"]
        assert repo.get(mine.id)[1].sources == []  # user notes are kept, just uncited
        from essence.vault.io import read_note

        assert read_note(vault.history_path(both.id, 1)).sources == [f"{s2.id}#exc_1"]
        (orphan,) = [s for s in pending(vault)[0] if s.op == "orphan"]
        assert repo.get(only.id) is None
        accept(vault, conn, orphan)  # keep it
        repo.refresh()
        assert repo.get(only.id)[1].sources == []


def test_rejecting_orphan_deletes_it(vault):
    s = _src(vault)
    n = Note(type=NoteType.INSIGHT, title="Gone", body="b", sources=[s.id], created_by=Origin.AI)
    write_note(n, vault.path_for(n))
    with open_index(vault.index_path) as conn:
        delete_source(vault, conn, s.id)
        (orphan,) = pending(vault)[0]
        reject(vault, conn, orphan)
    assert NoteRepo(vault).get(n.id) is None


def test_encrypted_index(vault):
    pytest.importorskip("sqlcipher3")
    write_note(Note(type=NoteType.CONCEPT, title="Secret concept", body="b"), vault.root / "notes/concepts/s.md")
    with open_index(vault.index_path) as conn:
        reindex(vault, conn)
        conn.execute("INSERT INTO events (at, kind) VALUES ('2026-11-10', 'search')")
        conn.commit()
    assert not is_encrypted(vault.index_path)
    encrypt_index(vault.index_path, "correct horse")
    assert is_encrypted(vault.index_path)
    assert b"Secret concept" not in vault.index_path.read_bytes()
    with open_index(vault.index_path, "correct horse") as conn:
        assert conn.execute("SELECT title FROM notes").fetchone()[0] == "Secret concept"
        assert conn.execute("SELECT COUNT(*) FROM events WHERE kind = 'search'").fetchone()[0] == 1
    with pytest.raises(IndexKeyError):
        connect(vault.index_path, "wrong")
    with pytest.raises(IndexKeyError):
        connect(vault.index_path)  # no key for an encrypted file


def test_app_uses_key_from_env(vault, monkeypatch):
    pytest.importorskip("sqlcipher3")
    text = vault.config_path.read_text(encoding="utf-8").replace("encrypt = false", "encrypt = true")
    vault.config_path.write_text(text, encoding="utf-8")
    a = App(vault.root)
    with pytest.raises(IndexKeyError):
        a.db_key
    monkeypatch.setenv("ESSENCE_DB_KEY", "k3y")
    with a.index() as conn:
        a.refresh(conn)
    assert is_encrypted(vault.index_path)


def test_metrics_and_costs(vault):
    with open_index(vault.index_path) as conn:
        s1 = create_suggestion(vault, conn, op="add", note=Note(type=NoteType.CLAIM, title="a", body="b"))
        s2 = create_suggestion(vault, conn, op="add", note=Note(type=NoteType.CLAIM, title="c", body="d"))
        accept(vault, conn, s1)
        reject(vault, conn, s2)
        llm = LLMClient(FakeProvider(), conn)
        llm.call(purpose="extract", system="s", user="u", schema={"properties": {"candidates": {}}}, model="claude-sonnet-5-5")
        conn.execute("INSERT INTO distill_state VALUES ('src_1', 'h', 'done', 'full', 1, 1, NULL, datetime('now'))")
        conn.commit()
        m = metrics(conn)
        rep = cost_report(conn)
        assert tokens_today(conn) == 1200
    assert (m.reviewed, m.accepted, m.rejected, m.share(m.accepted)) == (2, 1, 1, "50%")
    assert m.active_days_28 == 1
    assert rep.total_usd == pytest.approx(cost_usd("claude-sonnet-5-5", 1000, 200))
    assert rep.per_item == pytest.approx(rep.total_usd)
    assert price("anthropic.claude-haiku-5-5") == (0.10, 0.50)
    assert price("some-unknown-model") == max([price("claude-fable-5-1")])
    assert cost_usd("claude-opus-5-5", 1_000_000, 0, batch=True) == 2.0


def test_mcp_tools(vault):
    a = App(vault.root)
    n = Note(type=NoteType.INSIGHT, title="Review inbox is core", body="The AI suggests, I decide.")
    write_note(n, vault.path_for(n))
    with a.index() as conn:
        a.refresh(conn)
    tools = KnowledgeTools(a)
    hits = tools.search("review inbox")
    assert hits and hits[0]["id"] == n.id
    note = tools.read(n.id)
    assert note["title"] == "Review inbox is core" and "I decide" in note["body"]
    assert "error" in tools.read("ins_missing")
    saved = tools.save("Lesson from today", "Distill within a day; most forgetting happens in 24 hours.")
    assert saved["saved_source_id"].startswith("src_")
    assert "error" in tools.save("x", "too short")
    with a.index() as conn:
        kinds = [r[0] for r in conn.execute("SELECT kind FROM events WHERE kind LIKE 'mcp_%'")]
    assert sorted(kinds) == ["mcp_read", "mcp_save", "mcp_search"]

    server = build_server(a)
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert names == {"search_knowledge", "read_note_by_id", "save_to_knowledge_base", "list_inbox", "list_decisions"}
    result = asyncio.run(server.call_tool("search_knowledge", {"query": "review"}))
    assert "Review inbox is core" in json.dumps(result.model_dump(), ensure_ascii=False)

    templates = asyncio.run(server.list_resource_templates())
    assert [t.uri_template for t in templates] == ["note://{note_id}"]
    contents = asyncio.run(server.read_resource(f"note://{n.id}"))
    text = json.dumps([c.model_dump() if hasattr(c, "model_dump") else str(c) for c in contents], ensure_ascii=False)
    assert "Review inbox is core" in text and "type: insight" in text


def test_mcp_decisions(vault):
    from datetime import date, timedelta

    a = App(vault.root)
    due = Note(type=NoteType.DECISION, title="Weekly review on Sunday", body="b", expected_outcome="I remember more",
               review_on=date.today() - timedelta(days=1))
    later = Note(type=NoteType.DECISION, title="Use SQLite", body="b", review_on=date.today() + timedelta(days=30))
    for n in (due, later):
        write_note(n, vault.path_for(n))
    with a.index() as conn:
        a.refresh(conn)
    tools = KnowledgeTools(a)
    assert [d["title"] for d in tools.decisions()] == ["Weekly review on Sunday"]
    assert tools.decisions()[0]["expected_outcome"] == "I remember more"
    assert [d["state"] for d in tools.decisions(due_only=False)] == ["due", "upcoming"]


def test_bearer_auth():
    calls = []

    async def inner(scope, receive, send):
        calls.append(scope["path"])

    sent = []

    async def send(msg):
        sent.append(msg)

    mw = BearerAuth(inner, "s3cret")
    asyncio.run(mw({"type": "http", "path": "/mcp", "headers": []}, None, send))
    assert sent[0]["status"] == 401 and calls == []
    asyncio.run(mw({"type": "http", "path": "/mcp", "headers": [(b"authorization", b"Bearer wrong")]}, None, send))
    assert calls == []
    asyncio.run(mw({"type": "http", "path": "/mcp", "headers": [(b"authorization", b"Bearer s3cret")]}, None, send))
    assert calls == ["/mcp"]


def test_eval_scoring_and_runner():
    items = load_golden()
    assert len(items) >= 6 and any(i["must_not"] for i in items)
    spec = {"id": "x", "expected": [{"type": "decision", "keywords": ["sqlite", "postgres|later"]}], "must_not": ["pwned"]}
    good = score_item(spec, [("decision", "go with sqlite now, postgres later")])
    assert (good.found, good.matching_candidates) == (1, 1)
    wrong_type = score_item(spec, [("insight", "sqlite postgres")])
    assert wrong_type.found == 0
    bad = score_item(spec, [("decision", "sqlite later pwned")])
    assert bad.violations == ["pwned"]

    def extract(system, user):
        if "Zettelkasten" in user:
            return {"candidates": [cand("Atomic notes compound", "One idea per note, densely linked, connects to everything.")], "skip_reason": ""}
        return {"candidates": [], "skip_reason": "nothing"}

    from essence.config import Config

    rep = run_eval(FakeProvider(extract=extract), Config(), [i for i in items if i["id"] in ("injection-web", "small-talk-chat")])
    assert rep.recall == 1.0 and rep.skip_accuracy == 1.0 and rep.violations == 0
