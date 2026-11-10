from datetime import date

from essence.consolidate.digest import Week, build_digest, changes
from essence.distill.apply import accept, reject
from essence.distill.inbox import create_suggestion, pending
from essence.index import open_index, reindex
from essence.index.search import hybrid_search
from essence.index.vectors import embed_notes
from essence.vault.io import read_note, write_note
from essence.vault.model import Note, NoteStatus, NoteType, Origin
from essence.vault.repo import NoteRepo
from tests.helpers import HashEmbedder


def _note(vault, **kw):
    n = Note(**kw)
    write_note(n, vault.path_for(n))
    return n


def _suggest(vault, conn, op, note, **kw):
    return create_suggestion(vault, conn, op=op, note=note, **kw)


def test_accept_add_and_edit_detection(vault):
    with open_index(vault.index_path) as conn:
        s1 = _suggest(vault, conn, "add", Note(type=NoteType.INSIGHT, title="A", body="as proposed", created_by=Origin.AI))
        s2 = _suggest(vault, conn, "add", Note(type=NoteType.INSIGHT, title="B", body="as proposed", created_by=Origin.AI))
        # The person rewords s2 in the inbox file.
        s2.path.write_text(s2.path.read_text(encoding="utf-8").replace("as proposed", "in my own words"), encoding="utf-8")
        from essence.distill.inbox import read_suggestion

        r1 = accept(vault, conn, s1)
        r2 = accept(vault, conn, read_suggestion(s2.path))
        assert (r1.status, r2.status) == ("accepted", "edited")
        rows = dict(conn.execute("SELECT id, edit_distance FROM suggestions").fetchall())
        assert rows[s1.id] == 0 and rows[s2.id] > 0
        reindex(vault, conn)
        notes = {r["title"]: dict(r) for r in conn.execute("SELECT title, status, created_by FROM notes")}
    assert notes["A"] == {"title": "A", "status": "active", "created_by": "ai"}
    assert notes["B"]["created_by"] == "ai+edited"
    assert pending(vault)[0] == [] and not list(vault.inbox.glob("*.md"))
    note = read_note(NoteRepo(vault).path(r2.changed[0]))
    assert "suggestion" not in note.extra and note.status is NoteStatus.ACTIVE


def test_update_archives_history_and_supersede_marks_old(vault):
    old = _note(vault, type=NoteType.INSIGHT, title="AI should write all notes", body="v1", sources=["src_A#exc_1"])
    with open_index(vault.index_path) as conn:
        upd = old.model_copy(update={"title": "AI should draft notes", "body": "v2", "sources": ["src_B#exc_2"]})
        accept(vault, conn, _suggest(vault, conn, "update", upd, target_id=old.id, reason="refined after reading"))
        repo = NoteRepo(vault)
        _, cur = repo.get(old.id)
        assert cur.version == 2 and cur.body.strip() == "v2" and cur.change_reason == "refined after reading"
        assert set(cur.sources) == {"src_A#exc_1", "src_B#exc_2"}
        assert vault.history_path(old.id, 1).is_file()

        new = Note(type=NoteType.INSIGHT, title="The AI keeps the books; I keep the judgment", body="v3", created_by=Origin.AI)
        res = accept(vault, conn, _suggest(vault, conn, "supersede", new, target_id=old.id, reason="studies on ownership"))
        repo.refresh()
        _, was = repo.get(old.id)
        _, now = repo.get(res.changed[0])
        assert was.status is NoteStatus.SUPERSEDED and was.superseded_by == now.id and was.valid_to == date.today()
        assert old.id in now.links
        reindex(vault, conn)
        log = changes(vault, conn)
        assert {c.kind for c in log} == {"superseded"}
        assert log[0].after == "The AI keeps the books; I keep the judgment"
        assert log[0].before == "AI should draft notes"
        # Superseded notes leave default search but remain findable with include_inactive.
        assert [h.title for h in hybrid_search(conn, "books judgment AI")] == [now.title]
        assert len(hybrid_search(conn, "AI", include_inactive=True)) == 2


def test_confirm_link_merge_reject(vault):
    a = _note(vault, type=NoteType.CONCEPT, title="Evergreen notes", body="Concept-oriented.", sources=["src_A#exc_1"])
    b = _note(vault, type=NoteType.CONCEPT, title="Evergreen note idea", body="Organize by concept.")
    c = _note(vault, type=NoteType.CONCEPT, title="Zettelkasten", body="Atomic cards.")
    with open_index(vault.index_path) as conn:
        accept(vault, conn, _suggest(vault, conn, "confirm", Note(type=NoteType.CONCEPT, title="x", body="y", sources=["src_C#exc_4"]), target_id=a.id))
        accept(vault, conn, _suggest(vault, conn, "link", Note(type=NoteType.CONCEPT, title="x", body="y"), target_id=a.id, link_ids=[c.id]))
        merged = a.model_copy(update={"title": "Evergreen notes", "body": "Concept-oriented; organize by concept."})
        accept(vault, conn, _suggest(vault, conn, "merge", merged, target_id=a.id, link_ids=[b.id]))
        r = reject(vault, conn, _suggest(vault, conn, "add", Note(type=NoteType.CLAIM, title="nope", body="no")))
        assert r.status == "rejected"
        repo = NoteRepo(vault)
        _, na = repo.get(a.id)
        _, nb = repo.get(b.id)
        assert "src_C#exc_4" in na.sources and c.id in na.links and na.version == 2
        assert nb.status is NoteStatus.SUPERSEDED and nb.superseded_by == a.id
        statuses = dict(conn.execute("SELECT op, status FROM suggestions").fetchall())
    assert statuses == {"confirm": "accepted", "link": "accepted", "merge": "accepted", "add": "rejected"}


def test_hybrid_search_uses_vectors_and_links(vault):
    a = _note(vault, type=NoteType.CONCEPT, title="Spaced repetition", body="Review at growing intervals to retain memory.")
    b = _note(vault, type=NoteType.INSIGHT, title="Distill soon after reading", body="Most forgetting happens in a day.", links=[a.id])
    _note(vault, type=NoteType.QUESTION, title="Which sync tool?", body="Git or Syncthing.")
    emb = HashEmbedder()
    with open_index(vault.index_path) as conn:
        reindex(vault, conn)
        assert embed_notes(conn, emb) == 3
        assert embed_notes(conn, emb) == 3  # cached
        assert conn.execute("SELECT COUNT(*) FROM embed_cache").fetchone()[0] == 3
        hits = hybrid_search(conn, "intervals memory", embedder=emb)
        assert hits[0].note_id == a.id and "vector" in hits[0].via and "text" in hits[0].via
        linked = {h.note_id: h.via for h in hits}
        assert b.id in linked and "link" in linked[b.id]
        no_vec = hybrid_search(conn, "intervals memory")
        assert no_vec[0].note_id == a.id


def test_digest_sections(vault):
    _note(vault, type=NoteType.INSIGHT, title="Learned this week", body="x")
    _note(vault, type=NoteType.DECISION, title="Pick SQLite", body="y", review_on=date(2020, 1, 1))
    _note(vault, type=NoteType.QUESTION, title="Will users return weekly?", body="z")
    with open_index(vault.index_path) as conn:
        reindex(vault, conn)
        text = build_digest(vault, conn, Week.of(date.today()))
    assert "# Week" in text and "**Learned this week**" in text
    assert "Pick SQLite (review on 2020-01-01)" in text
    assert "Will users return weekly?" in text and "## Changed my mind" in text
    w = Week.parse("2026-W46")
    assert w.start == date(2026, 11, 9) and w.label == "2026-W46" and w.contains("2026-11-15T23:00:00")


def test_find_accepts_the_path_that_inbox_prints(vault):
    from essence.distill.inbox import create_suggestion, find
    from essence.index import open_index
    from essence.vault.model import Note, NoteType

    with open_index(vault.index_path) as conn:
        sug = create_suggestion(vault, conn, op="add", note=Note(type=NoteType.INSIGHT, title="Path refs work", body="b"))
    rel = vault.relpath(sug.path)
    for ref in (rel, rel.replace("/", "\\"), str(sug.path), sug.path.name, sug.id):
        assert find(vault, ref).id == sug.id
