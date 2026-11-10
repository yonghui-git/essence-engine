from essence.index import open_index, reindex, search
from essence.index.db import _MIGRATIONS, schema_version
from essence.index.search import to_fts_query
from essence.vault.io import write_note
from essence.vault.layout import init_vault, slugify
from essence.vault.model import Note, NoteStatus, NoteType


def _add(vault, **kw):
    note = Note(**kw)
    write_note(note, vault.path_for(note))
    return note


def test_init_is_idempotent_and_keeps_user_edits(vault):
    vault.schema_path.write_text("my schema", encoding="utf-8")
    _, created = init_vault(vault.root)
    assert created == []
    assert vault.schema_path.read_text(encoding="utf-8") == "my schema"


def test_reindex_builds_notes_edges_and_sources(vault):
    src = Note(type=NoteType.SOURCE, title="Karpathy LLM Wiki", source_kind="web", origin_ref="https://example.com")
    write_note(src, vault.raw / "web" / "llm-wiki.md")
    con = _add(vault, type=NoteType.CONCEPT, title="LLM Wiki", body="An LLM maintains the wiki.")
    ins = _add(
        vault,
        type=NoteType.INSIGHT,
        title="Upkeep killed wikis",
        body="People abandon wikis because upkeep grows faster than value.",
        links=[con.id],
        sources=[f"{src.id}#exc_1"],
    )

    with open_index(vault.index_path) as conn:
        report = reindex(vault, conn)
        assert (report.notes, report.sources, report.edges) == (2, 1, 2)
        assert report.errors == [] and report.dangling == []
        edges = {tuple(r) for r in conn.execute("SELECT src, dst, kind FROM edges")}
        assert edges == {(ins.id, con.id, "links"), (ins.id, src.id, "derived_from")}


def test_reindex_is_repeatable_and_keeps_operational_tables(vault):
    _add(vault, type=NoteType.CONCEPT, title="Zettelkasten")
    with open_index(vault.index_path) as conn:
        conn.execute(
            "INSERT INTO suggestions (id, op, payload, created) VALUES ('sug_1', 'add', '{}', '2026-11-10')"
        )
        conn.commit()
        first = reindex(vault, conn)
        second = reindex(vault, conn)
        assert first.notes == second.notes == 1
        assert conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM notes_fts").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM suggestions").fetchone()[0] == 1


def test_index_can_be_deleted_and_rebuilt(vault):
    _add(vault, type=NoteType.CLAIM, title="Forgetting curve is steep", body="Decay within days.")
    with open_index(vault.index_path) as conn:
        reindex(vault, conn)
    vault.index_path.unlink()
    for extra in ("-wal", "-shm"):
        vault.index_path.with_name(vault.index_path.name + extra).unlink(missing_ok=True)
    with open_index(vault.index_path) as conn:
        assert schema_version(conn) == len(_MIGRATIONS)
        reindex(vault, conn)
        assert [h.title for h in search(conn, "forgetting")] == ["Forgetting curve is steep"]


def test_bad_and_duplicate_files_are_reported_not_fatal(vault):
    good = _add(vault, type=NoteType.CONCEPT, title="Good")
    (vault.root / "notes/concepts/broken.md").write_text("no frontmatter", encoding="utf-8")
    dup = vault.root / "notes/concepts/zz-copy.md"
    dup.write_bytes(vault.path_for(good).read_bytes())
    with open_index(vault.index_path) as conn:
        report = reindex(vault, conn)
    assert report.notes == 1
    assert len(report.errors) == 1 and "broken.md" in report.errors[0]
    assert len(report.duplicates) == 1


def test_dangling_links_are_reported(vault):
    _add(vault, type=NoteType.INSIGHT, title="Orphan link", links=["con_MISSING"])
    with open_index(vault.index_path) as conn:
        report = reindex(vault, conn)
    assert len(report.dangling) == 1 and "con_MISSING (links)" in report.dangling[0]


def test_history_versions_are_indexed(vault):
    v1 = Note(type=NoteType.INSIGHT, title="AI should write everything", version=1, status=NoteStatus.SUPERSEDED)
    v2 = v1.model_copy(update={"title": "AI keeps the books, I keep the judgment", "version": 2, "status": NoteStatus.ACTIVE})
    write_note(v2, vault.path_for(v2))
    write_note(v1, vault.history_path(v1.id, 1))
    with open_index(vault.index_path) as conn:
        report = reindex(vault, conn)
        assert report.notes == 1 and report.versions == 1
        row = conn.execute("SELECT title, status FROM note_versions WHERE note_id = ?", (v1.id,)).fetchone()
        assert tuple(row) == ("AI should write everything", "superseded")


def test_search_ranks_title_matches_and_filters_status(vault):
    _add(vault, type=NoteType.CONCEPT, title="Review inbox", body="Suggestions wait for the user.")
    _add(vault, type=NoteType.INSIGHT, title="Ownership", body="A review step keeps ownership of ideas.")
    _add(vault, type=NoteType.CLAIM, title="Old review claim", status=NoteStatus.INVALID, body="review")
    with open_index(vault.index_path) as conn:
        reindex(vault, conn)
        titles = [h.title for h in search(conn, "review")]
        assert titles == ["Review inbox", "Ownership"]
        assert len(search(conn, "review", include_inactive=True)) == 3
        assert [h.title for h in search(conn, "review", types=["insight"])] == ["Ownership"]
        assert [h.title for h in search(conn, "revi")] == ["Review inbox", "Ownership"]  # prefix
        assert conn.execute("SELECT COUNT(*) FROM events WHERE kind = 'search'").fetchone()[0] == 4


def test_fts_query_escapes_syntax():
    assert to_fts_query('title:foo AND "bar') == '"title" "foo" "AND" "bar"*'
    assert to_fts_query("  ") == ""


def test_search_survives_hostile_input(vault):
    _add(vault, type=NoteType.CONCEPT, title="NEAR operator")
    with open_index(vault.index_path) as conn:
        reindex(vault, conn)
        for q in ['"', "NEAR(", "title:", "*", "a OR", "-x"]:
            search(conn, q)  # must not raise


def test_slugify():
    assert slugify("Évergreen notes: concept-oriented!") == "evergreen-notes-concept-oriented"
    assert slugify("你好") == "note"
