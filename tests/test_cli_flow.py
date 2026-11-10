import json

from typer.testing import CliRunner

from essence.cli import app
from essence.distill.inbox import create_suggestion
from essence.index import open_index
from essence.vault.layout import Vault
from essence.vault.model import Note, NoteType

runner = CliRunner()


def _cli(root, *args):
    r = runner.invoke(app, ["--vault", str(root), *args])
    return r


def test_full_cli_flow_without_llm(tmp_path):
    root = tmp_path / "kb"
    assert runner.invoke(app, ["init", str(root)]).exit_code == 0

    export = tmp_path / "conversations.json"
    export.write_text(json.dumps([{
        "uuid": "c1", "name": "Storage choice",
        "chat_messages": [{"sender": "human", "text": "SQLite or Postgres for a local tool?"},
                          {"sender": "assistant", "text": "SQLite: one file, FTS5, no server."}],
    }]), encoding="utf-8")
    r = _cli(root, "import", "claude", str(export))
    assert r.exit_code == 0 and "1 new" in r.output

    r = _cli(root, "distill")
    assert r.exit_code == 1 and "No LLM provider" in r.output  # tests run with provider none

    vault = Vault(root.resolve())
    with open_index(vault.index_path) as conn:
        sug = create_suggestion(vault, conn, op="add", note=Note(type=NoteType.DECISION, title="Use SQLite", body="One file."))
    r = _cli(root, "inbox")
    assert "Use SQLite" in r.output and "1 pending" in r.output
    r = _cli(root, "accept", sug.id[-6:])
    assert r.exit_code == 0 and "accepted" in r.output
    assert "Inbox is empty" in _cli(root, "inbox").output

    r = _cli(root, "search", "sqlite")
    assert "Use SQLite" in r.output
    note_id = r.output.split("|")[1].strip()
    r = _cli(root, "show", note_id)
    assert r.exit_code == 0 and "One file." in r.output

    r = _cli(root, "digest", "--no-llm")
    assert r.exit_code == 0 and "Use SQLite" in r.output
    assert list((root / "digests").glob("*.md"))

    assert "accepted as-is: 1" in _cli(root, "metrics").output
    assert "No LLM calls" in _cli(root, "cost").output
    assert "No changes yet" in _cli(root, "changes").output
    assert "vault at" in _cli(root, "doctor").output

    with open_index(vault.index_path) as conn:
        src = conn.execute("SELECT id FROM sources").fetchone()[0]
    r = _cli(root, "delete", src, "--yes")
    assert r.exit_code == 0 and "Deleted" in r.output
    assert "sources: 0" in _cli(root, "stats").output


def test_reject_requires_match(tmp_path):
    runner.invoke(app, ["init", str(tmp_path)])
    r = _cli(tmp_path, "reject", "nothing-like-this")
    assert r.exit_code == 1


def test_serve_show_token_is_stable(tmp_path):
    runner.invoke(app, ["init", str(tmp_path)])
    a = _cli(tmp_path, "serve", "--show-token").output.strip()
    b = _cli(tmp_path, "serve", "--show-token").output.strip()
    assert a == b and len(a) >= 32
