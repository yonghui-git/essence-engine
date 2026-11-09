from typer.testing import CliRunner

from essence.cli import app

runner = CliRunner()


def test_cli_end_to_end(tmp_path):
    root = tmp_path / "kb"
    r = runner.invoke(app, ["init", str(root)])
    assert r.exit_code == 0, r.output

    r = runner.invoke(app, ["--vault", str(root), "new", "concept", "Evergreen notes", "-b", "Organize by concept."])
    assert r.exit_code == 0, r.output
    note_id = r.output.split()[0]
    assert note_id.startswith("con_")

    r = runner.invoke(app, ["--vault", str(root), "reindex"])
    assert r.exit_code == 0 and "Indexed 1 notes" in r.output

    r = runner.invoke(app, ["--vault", str(root), "search", "concept"])
    assert r.exit_code == 0 and "Evergreen notes" in r.output

    r = runner.invoke(app, ["--vault", str(root), "stats"])
    assert "concept" in r.output and "active" in r.output


def test_cli_requires_initialized_vault(tmp_path):
    r = runner.invoke(app, ["--vault", str(tmp_path), "search", "x"])
    assert r.exit_code == 1


def test_cli_rejects_unknown_type(tmp_path):
    runner.invoke(app, ["init", str(tmp_path)])
    r = runner.invoke(app, ["--vault", str(tmp_path), "new", "source", "x"])
    assert r.exit_code != 0
    r = runner.invoke(app, ["--vault", str(tmp_path), "new", "banana", "x"])
    assert r.exit_code != 0


def test_reindex_exit_code_on_errors(tmp_path):
    runner.invoke(app, ["init", str(tmp_path)])
    (tmp_path / "notes" / "concepts" / "bad.md").write_text("oops", encoding="utf-8")
    r = runner.invoke(app, ["--vault", str(tmp_path), "reindex"])
    assert r.exit_code == 2
