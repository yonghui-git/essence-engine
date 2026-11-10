import json
from datetime import date

from typer.testing import CliRunner

from essence.app import App
from essence.capture.base import CapturedItem, SourceStore
from essence.cli import app as cli
from essence.config import Config
from essence.nightly import run_nightly
from essence.vault.io import write_note
from essence.vault.model import Note, NoteType
from tests.helpers import LONG_CHAT


def test_nightly_without_llm_or_embeddings_still_does_the_rest(vault):
    SourceStore(vault, Config()).store(CapturedItem("chat", "c1", "A chat", LONG_CHAT))
    d = Note(type=NoteType.DECISION, title="Try the inbox", body="b", review_on=date(2026, 11, 1))
    write_note(d, vault.path_for(d))
    a = App(vault.root)
    rep = run_nightly(a, today=date(2026, 11, 10))
    text = "\n".join(rep.lines)
    assert "skipped: no LLM provider" in text and "skipped: embeddings are off" in text
    assert "1 due for review: Try the inbox" in text
    assert (vault.digests / "2026-W45.md").exists()  # last full week, written once
    again = run_nightly(a, today=date(2026, 11, 11))
    assert any("2026-W45 already written" in line for line in again.lines) and not again.problems


def test_metrics_json_has_counts_only(vault, monkeypatch):
    monkeypatch.setenv("ESSENCE_VAULT", str(vault.root))
    out = CliRunner().invoke(cli, ["metrics", "--json"])
    assert out.exit_code == 0, out.output
    data = json.loads(out.output)
    assert {"reviewed", "accepted", "edited", "rejected", "active_days_28", "version"} <= set(data)
    assert all(isinstance(v, (int, float, str)) for v in data.values())
