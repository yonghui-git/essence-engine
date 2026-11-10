from datetime import date, timedelta

import pytest

from essence.consolidate.digest import Week, build_digest, changes
from essence.decisions import DecisionError, list_decisions, record_outcome
from essence.distill.pipeline import _candidate_note
from essence.index import open_index, reindex
from essence.vault.io import read_note, write_note
from essence.vault.model import Note, NoteType


def _decision(vault, title, review_on=None, expected="It works"):
    n = Note(type=NoteType.DECISION, title=title, body="Chosen because.\n", expected_outcome=expected, review_on=review_on)
    write_note(n, vault.path_for(n))
    return n


def test_suggested_decisions_get_a_review_date():
    cand = {"type": "decision", "title": "Use SQLite", "body": "b", "applies_when": "", "not_when": "",
            "excerpt_ids": ["exc_1"], "expected_outcome": "fast enough"}
    assert _candidate_note(cand, "src_X", 90).review_on == date.today() + timedelta(days=90)
    assert _candidate_note(cand, "src_X", 0).review_on is None
    assert _candidate_note({**cand, "type": "insight"}, "src_X", 90).review_on is None


def test_list_orders_due_first(vault):
    today = date(2026, 11, 10)
    _decision(vault, "Later", today + timedelta(days=30))
    _decision(vault, "Overdue", today - timedelta(days=3))
    _decision(vault, "No date")
    with open_index(vault.index_path) as conn:
        reindex(vault, conn)
        rows = list_decisions(conn, vault, today=today)
    assert [(r.title, r.state) for r in rows] == [("Overdue", "due"), ("Later", "upcoming"), ("No date", "open")]


def test_record_outcome_versions_and_shows_in_digest(vault):
    d = _decision(vault, "Weekly Sunday review", date(2026, 11, 1), expected="I remember more")
    updated = record_outcome(vault, d.id, actual="Skipped 4 of 6", lesson="Make review cheap", next_review=None)
    assert updated.version == 2 and updated.review_on is None and updated.change_reason == "outcome recorded"
    assert vault.history_path(d.id, 1).exists()  # the expectation-only version is kept
    on_disk = read_note(vault.path_for(updated))
    assert on_disk.actual_outcome == "Skipped 4 of 6" and on_disk.lesson == "Make review cheap"
    with open_index(vault.index_path) as conn:
        reindex(vault, conn)
        (c,) = changes(vault, conn)
        assert c.kind == "outcome" and c.before == "I remember more" and c.after == "Skipped 4 of 6"
        text = build_digest(vault, conn, Week.of(date.today()))
        rows = list_decisions(conn, vault)
    assert "## Decisions reviewed" in text and "actually Skipped 4 of 6. Lesson: Make review cheap" in text
    changed = text.split("## Changed my mind")[1].split("##")[0]
    assert "Weekly Sunday review" not in changed  # an outcome is not a change of mind by itself
    assert rows[0].state == "reviewed"


def test_record_outcome_rejects_non_decisions(vault):
    n = Note(type=NoteType.INSIGHT, title="Not a decision", body="b\n")
    write_note(n, vault.path_for(n))
    with pytest.raises(DecisionError):
        record_outcome(vault, n.id, actual="x")
    d = _decision(vault, "Needs text")
    with pytest.raises(DecisionError):
        record_outcome(vault, d.id, actual="  ")
