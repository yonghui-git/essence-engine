from datetime import date

import pytest

from essence.ids import ulid
from essence.vault.io import NoteParseError, parse_note, render_note
from essence.vault.model import Note, NoteStatus, NoteType, Origin


def test_ids_are_prefixed_and_sortable():
    a = ulid(1_000)
    b = ulid(2_000)
    assert len(a) == 26 and a < b
    assert Note(type=NoteType.INSIGHT, title="x").id.startswith("ins_")
    assert Note(type=NoteType.SOURCE, title="x").id.startswith("src_")


def test_round_trip_is_lossless_for_every_field():
    note = Note(
        type=NoteType.DECISION,
        title="Start with consultants",
        status=NoteStatus.ACTIVE,
        applies_when="choosing a first market",
        not_when="raising venture money",
        sources=["src_01ABC#exc_3", "src_01DEF"],
        links=["con_01XYZ"],
        valid_from=date(2026, 11, 9),
        version=2,
        created_by=Origin.AI_EDITED,
        expected_outcome="a third prepay",
        review_on=date(2027, 2, 1),
        body="We chose consultants because they reuse insights across clients.\n\n- point one\n",
    )
    text = render_note(note)
    again = parse_note(text)
    assert again == note
    assert render_note(again) == text


def test_unknown_frontmatter_keys_are_preserved():
    text = (
        "---\n"
        "id: con_01TEST\n"
        "type: concept\n"
        "title: Evergreen notes\n"
        "status: active\n"
        "version: 1\n"
        "created_by: user\n"
        "created: '2026-11-10T08:00:00Z'\n"
        "tags: [pkm, writing]\n"
        "aliases:\n  - concept-oriented notes\n"
        "reviewed: 2026-11-01\n"
        "---\n"
        "Organize by concept.\n"
    )
    note = parse_note(text)
    assert note.extra == {
        "tags": ["pkm", "writing"],
        "aliases": ["concept-oriented notes"],
        "reviewed": "2026-11-01",
    }
    assert parse_note(render_note(note)) == note


def test_empty_optional_fields_are_not_written():
    text = render_note(Note(type=NoteType.CONCEPT, title="Memex"))
    assert "applies_when" not in text
    assert "links" not in text
    assert "confidence" not in text


def test_crlf_and_bom_are_tolerated():
    text = "﻿---\r\nid: qst_01A\r\ntype: question\r\ntitle: Why?\r\n---\r\nline one\r\nline two\r\n"
    note = parse_note(text)
    assert note.body == "line one\nline two\n"


@pytest.mark.parametrize(
    "text, message",
    [
        ("no frontmatter here", "missing YAML frontmatter"),
        ("---\n: : bad\n---\n", "invalid YAML"),
        ("---\n- a list\n---\n", "not a mapping"),
        ("---\ntype: concept\n---\n", "title"),
        ("---\ntype: claim\ntitle: x\nconfidence: 2\n---\n", "confidence"),
        ("---\ntype: nonsense\ntitle: x\n---\n", "type"),
    ],
)
def test_parse_errors_are_reported(text, message):
    with pytest.raises(NoteParseError, match=message):
        parse_note(text)
