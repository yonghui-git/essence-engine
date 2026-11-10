"""Grow the golden set from your own reviews.

Each distilled source you reviewed becomes an eval item: the notes you accepted (as written or
edited, in their final form) are what the extractor should find again; the ones you rejected
are what it should avoid. A source where you rejected everything becomes a "should skip" item.
The file contains source text, so it stays on your machine like the vault itself.
"""

from __future__ import annotations

import json
import re
import sqlite3

from essence.index.search import _STOPWORDS
from essence.vault.io import NoteParseError, read_note
from essence.vault.layout import Vault

_CJK_RUN = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]{2,}")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9'-]{3,}")


def _latin(word: str) -> str:
    word = word.lower().split("'")[0]  # forum's -> forum
    return max(word.split("-"), key=len) if "-" in word else word  # three-step -> three/step: keep the longer


def keywords_for(title: str, limit: int = 3) -> list[str]:
    """A few pieces of a title that a re-extracted note would likely contain.

    Only a fallback: items from reviews are matched by meaning when embeddings are on.
    Latin words: the longest non-stopwords, with a 6-letter stem as an alternative.
    CJK phrases: the first and last two characters of each run, either of which may match.
    """
    latin = [_latin(w) for w in _WORD.findall(title)]
    latin = [w for w in dict.fromkeys(latin) if len(w) >= 4 and w not in _STOPWORDS]
    latin.sort(key=len, reverse=True)
    cjk = []
    for run in _CJK_RUN.findall(title):
        head, tail = run[:2], run[-2:]
        cjk.append(head if head == tail else f"{head}|{tail}")
    picked = (cjk[:limit] + latin)[:limit]
    return [w if len(w) <= 6 or not w.isascii() else f"{w}|{w[:6]}" for w in picked]


def export_reviews(vault: Vault, conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT s.id, s.op, s.status, s.payload, s.source_id, e.note_id"
        " FROM suggestions s LEFT JOIN events e ON e.detail = s.op || ' ' || s.id AND e.kind IN ('accept', 'edit')"
        " WHERE s.status IN ('accepted', 'edited', 'rejected') AND s.source_id IS NOT NULL"
        " AND s.op IN ('add', 'update', 'supersede') ORDER BY s.created"
    ).fetchall()
    by_source: dict[str, dict] = {}
    for r in rows:
        payload = json.loads(r["payload"])
        kind, title, body = payload["type"], payload["title"], payload.get("body") or ""
        if r["status"] != "rejected" and r["note_id"]:
            final = conn.execute(
                "SELECT n.type, n.title, f.body FROM notes n JOIN notes_fts f ON f.note_id = n.id WHERE n.id = ?",
                (r["note_id"],),
            ).fetchone()
            if final:
                kind, title, body = final["type"], final["title"], final["body"] or ""
        # "text" is matched by meaning (embeddings); keywords are the fallback without them.
        text = title + "\n" + " ".join(body.split())[:400]
        spec = {"type": kind, "title": title, "text": text, "keywords": keywords_for(title)}
        entry = by_source.setdefault(r["source_id"], {"expected": [], "avoid": []})
        entry["avoid" if r["status"] == "rejected" else "expected"].append(spec)

    # Sources the model found nothing in, and that you never had to review: keep them as
    # "should skip" checks, so a prompt change that starts inventing notes shows up.
    for r in conn.execute("SELECT source_id FROM distill_state WHERE status = 'done' AND suggestions = 0"):
        by_source.setdefault(r["source_id"], {"expected": [], "avoid": []})

    items = []
    for sid, entry in by_source.items():
        src = conn.execute("SELECT title, path, source_kind FROM sources WHERE id = ?", (sid,)).fetchone()
        if not src:
            continue  # the source was deleted since
        try:
            body = read_note(vault.root / src["path"]).body
        except (NoteParseError, OSError):
            continue
        items.append({
            "id": f"review-{sid[-10:].lower()}",
            "kind": src["source_kind"] or "doc",
            "title": src["title"],
            "body": body,
            "expected": entry["expected"],
            "avoid": entry["avoid"],
            "must_not": [],
        })
    return items
