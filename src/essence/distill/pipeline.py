"""Distillation pipeline: source -> redact -> triage -> extract -> reconcile -> inbox."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone

from essence.config import Config
from essence.distill import prompts
from essence.distill.inbox import create_suggestion
from essence.distill.triage import triage
from essence.index.search import hybrid_search
from essence.llm.base import BudgetExceeded, LLMClient, LLMError
from essence.privacy.redact import redact
from essence.vault.io import read_note
from essence.vault.layout import Vault
from essence.vault.model import Note, NoteType, Origin
from essence.vault.repo import NoteRepo

WINDOW_CHARS = 60_000
MAX_CANDIDATES_TOTAL = 12


@dataclass
class SourceOutcome:
    source_id: str
    title: str
    status: str
    triage: str = ""
    candidates: int = 0
    suggestions: int = 0
    error: str = ""


@dataclass
class DistillReport:
    outcomes: list[SourceOutcome] = field(default_factory=list)
    stopped: str = ""

    def count(self, status: str) -> int:
        return sum(1 for o in self.outcomes if o.status == status)

    @property
    def suggestions(self) -> int:
        return sum(o.suggestions for o in self.outcomes)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def pending_sources(conn: sqlite3.Connection, *, force: bool = False, source_ids: list[str] | None = None) -> list[sqlite3.Row]:
    sql = (
        "SELECT s.* FROM sources s LEFT JOIN distill_state d ON d.source_id = s.id"
        " WHERE 1=1"
    )
    params: list = []
    if source_ids:
        sql += f" AND s.id IN ({','.join('?' * len(source_ids))})"
        params += source_ids
    if not force:
        sql += " AND (d.source_id IS NULL OR d.content_hash != s.hash OR d.status IN ('failed', 'budget'))"
    sql += " ORDER BY s.captured"
    return conn.execute(sql, params).fetchall()


def note_dicts(conn: sqlite3.Connection, ids: list[str]) -> list[dict]:
    if not ids:
        return []
    rows = conn.execute(
        f"SELECT n.id, n.type, n.status, n.title, f.body FROM notes n JOIN notes_fts f ON f.note_id = n.id"
        f" WHERE n.id IN ({','.join('?' * len(ids))})",
        ids,
    ).fetchall()
    by_id = {r["id"]: dict(r) for r in rows}
    return [by_id[i] for i in ids if i in by_id]


def _related(conn, embedder, text: str, k: int, extra: list[str] | None = None) -> list[dict]:
    hits = hybrid_search(conn, text[:4000], embedder=embedder, limit=k, mode="any", log=False)
    ids = list(dict.fromkeys([*(extra or []), *(h.note_id for h in hits)]))[: k + len(extra or [])]
    return note_dicts(conn, ids)


def _windows(excerpts: list[tuple[str, str]]) -> list[list[tuple[str, str]]]:
    out, cur, size = [], [], 0
    for exc in excerpts:
        if cur and size + len(exc[1]) > WINDOW_CHARS:
            out.append(cur)
            cur, size = [], 0
        cur.append(exc)
        size += len(exc[1])
    if cur:
        out.append(cur)
    return out


_EXC = re.compile(r"exc_(\d+)$")


def validate_candidates(raw: list[dict], valid_excerpts: set[str], known_ids: set[str], limit: int) -> list[dict]:
    """Drop or repair anything the model got wrong. Every kept candidate has real provenance."""
    out = []
    for c in raw:
        if c.get("type") not in prompts.NOTE_TYPES:
            continue
        title, body = (c.get("title") or "").strip(), (c.get("body") or "").strip()
        if not title or not body:
            continue
        excs = []
        for e in c.get("excerpt_ids") or []:
            m = _EXC.search(str(e).strip())
            if m and f"exc_{m.group(1)}" in valid_excerpts:
                excs.append(f"exc_{m.group(1)}")
        if not excs:
            continue  # no traceable source: not admissible
        rel = c.get("related_note_id") or ""
        relation = c.get("relation") if c.get("relation") in prompts.RELATIONS else "new"
        if rel not in known_ids:
            rel = ""
            if relation != "new":
                relation = "new"
        conf = c.get("confidence")
        out.append(
            {
                "type": c["type"],
                "title": title[:200],
                "body": body,
                "applies_when": (c.get("applies_when") or "").strip(),
                "not_when": (c.get("not_when") or "").strip(),
                "relation": relation,
                "related_note_id": rel,
                "excerpt_ids": list(dict.fromkeys(excs)),
                "confidence": max(0.0, min(1.0, float(conf))) if c["type"] == "claim" and isinstance(conf, (int, float)) else None,
                "expected_outcome": (c.get("expected_outcome") or "").strip() if c["type"] == "decision" else "",
                "why_reusable": (c.get("why_reusable") or "").strip(),
            }
        )
        if len(out) >= limit:
            break
    return out


def validate_decisions(decisions: list[dict], items: list[tuple[str, dict, list[dict]]]) -> dict[str, dict]:
    by_cid = {}
    allowed = {cid: {n["id"] for n in neigh} for cid, _, neigh in items}
    for d in decisions:
        cid = d.get("candidate")
        if cid not in allowed or cid in by_cid:
            continue
        op = d.get("op") if d.get("op") in prompts.OPS else "add"
        target = d.get("target_id") or ""
        links = [i for i in d.get("link_ids") or [] if i in allowed[cid] and i != target]
        if op in ("update", "confirm", "supersede", "link") and target not in allowed[cid]:
            op, target = ("noop" if op == "link" else "add"), ""
        if op == "link" and not links:
            op = "noop"
        by_cid[cid] = {**d, "op": op, "target_id": target, "link_ids": links}
    for cid, cand, _ in items:  # anything the model skipped defaults to add
        by_cid.setdefault(cid, {"candidate": cid, "op": "add", "target_id": "", "link_ids": [], "title": "", "body": "", "reason": ""})
    return by_cid


def _candidate_note(cand: dict, source_id: str) -> Note:
    kwargs = {}
    if cand["type"] == "claim" and cand.get("confidence") is not None:
        kwargs["confidence"] = cand["confidence"]
    if cand["type"] == "decision" and cand.get("expected_outcome"):
        kwargs["expected_outcome"] = cand["expected_outcome"]
    extra = {"why_reusable": cand["why_reusable"]} if cand.get("why_reusable") else {}
    return Note(
        type=NoteType(cand["type"]),
        title=cand["title"],
        body=cand["body"] + "\n",
        applies_when=cand["applies_when"] or None,
        not_when=cand["not_when"] or None,
        sources=[f"{source_id}#{e}" for e in cand["excerpt_ids"]],
        created_by=Origin.AI,
        extra=extra,
        **kwargs,
    )


def distill_source(
    vault: Vault,
    conn: sqlite3.Connection,
    llm: LLMClient,
    config: Config,
    source: sqlite3.Row,
    *,
    embedder=None,
    repo: NoteRepo | None = None,
    schema_text: str = "",
) -> SourceOutcome:
    repo = repo or NoteRepo(vault)
    sid = source["id"]
    out = SourceOutcome(sid, source["title"], "done")
    note = read_note(vault.root / source["path"])

    if (note.never_cloud or source["never_cloud"]) and not llm.is_local:
        out.status = "never_cloud"
        return out

    t = triage(note.body, source["source_kind"] or "", config.min_chars)
    out.triage = t.level
    if t.level == "skip":
        out.status = "skipped"
        out.error = t.reason
        return out

    rows = conn.execute("SELECT id, text FROM excerpts WHERE source_id = ? ORDER BY start_offset", (sid,)).fetchall()
    redactions = 0
    excerpts = []
    for r in rows:
        text = r["text"]
        if config.redact:
            red = redact(text)
            text, redactions = red.text, redactions + red.total
        excerpts.append((r["id"].split("#", 1)[1], text))
    valid = {e for e, _ in excerpts}

    candidates: list[dict] = []
    context = _related(conn, embedder, note.title + "\n" + "\n".join(t for _, t in excerpts[:3]), config.context_notes)
    known_ids = {n["id"] for n in context}
    system = prompts.EXTRACT_SYSTEM.format(
        security=prompts.SECURITY_RULES, schema=schema_text.strip() or "(default)", max_candidates=t.max_candidates
    )
    for window in _windows(excerpts):
        data = llm.call(
            purpose="extract",
            system=system,
            user=prompts.extraction_user(note.title, source["source_kind"] or "", window, context),
            schema=prompts.extraction_schema(),
            model=config.model_extract,
            effort=config.extract_effort or None,
            source_ids=[sid],
            redactions=redactions,
        )
        candidates += validate_candidates(data.get("candidates") or [], valid, known_ids, t.max_candidates)
    candidates = candidates[:MAX_CANDIDATES_TOTAL]
    out.candidates = len(candidates)
    if not candidates:
        return out

    items = []
    for i, cand in enumerate(candidates, 1):
        neighbors = _related(
            conn, embedder, f"{cand['title']}\n{cand['body']}", config.context_notes,
            extra=[cand["related_note_id"]] if cand["related_note_id"] else None,
        )
        items.append((f"c{i}", cand, neighbors))

    if any(neigh for _, _, neigh in items):
        data = llm.call(
            purpose="reconcile",
            system=prompts.RECONCILE_SYSTEM.format(security=prompts.SECURITY_RULES),
            user=prompts.reconcile_user(items),
            schema=prompts.reconcile_schema(),
            model=config.model_extract,
            effort=config.extract_effort or None,
            source_ids=[sid],
        )
        decisions = validate_decisions(data.get("decisions") or [], items)
    else:
        decisions = validate_decisions([], items)

    for cid, cand, _ in items:
        d = decisions[cid]
        op = d["op"]
        if op == "noop":
            continue
        cnote = _candidate_note(cand, sid)
        if op == "update":
            hit = repo.get(d["target_id"])
            if hit is None:
                op = "add"
            else:
                _, target = hit
                cnote = target.model_copy(
                    update={
                        "title": (d.get("title") or target.title).strip(),
                        "body": (d.get("body") or target.body).strip() + "\n",
                        "sources": list(dict.fromkeys([*target.sources, *cnote.sources])),
                    }
                )
        elif op == "add" and (d.get("title") or d.get("body")):
            cnote = cnote.model_copy(
                update={"title": (d.get("title") or cnote.title).strip(), "body": (d.get("body") or cnote.body).strip() + "\n"}
            )
        create_suggestion(
            vault, conn,
            op=op, note=cnote,
            target_id=d["target_id"] or None,
            link_ids=d["link_ids"],
            relation=cand["relation"],
            reason=(d.get("reason") or cand.get("why_reusable") or "").strip(),
            source_id=sid,
        )
        out.suggestions += 1
    return out


def distill(
    vault: Vault,
    conn: sqlite3.Connection,
    llm: LLMClient,
    config: Config,
    *,
    embedder=None,
    limit: int | None = None,
    source_ids: list[str] | None = None,
    force: bool = False,
    on_progress=None,
) -> DistillReport:
    report = DistillReport()
    repo = NoteRepo(vault)
    schema_text = vault.schema_path.read_text(encoding="utf-8") if vault.schema_path.is_file() else ""
    sources = pending_sources(conn, force=force, source_ids=source_ids)
    if limit is not None:
        sources = sources[:limit]
    for source in sources:
        try:
            outcome = distill_source(vault, conn, llm, config, source, embedder=embedder, repo=repo, schema_text=schema_text)
        except BudgetExceeded as exc:
            outcome = SourceOutcome(source["id"], source["title"], "budget", error=str(exc))
            report.stopped = str(exc)
        except LLMError as exc:
            outcome = SourceOutcome(source["id"], source["title"], "failed", error=str(exc))
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO distill_state (source_id, content_hash, status, triage, candidates, suggestions, error, updated)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (outcome.source_id, source["hash"], outcome.status, outcome.triage, outcome.candidates,
                 outcome.suggestions, outcome.error or None, _now()),
            )
        report.outcomes.append(outcome)
        if on_progress:
            on_progress(outcome)
        if report.stopped:
            break
    return report
