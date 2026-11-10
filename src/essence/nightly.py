"""The nightly job: distill what arrived, propose merges, write last week's digest once.

Meant for a scheduler (Windows Task Scheduler, cron, launchd). Every step reports and never
stops the others; LLM spend stays inside the daily token budget, and the digest of a finished
week is written only once, so running it every night costs no more than running it weekly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta


@dataclass
class NightlyReport:
    lines: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def add(self, step: str, text: str) -> None:
        self.lines.append(f"{step:<12} {text}")

    def fail(self, step: str, exc: Exception) -> None:
        self.problems.append(f"{step}: {exc}")
        self.add(step, f"FAILED: {exc}")


def run_nightly(app, *, distill: bool = True, summary: bool = True, today: date | None = None) -> NightlyReport:
    from essence.consolidate.digest import Week, write_digest
    from essence.consolidate.merge import propose_merges
    from essence.decisions import list_decisions
    from essence.distill.pipeline import distill as run_distill

    today = today or date.today()
    rep = NightlyReport()
    with app.index() as conn:
        app.refresh(conn)
        llm = app.llm(conn)

        if not distill:
            rep.add("distill", "skipped (--no-distill)")
        elif llm is None:
            rep.add("distill", "skipped: no LLM provider configured")
        else:
            try:
                d = run_distill(app.vault, conn, llm, app.config, embedder=app.embedder)
                text = (f"{len(d.outcomes)} sources -> {d.suggestions} suggestions "
                        f"({d.count('skipped')} skipped, {d.count('failed')} failed)")
                rep.add("distill", text + (f"; stopped: {d.stopped}" if d.stopped else ""))
                app.refresh(conn)
            except Exception as exc:  # noqa: BLE001 - report and carry on
                rep.fail("distill", exc)

        if app.embedder is None:
            rep.add("merges", "skipped: embeddings are off")
        else:
            try:
                ids = propose_merges(app.vault, conn, app.embedder.model_name)
                rep.add("merges", f"{len(ids)} merge suggestions")
            except Exception as exc:  # noqa: BLE001
                rep.fail("merges", exc)

        try:
            due = [d for d in list_decisions(conn, app.vault, today=today) if d.state == "due"]
            rep.add("decisions", f"{len(due)} due for review" + (": " + "; ".join(d.title[:50] for d in due[:3]) if due else ""))
        except Exception as exc:  # noqa: BLE001
            rep.fail("decisions", exc)

        last_week = Week.of(today - timedelta(days=7))
        path = app.vault.digests / f"{last_week.label}.md"
        if path.exists():
            rep.add("digest", f"{last_week.label} already written")
        else:
            try:
                write_digest(app.vault, conn, last_week, llm=llm if summary else None, config=app.config)
                rep.add("digest", f"wrote digests/{last_week.label}.md")
            except Exception as exc:  # noqa: BLE001
                rep.fail("digest", exc)

        pending = conn.execute("SELECT COUNT(*) FROM suggestions WHERE status = 'pending'").fetchone()[0]
        rep.add("inbox", f"{pending} suggestions waiting for review")
        with conn:
            conn.execute("INSERT INTO events (at, kind, detail) VALUES (datetime('now'), 'nightly', ?)",
                         ("ok" if not rep.problems else f"{len(rep.problems)} problems",))
    return rep
