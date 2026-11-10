"""Command-line interface: `essence ...`."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Annotated, Optional

import typer

from essence import __version__
from essence.app import App
from essence.index.db import IndexKeyError
from essence.vault.io import write_note
from essence.vault.layout import init_vault, note_type_from_str
from essence.vault.model import KNOWLEDGE_TYPES, Note

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Distill daily inputs into owned, reusable knowledge.")

_state: dict[str, Path] = {}

# Windows consoles default to a legacy code page; keep output readable.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass


def _version(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    vault: Annotated[
        Optional[Path],
        typer.Option("--vault", "-V", help="Vault root. Defaults to $ESSENCE_VAULT, then the current directory."),
    ] = None,
    version: Annotated[bool, typer.Option("--version", callback=_version, is_eager=True)] = False,
) -> None:
    _state["root"] = vault or Path(os.environ.get("ESSENCE_VAULT", "."))


def _app() -> App:
    a = App(_state["root"])
    if not a.vault.is_initialized():
        _fail(f"No vault at {a.vault.root}. Run `essence init {a.vault.root}` first.")
    return a


def _fail(msg: str, code: int = 1):
    typer.secho(msg, fg=typer.colors.RED, err=True)
    raise typer.Exit(code)


def _warn(msg: str) -> None:
    typer.secho(msg, fg=typer.colors.YELLOW, err=True)


@contextmanager
def _index(a: App):
    try:
        conn = a.connect()
    except IndexKeyError as exc:
        _fail(str(exc))
    try:
        yield conn
    finally:
        conn.close()


def _llm_or_fail(a: App, conn):
    try:
        llm = a.llm(conn)
    except Exception as exc:  # noqa: BLE001
        _fail(f"LLM provider unavailable: {exc}")
    if llm is None:
        _fail("No LLM provider configured ([llm] provider in .essence/config.toml).")
    if llm.provider.name == "anthropic" and not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        _warn("ANTHROPIC_API_KEY is not set; relying on another Anthropic credential source.")
    return llm


# --- setup -------------------------------------------------------------------------------------


@app.command()
def init(path: Annotated[Optional[Path], typer.Argument(help="Where to create the vault.")] = None) -> None:
    """Create a vault (safe to re-run; never overwrites files)."""
    vault, created = init_vault(path or _state["root"])
    _state["root"] = vault.root
    with _index(App(vault.root)):
        pass
    typer.echo(f"Vault ready at {vault.root} ({len(created)} new paths).")
    typer.echo(f"Settings: {vault.config_path}")


@app.command("import")
def import_cmd(
    kind: Annotated[str, typer.Argument(help="chatgpt | claude | chat (auto-detect) | web | docs")],
    path: Annotated[Path, typer.Argument(help="Export .zip/folder, a web-clip folder, or a document/folder.")],
    no_docling: Annotated[bool, typer.Option("--no-docling", help="Use the built-in PDF/DOCX parsers.")] = False,
    distill_now: Annotated[bool, typer.Option("--distill", help="Distill the imported items right away.")] = False,
) -> None:
    """Import AI chat exports, web clips or documents as raw sources."""
    from essence.capture import import_path
    from essence.capture.base import SourceStore

    a = _app()
    try:
        result = import_path(SourceStore(a.vault, a.config), kind, path, use_docling=not no_docling)
    except (FileNotFoundError, ValueError) as exc:
        _fail(str(exc))
    typer.echo(f"Imported: {result.summary()}")
    for err in result.errors[:20]:
        _warn(f"  {err}")
    with _index(a) as conn:
        a.refresh(conn)
    if distill_now and (result.new or result.updated):
        _run_distill(a, source_ids=result.new + result.updated)


@app.command()
def new(
    note_type: Annotated[str, typer.Argument(metavar="TYPE", help=", ".join(t.value for t in KNOWLEDGE_TYPES))],
    title: Annotated[str, typer.Argument()],
    body: Annotated[str, typer.Option("--body", "-b")] = "",
    applies_when: Annotated[Optional[str], typer.Option("--applies-when")] = None,
    link: Annotated[Optional[list[str]], typer.Option("--link", "-l", help="Id of a related note (repeatable).")] = None,
    review_on: Annotated[Optional[str], typer.Option("--review-on", help="Decisions: YYYY-MM-DD to revisit.")] = None,
) -> None:
    """Write a note by hand (status active, created_by user) and index it."""
    a = _app()
    try:
        t = note_type_from_str(note_type)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if t not in KNOWLEDGE_TYPES:
        raise typer.BadParameter("sources are created by importers, not by hand")
    note = Note(
        type=t, title=title, body=body, applies_when=applies_when, links=link or [],
        review_on=date.fromisoformat(review_on) if review_on else None,
    )
    path = a.vault.path_for(note)
    write_note(note, path)
    with _index(a) as conn:
        a.refresh(conn)
    typer.echo(f"{note.id}  {a.vault.relpath(path)}")


@app.command("reindex")
def reindex_cmd() -> None:
    """Rebuild the search index from the Markdown vault."""
    a = _app()
    with _index(a) as conn:
        report = a.refresh(conn)
    typer.echo(
        f"Indexed {report.notes} notes, {report.sources} sources ({report.excerpts} excerpts), "
        f"{report.versions} past versions, {report.edges} edges."
    )
    for label, items, color in [
        ("error", report.errors, typer.colors.RED),
        ("duplicate id", report.duplicates, typer.colors.YELLOW),
        ("dangling link", report.dangling, typer.colors.YELLOW),
    ]:
        for item in items:
            typer.secho(f"  {label}: {item}", fg=color, err=True)
    if report.errors:
        raise typer.Exit(2)


# --- use ------------------------------------------------------------------------------------------


@app.command("search")
def search_cmd(
    query: Annotated[str, typer.Argument()],
    limit: Annotated[int, typer.Option("--limit", "-n")] = 10,
    note_type: Annotated[Optional[list[str]], typer.Option("--type", "-t")] = None,
    all_: Annotated[bool, typer.Option("--all", help="Include superseded and invalid notes.")] = False,
    text_only: Annotated[bool, typer.Option("--text-only", help="Full-text only (no vectors or links).")] = False,
) -> None:
    """Search notes (hybrid: keywords + meaning + links)."""
    from essence.index.search import hybrid_search, search

    a = _app()
    with _index(a) as conn:
        if text_only:
            hits = search(conn, query, limit=limit, types=note_type, include_inactive=all_)
        else:
            hits = hybrid_search(conn, query, embedder=a.embedder, limit=limit, types=note_type, include_inactive=all_)
    if not hits:
        typer.echo("No matches.")
        return
    for h in hits:
        status = "" if h.status == "active" else f" [{h.status}]"
        typer.secho(f"{h.title}{status}", bold=True)
        typer.echo(f"  {h.type} | {h.note_id} | {h.path} | via {h.via}")
        if h.snippet.strip():
            typer.echo(f"  {h.snippet.strip()}")


@app.command()
def show(note_id: Annotated[str, typer.Argument()]) -> None:
    """Print a note with its supporting excerpts."""
    from essence.mcp_server.server import KnowledgeTools

    data = KnowledgeTools(_app()).read(note_id)
    if "error" in data:
        _fail(data["error"])
    typer.secho(f"{data['title']}  ({data['type']}, {data['status']}, v{data['version']})", bold=True)
    if data.get("applies_when"):
        typer.echo(f"Applies when: {data['applies_when']}")
    typer.echo("")
    typer.echo(data["body"].strip())
    for ev in data["evidence"]:
        typer.echo("")
        typer.secho(f"  [{ev['ref']}] {ev['source']}", fg=typer.colors.CYAN)
        typer.echo("  " + " ".join(ev["excerpt"].split())[:400])
    for ln in data["linked_notes"]:
        typer.echo(f"  -> {ln['title']} ({ln['id']})")


# --- distillation and review ---------------------------------------------------------------------------


def _run_distill(a: App, *, limit=None, source_ids=None, force=False) -> None:
    from essence.distill.pipeline import distill

    with _index(a) as conn:
        llm = _llm_or_fail(a, conn)

        def progress(o):
            color = {"done": None, "skipped": typer.colors.BRIGHT_BLACK, "failed": typer.colors.RED}.get(o.status)
            detail = f"{o.suggestions} suggestions" if o.status == "done" else (o.error or o.status)
            typer.secho(f"  {o.status:<11} {o.title[:60]:<60} {detail}", fg=color)

        report = distill(a.vault, conn, llm, a.config, embedder=a.embedder, limit=limit,
                         source_ids=source_ids, force=force, on_progress=progress)
    typer.echo(
        f"Distilled {report.count('done')} sources -> {report.suggestions} suggestions in the inbox "
        f"({report.count('skipped')} skipped, {report.count('failed')} failed, {report.count('never_cloud')} never-cloud)."
    )
    if report.stopped:
        _warn(f"Stopped: {report.stopped}")
    if report.suggestions:
        typer.echo("Review them with `essence review`.")


@app.command("distill")
def distill_cmd(
    limit: Annotated[Optional[int], typer.Option("--limit", "-n", help="At most this many sources.")] = None,
    source: Annotated[Optional[list[str]], typer.Option("--source", "-s", help="Only this source id (repeatable).")] = None,
    force: Annotated[bool, typer.Option("--force", help="Re-distill sources already processed.")] = False,
) -> None:
    """Turn new or changed sources into suggested notes in the review inbox."""
    _run_distill(_app(), limit=limit, source_ids=source, force=force)


def _describe(a: App, sug) -> list[str]:
    lines = [f"[{sug.op}] {sug.note.type.value}: {sug.note.title}"]
    if sug.reason:
        lines.append(f"  why: {sug.reason}")
    if sug.target_id:
        with _index(a) as conn:
            row = conn.execute("SELECT title FROM notes WHERE id = ?", (sug.target_id,)).fetchone()
        lines.append(f"  target: {row['title'] if row else sug.target_id}")
    lines.append(f"  file: {a.vault.relpath(sug.path)}  id: {sug.id}")
    return lines


@app.command()
def inbox() -> None:
    """List suggestions waiting for review."""
    from essence.distill.inbox import pending

    a = _app()
    items, errors = pending(a.vault)
    for e in errors:
        _warn(e)
    if not items:
        typer.echo("Inbox is empty.")
        return
    for sug in items:
        typer.echo("\n".join(_describe(a, sug)))
    typer.echo(f"\n{len(items)} pending.")


def _apply(a: App, refs: list[str], accept_all: bool, action: str) -> None:
    from essence.distill import apply as ap
    from essence.distill.inbox import find, pending
    from essence.vault.repo import NoteRepo

    items = pending(a.vault)[0] if accept_all else [find(a.vault, r) for r in refs]
    if any(s is None for s in items):
        _fail("No unique pending suggestion matches " + ", ".join(r for r, s in zip(refs, items) if s is None))
    repo = NoteRepo(a.vault)
    with _index(a) as conn:
        for sug in items:
            try:
                res = ap.accept(a.vault, conn, sug, repo) if action == "accept" else ap.reject(a.vault, conn, sug)
            except ap.ApplyError as exc:
                _warn(f"{sug.id}: {exc}")
                continue
            typer.echo(f"{res.status:<9} {sug.op:<9} {sug.note.title}")
        a.refresh(conn)


@app.command()
def accept(
    refs: Annotated[Optional[list[str]], typer.Argument(help="Suggestion ids (or unique suffixes / file names).")] = None,
    all_: Annotated[bool, typer.Option("--all", help="Accept everything in the inbox.")] = False,
) -> None:
    """Accept suggestions (your edits to the inbox files are kept)."""
    if not refs and not all_:
        _fail("Name suggestions to accept, or pass --all.")
    _apply(_app(), refs or [], all_, "accept")


@app.command()
def reject(refs: Annotated[list[str], typer.Argument(help="Suggestion ids (or unique suffixes / file names).")]) -> None:
    """Reject suggestions."""
    _apply(_app(), refs, False, "reject")


@app.command()
def review() -> None:
    """Go through the inbox one by one: accept, edit, reject or skip."""
    from essence.distill import apply as ap
    from essence.distill.inbox import pending, read_suggestion
    from essence.vault.repo import NoteRepo

    a = _app()
    items, _ = pending(a.vault)
    if not items:
        typer.echo("Inbox is empty.")
        return
    repo = NoteRepo(a.vault)
    done = 0
    with _index(a) as conn:
        for i, sug in enumerate(items, 1):
            typer.echo("")
            typer.secho(f"({i}/{len(items)})", fg=typer.colors.BRIGHT_BLACK)
            for line in _describe(a, sug):
                typer.echo(line)
            typer.echo("")
            typer.echo(sug.note.body.strip())
            choice = typer.prompt("[a]ccept  [e]dit  [r]eject  [s]kip  [q]uit", default="s").strip().lower()[:1]
            if choice == "q":
                break
            if choice == "e":
                edited = typer.edit(sug.path.read_text(encoding="utf-8"), extension=".md")
                if edited is not None:
                    sug.path.write_text(edited, encoding="utf-8", newline="\n")
                    sug = read_suggestion(sug.path)
                choice = "a"
            try:
                if choice == "a":
                    res = ap.accept(a.vault, conn, sug, repo)
                elif choice == "r":
                    res = ap.reject(a.vault, conn, sug)
                else:
                    continue
            except ap.ApplyError as exc:
                _warn(str(exc))
                continue
            done += 1
            typer.secho(f"  -> {res.status}", fg=typer.colors.GREEN if res.status != "rejected" else typer.colors.YELLOW)
        a.refresh(conn)
    typer.echo(f"\nReviewed {done}.")


# --- reflection ---------------------------------------------------------------------------------------


@app.command()
def digest(
    week: Annotated[Optional[str], typer.Option("--week", "-w", help="ISO week like 2026-W46 (default: this week).")] = None,
    no_llm: Annotated[bool, typer.Option("--no-llm", help="Skip the AI-written summary.")] = False,
) -> None:
    """Write the weekly 'what I learned and decided' digest to digests/."""
    from essence.consolidate.digest import Week, write_digest

    a = _app()
    w = Week.parse(week) if week else Week.of(date.today())
    with _index(a) as conn:
        a.refresh(conn)
        llm = None if no_llm else a.llm(conn)
        path, text = write_digest(a.vault, conn, w, llm=llm, config=a.config)
        with conn:
            conn.execute("INSERT INTO events (at, kind, detail) VALUES (datetime('now'), 'digest_open', ?)", (w.label,))
    typer.echo(text)
    typer.secho(f"\nSaved to {path}", fg=typer.colors.BRIGHT_BLACK)


@app.command()
def changes(limit: Annotated[int, typer.Option("--limit", "-n")] = 30) -> None:
    """The 'changed my mind' log: every revised or superseded note, newest first."""
    from essence.consolidate.digest import changes as all_changes

    a = _app()
    with _index(a) as conn:
        items = all_changes(a.vault, conn)[:limit]
    if not items:
        typer.echo("No changes yet.")
    for c in items:
        typer.echo(f"{c.when[:10]}  {c.kind:<10} {c.before}  ->  {c.after}")
        if c.reason:
            typer.secho(f"            {c.reason}", fg=typer.colors.BRIGHT_BLACK)


@app.command()
def consolidate(
    threshold: Annotated[float, typer.Option(help="Cosine similarity for near-duplicates.")] = 0.9,
    use_llm: Annotated[bool, typer.Option("--llm/--no-llm", help="Let the model confirm and write merges.")] = False,
    max_pairs: Annotated[int, typer.Option("--max")] = 10,
) -> None:
    """Propose merges of near-duplicate notes (lands in the inbox)."""
    from essence.consolidate.merge import propose_merges

    a = _app()
    if a.embedder is None:
        _fail("Consolidation needs embeddings ([embed] provider = \"fastembed\").")
    with _index(a) as conn:
        a.refresh(conn)
        llm = _llm_or_fail(a, conn) if use_llm else None
        ids = propose_merges(a.vault, conn, a.embedder.model_name, threshold=threshold, max_pairs=max_pairs, llm=llm, config=a.config)
    typer.echo(f"{len(ids)} merge suggestions added to the inbox.")


# --- trust ----------------------------------------------------------------------------------------------


@app.command()
def delete(
    source_id: Annotated[str, typer.Argument(help="Source id (src_...).")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask for confirmation.")] = False,
) -> None:
    """Delete a source and remove it from every note derived from it."""
    from essence.ops import delete_source

    a = _app()
    with _index(a) as conn:
        row = conn.execute("SELECT title, path FROM sources WHERE id = ?", (source_id,)).fetchone()
        if row is None:
            _fail(f"No source {source_id}.")
        cited = conn.execute("SELECT COUNT(DISTINCT src) FROM edges WHERE dst = ? AND kind = 'derived_from'", (source_id,)).fetchone()[0]
        if not yes:
            typer.confirm(f"Delete '{row['title']}' ({row['path']}), cited by {cited} notes? This cannot be undone.", abort=True)
        rep = delete_source(a.vault, conn, source_id)
        a.refresh(conn)
    typer.echo(
        f"Deleted. Notes updated: {len(rep.notes_updated)}; orphaned notes sent to the inbox: {len(rep.orphans)}; "
        f"pending suggestions discarded: {len(rep.suggestions_removed)}."
    )


@app.command("encrypt-index")
def encrypt_index_cmd() -> None:
    """Encrypt the existing index with SQLCipher (passphrase from ESSENCE_DB_KEY)."""
    from essence.index.db import is_encrypted
    from essence.ops import encrypt_index

    a = _app()
    key = os.environ.get("ESSENCE_DB_KEY")
    if not key:
        _fail("Set ESSENCE_DB_KEY (environment or .env) first.")
    if is_encrypted(a.vault.index_path):
        typer.echo("Index is already encrypted.")
    else:
        try:
            encrypt_index(a.vault.index_path, key)
        except ImportError:
            _fail("SQLCipher is not installed: pip install 'essence-engine[encrypt]'")
        typer.echo("Index encrypted.")
    text = a.vault.config_path.read_text(encoding="utf-8")
    if "encrypt = false" in text:
        a.vault.config_path.write_text(text.replace("encrypt = false", "encrypt = true"), encoding="utf-8")
        typer.echo("Set [index] encrypt = true in .essence/config.toml.")


@app.command()
def serve(
    http: Annotated[bool, typer.Option("--http", help="Serve streamable HTTP on localhost instead of stdio.")] = False,
    port: Annotated[Optional[int], typer.Option("--port")] = None,
    show_token: Annotated[bool, typer.Option("--show-token", help="Print the HTTP bearer token and exit.")] = False,
) -> None:
    """Run the MCP server so AI assistants can search and save to the knowledge base."""
    from essence.mcp_server.server import serve as run

    a = _app()
    if show_token:
        typer.echo(a.mcp_token())
        return
    if http:
        typer.secho(
            f"MCP on http://{a.config.mcp_host}:{port or a.config.mcp_port}/mcp "
            "(bearer token: `essence serve --show-token`)",
            err=True,
        )
    run(a, "streamable-http" if http else "stdio", port=port)


# --- measurement --------------------------------------------------------------------------------------


@app.command()
def stats() -> None:
    """Counts by note type and status."""
    a = _app()
    with _index(a) as conn:
        rows = conn.execute("SELECT type, status, COUNT(*) AS n FROM notes GROUP BY type, status ORDER BY type, status").fetchall()
        sources = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
        pending = conn.execute("SELECT COUNT(*) FROM suggestions WHERE status = 'pending'").fetchone()[0]
    typer.echo(f"sources: {sources}   inbox: {pending}")
    for r in rows:
        typer.echo(f"{r['type']:<10} {r['status']:<11} {r['n']}")


@app.command("metrics")
def metrics_cmd() -> None:
    """Pilot metrics: review outcomes, return visits and reuse."""
    from essence.ops import metrics

    a = _app()
    with _index(a) as conn:
        m = metrics(conn)
    typer.echo(f"Suggestions reviewed: {m.reviewed}  (pending {m.pending})")
    typer.echo(f"  accepted as-is: {m.accepted} ({m.share(m.accepted)})")
    typer.echo(f"  edited:         {m.edited} ({m.share(m.edited)}), mean {m.mean_edit_chars:.0f} chars changed")
    typer.echo(f"  rejected:       {m.rejected} ({m.share(m.rejected)})")
    typer.echo(f"Active days in the last 28: {m.active_days_28}")
    typer.echo(f"Reuse events in the last 7 days: {m.reuse_events_7}")
    typer.echo(f"Notes: {m.notes_active} active, {m.notes_superseded} superseded; sources: {m.sources}")


@app.command()
def cost(days: Annotated[int, typer.Option("--days")] = 30) -> None:
    """LLM spend from the outbound-call log, and a monthly projection."""
    from essence.llm.base import tokens_today
    from essence.ops import cost_report

    a = _app()
    with _index(a) as conn:
        rep = cost_report(conn, days)
        today = tokens_today(conn)
    if not rep.rows:
        typer.echo(f"No LLM calls in the last {days} days.")
    for purpose, model, calls, tin, tout, usd in rep.rows:
        typer.echo(f"{purpose:<12} {model:<22} {calls:>5} calls  {tin:>10,} in  {tout:>9,} out  ${usd:,.4f}")
    typer.echo(f"Total ${rep.total_usd:,.4f} over {days} days; {rep.sources_distilled} sources distilled.")
    if rep.sources_distilled:
        typer.echo(f"Per source ${rep.per_item:,.4f}; at 30 items/day ≈ ${rep.monthly_at(30):,.2f}/month.")
    budget = a.config.daily_token_budget
    typer.echo(f"Tokens today: {today:,}" + (f" of {budget:,} budget" if budget else ""))


@app.command("eval")
def eval_cmd(
    file: Annotated[Optional[Path], typer.Option("--file", help="Golden set JSON (default: the built-in set).")] = None,
) -> None:
    """Score distillation quality on a golden set with the configured model (makes real LLM calls)."""
    from essence.evaluation.run import load_golden, run_eval
    from essence.llm.base import make_provider

    a = _app()
    provider = make_provider(a.config)
    if provider is None:
        _fail("No LLM provider configured.")
    items = load_golden(file)
    typer.echo(f"Evaluating {len(items)} items with {a.config.model_extract} ...")

    def show_item(s):
        mark = "ok " if (s.found == s.expected and not s.violations and not s.error) else "-- "
        typer.echo(f"{mark}{s.id:<24} found {s.found}/{s.expected}, {s.candidates} candidates"
                   + (f", VIOLATIONS {s.violations}" if s.violations else "") + (f", error: {s.error}" if s.error else ""))
        for t in s.titles:
            typer.secho(f"      - {t}", fg=typer.colors.BRIGHT_BLACK)

    rep = run_eval(provider, a.config, items, on_item=show_item)
    typer.echo(
        f"\nRecall {rep.recall:.0%}  precision {rep.precision:.0%}  skip accuracy {rep.skip_accuracy:.0%}  "
        f"injection violations {rep.violations}  cost ${rep.cost_usd:.4f}"
    )


@app.command()
def doctor() -> None:
    """Check the setup: vault, index, encryption, LLM credentials, embeddings, MCP."""
    from essence.index.db import is_encrypted

    a = _app()
    c = a.config

    def line(ok: bool | None, text: str) -> None:
        mark, color = {True: ("ok  ", typer.colors.GREEN), False: ("FAIL", typer.colors.RED), None: ("warn", typer.colors.YELLOW)}[ok]
        typer.secho(f"[{mark}] {text}", fg=color)

    line(True, f"vault at {a.vault.root}")
    enc = is_encrypted(a.vault.index_path)
    if c.encrypt:
        line(enc and bool(os.environ.get("ESSENCE_DB_KEY")), "index encrypted with SQLCipher" if enc else "encrypt = true but the index is not encrypted (run `essence encrypt-index`)")
    else:
        line(None, "index not encrypted (set ESSENCE_DB_KEY and run `essence encrypt-index`)")
    try:
        with a.index() as conn:
            n = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
        line(True, f"index opens ({n} notes)")
    except IndexKeyError as exc:
        line(False, str(exc))
    if c.llm_provider == "anthropic":
        has_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
        line(has_key or None, f"Anthropic credentials {'found' if has_key else 'not in environment'}; "
                              f"extract model {c.model_extract}, synth model {c.model_synth}")
    else:
        line(True, f"LLM provider: {c.llm_provider}")
    line(a.embedder is not None or None, f"embeddings: {c.embed_model if a.embedder else 'off (keyword + link search only)'}")
    line(True if c.redact else None, f"redaction before LLM calls: {'on' if c.redact else 'OFF'}")
    try:
        import mcp  # noqa: F401
        line(True, f"MCP server available (stdio, or HTTP on {c.mcp_host}:{c.mcp_port} with a bearer token)")
    except ImportError:
        line(False, "mcp package missing")
    try:
        import docling  # noqa: F401  # type: ignore[import-not-found]
        line(True, "Docling installed for document parsing")
    except ImportError:
        line(None, "Docling not installed; using built-in PDF/DOCX parsers (pip install 'essence-engine[docs]')")


if __name__ == "__main__":
    app()
