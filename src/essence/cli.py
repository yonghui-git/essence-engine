"""Command-line interface: `essence ...`."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Optional

import typer

from essence import __version__
from essence.index import open_index, reindex, search
from essence.vault.io import write_note
from essence.vault.layout import Vault, init_vault, note_type_from_str
from essence.vault.model import KNOWLEDGE_TYPES, Note

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Distill daily inputs into owned, reusable knowledge.")

_state: dict[str, Path] = {}


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
    root = vault or Path(os.environ.get("ESSENCE_VAULT", "."))
    _state["root"] = root


def _open_vault() -> Vault:
    v = Vault(_state["root"].resolve())
    if not v.is_initialized():
        typer.secho(f"No vault at {v.root}. Run `essence init {v.root}` first.", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    return v


@app.command()
def init(path: Annotated[Optional[Path], typer.Argument(help="Where to create the vault.")] = None) -> None:
    """Create a vault (safe to re-run; never overwrites files)."""
    root = path or _state["root"]
    vault, created = init_vault(root)
    with open_index(vault.index_path):
        pass
    typer.echo(f"Vault ready at {vault.root} ({len(created)} new paths).")


@app.command()
def new(
    note_type: Annotated[str, typer.Argument(metavar="TYPE", help=", ".join(t.value for t in KNOWLEDGE_TYPES))],
    title: Annotated[str, typer.Argument()],
    body: Annotated[str, typer.Option("--body", "-b")] = "",
    applies_when: Annotated[Optional[str], typer.Option("--applies-when")] = None,
    link: Annotated[Optional[list[str]], typer.Option("--link", "-l", help="Id of a related note (repeatable).")] = None,
) -> None:
    """Write a note by hand (status active, created_by user) and index it."""
    vault = _open_vault()
    try:
        t = note_type_from_str(note_type)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if t not in KNOWLEDGE_TYPES:
        raise typer.BadParameter("sources are created by importers, not by hand")
    note = Note(type=t, title=title, body=body, applies_when=applies_when, links=link or [])
    path = vault.path_for(note)
    write_note(note, path)
    with open_index(vault.index_path) as conn:
        reindex(vault, conn)
    typer.echo(f"{note.id}  {path.relative_to(vault.root).as_posix()}")


@app.command("reindex")
def reindex_cmd() -> None:
    """Rebuild the search index from the Markdown vault."""
    vault = _open_vault()
    with open_index(vault.index_path) as conn:
        report = reindex(vault, conn)
    typer.echo(
        f"Indexed {report.notes} notes, {report.sources} sources, "
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


@app.command("search")
def search_cmd(
    query: Annotated[str, typer.Argument()],
    limit: Annotated[int, typer.Option("--limit", "-n")] = 10,
    note_type: Annotated[Optional[list[str]], typer.Option("--type", "-t")] = None,
    all_: Annotated[bool, typer.Option("--all", help="Include superseded and invalid notes.")] = False,
) -> None:
    """Full-text search over notes."""
    vault = _open_vault()
    with open_index(vault.index_path) as conn:
        hits = search(conn, query, limit=limit, types=note_type, include_inactive=all_)
    if not hits:
        typer.echo("No matches.")
        return
    for h in hits:
        status = "" if h.status == "active" else f" [{h.status}]"
        typer.secho(f"{h.title}{status}", bold=True)
        typer.echo(f"  {h.type} | {h.note_id} | {h.path}")
        if h.snippet.strip():
            typer.echo(f"  {h.snippet.strip()}")


@app.command()
def stats() -> None:
    """Counts by note type and status."""
    vault = _open_vault()
    with open_index(vault.index_path) as conn:
        rows = conn.execute("SELECT type, status, COUNT(*) AS n FROM notes GROUP BY type, status ORDER BY type, status").fetchall()
        sources = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
        pending = conn.execute("SELECT COUNT(*) FROM suggestions WHERE status = 'pending'").fetchone()[0]
    typer.echo(f"sources: {sources}   inbox: {pending}")
    for r in rows:
        typer.echo(f"{r['type']:<10} {r['status']:<11} {r['n']}")


if __name__ == "__main__":
    app()
