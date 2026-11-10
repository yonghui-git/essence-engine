"""MCP server: every AI assistant reads from and saves to the same knowledge base.

Transports:
- stdio (Claude Desktop / Claude Code launch it as a local process; no network)
- streamable HTTP bound to localhost, with a bearer token

Saving never writes canonical notes: `save_to_knowledge_base` stores a raw
source that goes through distillation and the review inbox like any other.
"""

from __future__ import annotations

import hmac
from datetime import datetime, timezone
from typing import Any

from essence.app import App
from essence.capture.base import CapturedItem, SourceStore
from essence.distill.inbox import pending
from essence.index.search import hybrid_search
from essence.vault.io import NoteParseError, read_note

INSTRUCTIONS = """\
This server is the user's personal knowledge base: concepts, claims, insights, decisions and open questions
they have approved, each traceable to sources. Search it before answering questions where the user's own
prior knowledge, decisions or context matter. Cite note titles when you use them. To remember something from
this conversation, call save_to_knowledge_base; it will be reviewed by the user before it becomes a note."""


class KnowledgeTools:
    """Plain functions behind the MCP tools (easy to test without a transport)."""

    def __init__(self, app: App):
        self.app = app

    def _event(self, conn, kind: str, note_id: str | None = None, detail: str = "") -> None:
        with conn:
            conn.execute(
                "INSERT INTO events (at, kind, note_id, detail) VALUES (?, ?, ?, ?)",
                (datetime.now(timezone.utc).isoformat(timespec="seconds"), kind, note_id, detail),
            )

    def search(self, query: str, limit: int = 8, types: list[str] | None = None) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 25))
        with self.app.index() as conn:
            hits = hybrid_search(conn, query, embedder=self.app.embedder, limit=limit, types=types or None, log=False)
            self._event(conn, "mcp_search", detail=f"hits={len(hits)}")
        return [
            {"id": h.note_id, "type": h.type, "title": h.title, "snippet": h.snippet.strip(), "match": h.via}
            for h in hits
        ]

    def read(self, note_id: str) -> dict[str, Any]:
        with self.app.index() as conn:
            row = conn.execute("SELECT path FROM notes WHERE id = ?", (note_id,)).fetchone()
            if row is None:
                return {"error": f"no note with id {note_id}"}
            try:
                note = read_note(self.app.vault.root / row["path"])
            except (NoteParseError, OSError) as exc:
                return {"error": str(exc)}
            evidence = []
            for ref in note.sources[:8]:
                exc_row = conn.execute(
                    "SELECT e.text, s.title, s.origin_ref FROM excerpts e JOIN sources s ON s.id = e.source_id WHERE e.id = ?",
                    (ref,),
                ).fetchone()
                if exc_row:
                    evidence.append({"ref": ref, "source": exc_row["title"], "origin": exc_row["origin_ref"],
                                     "excerpt": exc_row["text"][:800]})
            links = conn.execute(
                "SELECT n.id, n.title FROM edges e JOIN notes n ON n.id = e.dst WHERE e.src = ? AND e.kind = 'links'",
                (note_id,),
            ).fetchall()
            self._event(conn, "mcp_read", note_id)
        fm = note.frontmatter()
        return {
            **{k: fm[k] for k in ("id", "type", "title", "status", "applies_when", "not_when", "version",
                                  "confidence", "expected_outcome", "review_on", "actual_outcome", "lesson", "superseded_by") if k in fm},
            "body": note.body,
            "linked_notes": [{"id": r["id"], "title": r["title"]} for r in links],
            "evidence": evidence,
        }

    def save(self, title: str, content: str, source: str = "assistant") -> dict[str, Any]:
        content = content.strip()
        if len(content) < 20:
            return {"error": "content is too short to be worth saving"}
        now = datetime.now(timezone.utc).replace(microsecond=0)
        item = CapturedItem(
            source_kind="chat",
            origin_ref=f"mcp:{source}:{now.isoformat()}:{abs(hash(content)) % 10**8}",
            title=title.strip()[:200] or "Saved from assistant",
            body=content + "\n",
            captured=now,
            subfolder="mcp",
            extra={"provider": f"mcp:{source}"},
        )
        status, note = SourceStore(self.app.vault, self.app.config).store(item)
        with self.app.index() as conn:
            self._event(conn, "mcp_save", note.id)
        return {"saved_source_id": note.id, "status": status,
                "next": "It will be distilled and shown to the user in their review inbox before becoming a note."}

    def decisions(self, due_only: bool = True) -> list[dict[str, Any]]:
        """Active decisions with expected outcomes; by default only those due for review."""
        from essence.decisions import list_decisions
        from essence.vault.repo import NoteRepo

        repo = NoteRepo(self.app.vault)
        with self.app.index() as conn:
            rows = list_decisions(conn, self.app.vault)
            self._event(conn, "mcp_decisions", detail=f"due_only={due_only}")
        out = []
        for r in rows:
            if due_only and r.state != "due":
                continue
            found = repo.get(r.id)
            note = found[1] if found else None
            out.append({"id": r.id, "title": r.title, "state": r.state, "review_on": r.review_on,
                        "expected_outcome": note.expected_outcome if note else None,
                        "actual_outcome": note.actual_outcome if note else None})
        return out

    def note_markdown(self, note_id: str) -> str:
        """The note file as stored in the vault (frontmatter + Markdown), for MCP resources."""
        with self.app.index() as conn:
            row = conn.execute("SELECT path FROM notes WHERE id = ?", (note_id,)).fetchone()
            if row is None:
                raise ValueError(f"no note with id {note_id}")
            self._event(conn, "mcp_read", note_id, "resource")
        return (self.app.vault.root / row["path"]).read_text(encoding="utf-8")

    def inbox(self) -> list[dict[str, Any]]:
        items, _ = pending(self.app.vault)
        return [{"id": s.id, "op": s.op, "type": s.note.type.value, "title": s.note.title, "reason": s.reason}
                for s in items[:50]]


def build_server(app: App):
    from mcp.server.mcpserver import MCPServer

    tools = KnowledgeTools(app)
    server = MCPServer(name="essence-engine", instructions=INSTRUCTIONS)

    @server.tool(description="Search the user's knowledge base (hybrid keyword + semantic). Returns note ids, titles and snippets.")
    def search_knowledge(query: str, limit: int = 8, types: list[str] | None = None) -> list[dict[str, Any]]:
        return tools.search(query, limit, types)

    @server.tool(description="Read one note in full, with its supporting source excerpts and linked notes.")
    def read_note_by_id(note_id: str) -> dict[str, Any]:
        return tools.read(note_id)

    @server.tool(description="Save something worth remembering from this conversation. It is distilled and reviewed by the user before it becomes a note.")
    def save_to_knowledge_base(title: str, content: str, source: str = "assistant") -> dict[str, Any]:
        return tools.save(title, content, source)

    @server.tool(description="List suggestions waiting in the user's review inbox.")
    def list_inbox() -> list[dict[str, Any]]:
        return tools.inbox()

    @server.tool(description="The user's decisions with expected outcomes and review dates. due_only=true (default) "
                             "returns only those whose review date has come; help the user compare expectation and outcome.")
    def list_decisions(due_only: bool = True) -> list[dict[str, Any]]:
        return tools.decisions(due_only)

    @server.resource("note://{note_id}", name="note", mime_type="text/markdown",
                     description="A note from the user's knowledge base, as its Markdown file with YAML frontmatter.")
    def note_resource(note_id: str) -> str:
        return tools.note_markdown(note_id)

    return server


class BearerAuth:
    """ASGI middleware: reject HTTP requests without the right bearer token."""

    def __init__(self, app, token: str):
        self.app = app
        self.token = token.encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope.get("headers") or [])
            auth = headers.get(b"authorization", b"")
            ok = auth.startswith(b"Bearer ") and hmac.compare_digest(auth[7:].strip(), self.token)
            if not ok:
                await send({"type": "http.response.start", "status": 401,
                            "headers": [(b"content-type", b"text/plain"), (b"www-authenticate", b"Bearer")]})
                await send({"type": "http.response.body", "body": b"unauthorized"})
                return
        await self.app(scope, receive, send)


def serve(app: App, transport: str = "stdio", host: str | None = None, port: int | None = None) -> None:
    server = build_server(app)
    if transport == "stdio":
        server.run("stdio")
        return
    host = host or app.config.mcp_host
    port = port or app.config.mcp_port
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit("refusing to bind the MCP server to a non-loopback address")
    import uvicorn

    asgi = BearerAuth(server.streamable_http_app(host=host), app.mcp_token())
    uvicorn.run(asgi, host=host, port=port, log_level="warning")
