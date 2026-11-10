"""Wiring: vault + config + index connection + embedder + LLM, shared by the CLI and MCP server."""

from __future__ import annotations

import os
import secrets
import sqlite3
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path

from essence.config import Config, load_dotenv
from essence.index.db import IndexKeyError, open_index
from essence.index.reindex import ReindexReport, reindex
from essence.vault.layout import Vault


class App:
    def __init__(self, root: Path, *, embed: bool | None = None):
        self.vault = Vault(root.resolve())
        load_dotenv(Path.cwd() / ".env", self.vault.root / ".env")
        self.config: Config = self.vault.config()
        self._embed_override = embed

    # --- index ---------------------------------------------------------------------------

    @property
    def db_key(self) -> str | None:
        if not self.config.encrypt:
            return None
        key = os.environ.get("ESSENCE_DB_KEY")
        if not key:
            raise IndexKeyError("[index] encrypt = true but ESSENCE_DB_KEY is not set")
        return key

    @contextmanager
    def index(self):
        with open_index(self.vault.index_path, self.db_key) as conn:
            yield conn

    def connect(self) -> sqlite3.Connection:
        from essence.index.db import connect

        return connect(self.vault.index_path, self.db_key)

    # --- models ----------------------------------------------------------------------------

    @cached_property
    def embedder(self):
        if self._embed_override is False:
            return None
        from essence.index.vectors import make_embedder

        return make_embedder(self.config)

    def llm(self, conn: sqlite3.Connection):
        from essence.llm.base import LLMClient, make_provider

        provider = make_provider(self.config)
        if provider is None:
            return None
        return LLMClient(provider, conn, self.config.daily_token_budget)

    # --- operations ------------------------------------------------------------------------

    def refresh(self, conn: sqlite3.Connection) -> ReindexReport:
        """Rebuild derived tables, then attach vectors (cached, so cheap after the first run)."""
        report = reindex(self.vault, conn)
        if self.embedder is not None:
            from essence.index.vectors import embed_notes

            embed_notes(conn, self.embedder)
        return report

    def mcp_token(self) -> str:
        token = os.environ.get("ESSENCE_MCP_TOKEN")
        if token:
            return token
        path = self.vault.state / "mcp_token"
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
        token = secrets.token_urlsafe(32)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(token, encoding="utf-8")
        try:
            path.chmod(0o600)
        except OSError:
            pass
        return token
