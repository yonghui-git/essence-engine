"""Provider-neutral LLM interface, with cost logging and a daily token budget.

Every call goes through `LLMClient.call`, which:
- refuses to run when today's token budget is spent,
- records the call (model, tokens, cost, source ids, redaction count) in
  `llm_calls`, the outbound-call log kept with the knowledge base,
- never sends tools: the model can only return JSON matching a schema.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

from essence.llm.cost import cost_usd


class LLMError(RuntimeError):
    pass


class LLMRefusal(LLMError):
    pass


class BudgetExceeded(LLMError):
    pass


@dataclass
class LLMResult:
    data: dict[str, Any]
    model: str
    input_tokens: int
    output_tokens: int
    batch: bool = False


class LLMProvider(Protocol):
    name: str
    is_local: bool

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        model: str,
        effort: str | None,
        max_tokens: int,
    ) -> LLMResult: ...


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def tokens_today(conn: sqlite3.Connection) -> int:
    day = datetime.now(timezone.utc).date().isoformat()
    row = conn.execute(
        "SELECT COALESCE(SUM(input_tokens + output_tokens), 0) FROM llm_calls WHERE substr(at, 1, 10) = ?",
        (day,),
    ).fetchone()
    return int(row[0])


class LLMClient:
    def __init__(self, provider: LLMProvider, conn: sqlite3.Connection, daily_token_budget: int = 0):
        self.provider = provider
        self.conn = conn
        self.daily_token_budget = daily_token_budget

    @property
    def is_local(self) -> bool:
        return self.provider.is_local

    def call(
        self,
        *,
        purpose: str,
        system: str,
        user: str,
        schema: dict[str, Any],
        model: str,
        effort: str | None = None,
        max_tokens: int = 16000,
        source_ids: list[str] | None = None,
        redactions: int = 0,
    ) -> dict[str, Any]:
        if self.daily_token_budget and tokens_today(self.conn) >= self.daily_token_budget:
            raise BudgetExceeded(f"daily token budget of {self.daily_token_budget:,} reached")
        result = self.provider.complete_json(
            system=system, user=user, schema=schema, model=model, effort=effort, max_tokens=max_tokens
        )
        with self.conn:
            self.conn.execute(
                "INSERT INTO llm_calls (at, provider, model, purpose, input_tokens, output_tokens, cost_usd,"
                " source_ids, redactions) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    _now(),
                    self.provider.name,
                    result.model,
                    purpose,
                    result.input_tokens,
                    result.output_tokens,
                    0.0 if self.provider.is_local else cost_usd(result.model, result.input_tokens, result.output_tokens, result.batch),
                    json.dumps(source_ids or []),
                    redactions,
                ),
            )
        return result.data


def make_provider(config) -> LLMProvider | None:
    if config.llm_provider == "anthropic":
        from essence.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider()
    if config.llm_provider == "ollama":
        from essence.llm.ollama_provider import OllamaProvider

        return OllamaProvider(config.ollama_url, config.ollama_model)
    if config.llm_provider in ("none", ""):
        return None
    raise LLMError(f"unknown llm provider {config.llm_provider!r}")
