"""Claude API provider (official `anthropic` SDK), JSON via structured outputs."""

from __future__ import annotations

import json
import os
from typing import Any

from essence.llm.base import LLMError, LLMRefusal, LLMResult

# Models that accept the server-side refusal fallback ("default" routing).
_FALLBACK_MODELS = {"claude-fable-5-2", "claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicProvider:
    name = "anthropic"
    is_local = False

    def __init__(self, client=None):
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - dependency is required
                raise LLMError("the anthropic package is not installed") from exc
            # Keys not scoped to a workspace must name one on every request.
            workspace = os.environ.get("ANTHROPIC_WORKSPACE_ID") or os.environ.get("ESSENCE_ANTHROPIC_WORKSPACE_ID")
            headers = {"anthropic-workspace-id": workspace.strip()} if workspace else None
            client = anthropic.Anthropic(default_headers=headers)
        self.client = client

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        model: str,
        effort: str | None,
        max_tokens: int,
    ) -> LLMResult:
        import anthropic

        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": schema}}
        if effort:
            output_config["effort"] = effort
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            # The system prompt is stable across calls, so cache it.
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user}],
            "output_config": output_config,
        }
        try:
            if model in _FALLBACK_MODELS:
                with self.client.beta.messages.stream(betas=[_FALLBACK_BETA], fallbacks="default", **params) as s:
                    response = s.get_final_message()
            else:
                with self.client.messages.stream(**params) as s:
                    response = s.get_final_message()
        except anthropic.AuthenticationError as exc:
            raise LLMError("Anthropic rejected the API key (check ANTHROPIC_API_KEY)") from exc
        except anthropic.NotFoundError as exc:
            raise LLMError(f"model {model!r} not found") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("rate limited by the Anthropic API; try again later") from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code == 400 and "anthropic-workspace-id" in str(exc.message):
                raise LLMError(
                    "this API key is not scoped to a workspace: set ANTHROPIC_WORKSPACE_ID in .env "
                    "(the workspace id from the Claude Console), or use a workspace-scoped key"
                ) from exc
            raise LLMError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("cannot reach the Anthropic API") from exc

        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise LLMRefusal(f"model declined (category: {category})")
        if response.stop_reason == "max_tokens":
            raise LLMError("response hit max_tokens before the JSON was complete")
        text = next((b.text for b in response.content if getattr(b, "type", None) == "text"), None)
        if text is None:
            raise LLMError("no text block in the response")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError(f"response was not valid JSON: {exc}") from exc
        usage = response.usage
        input_tokens = (
            (usage.input_tokens or 0)
            + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
            + (getattr(usage, "cache_read_input_tokens", 0) or 0)
        )
        return LLMResult(data, getattr(response, "model", model) or model, input_tokens, usage.output_tokens or 0)
