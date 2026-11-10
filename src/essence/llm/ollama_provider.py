"""Local model via Ollama's HTTP API (nothing leaves the machine)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from essence.llm.base import LLMError, LLMResult


class OllamaProvider:
    name = "ollama"
    is_local = True

    def __init__(self, url: str = "http://localhost:11434", model: str = "qwen3:8b"):
        self.url = url.rstrip("/")
        self.default_model = model

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
        # Claude model names mean nothing to Ollama; use the configured local model.
        local_model = model if not model.startswith("claude-") else self.default_model
        body = json.dumps(
            {
                "model": local_model,
                "stream": False,
                "format": schema,
                "options": {"num_predict": max_tokens},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            }
        ).encode()
        req = urllib.request.Request(f"{self.url}/api/chat", data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                payload = json.loads(resp.read().decode())
        except (urllib.error.URLError, TimeoutError) as exc:
            raise LLMError(f"cannot reach Ollama at {self.url}: {exc}") from exc
        try:
            data = json.loads(payload["message"]["content"])
        except (KeyError, json.JSONDecodeError) as exc:
            raise LLMError(f"Ollama returned no valid JSON: {exc}") from exc
        return LLMResult(data, local_model, payload.get("prompt_eval_count", 0), payload.get("eval_count", 0))
