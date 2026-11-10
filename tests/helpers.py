"""Test doubles: a scripted LLM provider and a deterministic bag-of-words embedder."""

from __future__ import annotations

import hashlib
import re

import numpy as np

from essence.llm.base import LLMResult


class FakeProvider:
    """Answers by schema: 'candidates' (extract), 'decisions' (reconcile), else `other`."""

    name = "fake"

    def __init__(self, extract=None, reconcile=None, other=None, is_local=False):
        self.extract = extract or (lambda system, user: {"candidates": [], "skip_reason": "none"})
        self.reconcile = reconcile or (lambda system, user: {"decisions": []})
        self.other = other or (lambda system, user: {})
        self.is_local = is_local
        self.calls: list[dict] = []

    def complete_json(self, *, system, user, schema, model, effort, max_tokens):
        props = schema.get("properties", {})
        kind = "extract" if "candidates" in props else "reconcile" if "decisions" in props else "other"
        self.calls.append({"kind": kind, "system": system, "user": user, "model": model, "schema": schema})
        data = getattr(self, kind)(system, user)
        return LLMResult(data, model, 1000, 200)


def cand(title, body, *, type="insight", excerpt_ids=("exc_1",), relation="new", related="", **kw):
    return {
        "type": type, "title": title, "body": body, "applies_when": kw.get("applies_when", ""),
        "not_when": "", "relation": relation, "related_note_id": related,
        "excerpt_ids": list(excerpt_ids), "confidence": kw.get("confidence", 0),
        "expected_outcome": kw.get("expected_outcome", ""), "why_reusable": kw.get("why", ""),
    }


class HashEmbedder:
    model_name = "test-hash-64"

    def embed(self, texts):
        out = []
        for t in texts:
            v = np.zeros(64, dtype=np.float32)
            for w in re.findall(r"\w+", t.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1.0
            out.append(v)
        return out


LONG_CHAT = (
    "## User\n\nHow should my knowledge base handle notes that the AI writes? I worry I will stop remembering.\n\n"
    "## Assistant\n\nLet the AI do the bookkeeping: linking, deduplication, contradictions and provenance. "
    "You confirm or reword the insights that matter, through a review inbox. Small studies show that people "
    "who let an LLM write everything feel less ownership and recall less of their own work.\n\n"
    "## User\n\nThen the review inbox is a core feature. I will build it first and measure how often I edit.\n"
)
