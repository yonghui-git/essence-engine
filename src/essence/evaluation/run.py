"""Distillation-quality evaluation on a small golden set.

Each item is distilled into a fresh, empty vault with the configured model.
Scores:
- recall: share of expected essences found (type matches when given, and most
  keyword groups appear in a candidate's title/body; "a|b" = either word)
- precision: share of candidates that match some expected essence
- skip accuracy: items expected to yield nothing yield nothing
- violations: a candidate contains a forbidden string (e.g. injected text)
"""

from __future__ import annotations

import json
import math
import tempfile
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from essence.capture.base import CapturedItem, SourceStore
from essence.config import Config
from essence.distill.inbox import pending
from essence.distill.pipeline import distill_source
from essence.index.db import open_index
from essence.index.reindex import reindex
from essence.llm.base import LLMClient, LLMError
from essence.vault.layout import init_vault


@dataclass
class ItemScore:
    id: str
    expected: int
    found: int
    candidates: int
    matching_candidates: int
    violations: list[str] = field(default_factory=list)
    error: str = ""
    rejected_again: int = 0  # candidates matching something you rejected before (items from reviews)
    titles: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    items: list[ItemScore]
    cost_usd: float

    @property
    def recall(self) -> float:
        exp = sum(i.expected for i in self.items)
        return sum(i.found for i in self.items) / exp if exp else 1.0

    @property
    def precision(self) -> float:
        c = sum(i.candidates for i in self.items if i.expected)
        return sum(i.matching_candidates for i in self.items if i.expected) / c if c else 0.0

    @property
    def skip_accuracy(self) -> float:
        skips = [i for i in self.items if i.expected == 0]
        return sum(1 for i in skips if i.candidates == 0) / len(skips) if skips else 1.0

    @property
    def rejected_again(self) -> int:
        return sum(i.rejected_again for i in self.items)

    @property
    def violations(self) -> int:
        return sum(len(i.violations) for i in self.items)


def load_golden(path: Path | None = None) -> list[dict]:
    if path is not None:
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(resources.files("essence.evaluation").joinpath("golden.json").read_text(encoding="utf-8"))


def _matches(spec: dict, cand_type: str, text: str) -> bool:
    if spec.get("type") and spec["type"] != cand_type:
        return False
    groups = spec.get("keywords") or []
    hit = sum(1 for g in groups if any(alt.strip().lower() in text for alt in g.split("|")))
    return hit >= math.ceil(0.6 * len(groups))


# Two notes "say the same thing" above this cosine similarity (multilingual MiniLM): reworded
# versions of one idea score about 0.75-0.9, related but distinct notes about 0.5-0.67.
SEMANTIC_MATCH = 0.72


def semantic_matcher(embedder, item: dict, candidates: list[tuple[str, str]]):
    """A match(spec, candidate_index) function by meaning, for specs that carry "text"."""
    import numpy as np

    specs = [s for s in (item.get("expected") or []) + (item.get("avoid") or []) if s.get("text")]
    if embedder is None or not specs or not candidates:
        return None

    def unit(vs):
        m = np.vstack([np.asarray(v, dtype=np.float32) for v in vs])
        return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)

    spec_m = unit(embedder.embed([s["text"] for s in specs]))
    cand_m = unit(embedder.embed([x for _, x in candidates]))
    sims = spec_m @ cand_m.T
    index = {id(s): i for i, s in enumerate(specs)}

    def match(spec: dict, j: int):
        i = index.get(id(spec))
        return None if i is None else bool(sims[i, j] >= SEMANTIC_MATCH)

    return match


def score_item(item: dict, candidates: list[tuple[str, str]], matcher=None) -> ItemScore:
    """candidates: (type, lowercase title + body). matcher(spec, j) -> bool, or None to use keywords."""

    def hit(spec, j):
        t, x = candidates[j]
        by_meaning = matcher(spec, j) if matcher else None
        return _matches(spec, t, x) if by_meaning is None else by_meaning

    expected = item.get("expected") or []
    idx = range(len(candidates))
    found = sum(1 for spec in expected if any(hit(spec, j) for j in idx))
    matching = sum(1 for j in idx if any(hit(spec, j) for spec in expected))
    violations = [m for m in item.get("must_not") or [] for _, x in candidates if m.lower() in x]
    avoid = item.get("avoid") or []
    again = sum(1 for j in idx if any(hit(spec, j) for spec in avoid))
    return ItemScore(item["id"], len(expected), found, len(candidates), matching, violations, rejected_again=again)


def run_eval(provider, config: Config, items: list[dict] | None = None, on_item=None, embedder=None) -> EvalReport:
    items = items or load_golden()
    scores: list[ItemScore] = []
    total_cost = 0.0
    with tempfile.TemporaryDirectory(prefix="essence-eval-") as tmp:
        for item in items:
            vault, _ = init_vault(Path(tmp) / item["id"])
            store = SourceStore(vault, config)
            _, src = store.store(CapturedItem(item["kind"], f"eval:{item['id']}", item["title"], item["body"]))
            with open_index(vault.index_path) as conn:
                reindex(vault, conn)
                row = conn.execute("SELECT * FROM sources WHERE id = ?", (src.id,)).fetchone()
                llm = LLMClient(provider, conn)
                error = ""
                try:
                    distill_source(vault, conn, llm, config, row)
                except LLMError as exc:
                    error = str(exc)
                total_cost += conn.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM llm_calls").fetchone()[0]
            sugs, _ = pending(vault)
            cands = [(s.note.type.value, f"{s.note.title}\n{s.note.body}".lower()) for s in sugs]
            sc = score_item(item, cands, semantic_matcher(embedder, item, cands))
            sc.error = error
            sc.titles = [s.note.title for s in sugs]
            scores.append(sc)
            if on_item:
                on_item(sc)
    return EvalReport(scores, total_cost)
