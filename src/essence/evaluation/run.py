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


def score_item(item: dict, candidates: list[tuple[str, str]]) -> ItemScore:
    """candidates: (type, lowercase title + body)."""
    expected = item.get("expected") or []
    found = sum(1 for spec in expected if any(_matches(spec, t, x) for t, x in candidates))
    matching = sum(1 for t, x in candidates if any(_matches(spec, t, x) for spec in expected))
    violations = [m for m in item.get("must_not") or [] for _, x in candidates if m.lower() in x]
    return ItemScore(item["id"], len(expected), found, len(candidates), matching, violations)


def run_eval(provider, config: Config, items: list[dict] | None = None, on_item=None) -> EvalReport:
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
            sc = score_item(item, cands)
            sc.error = error
            sc.titles = [s.note.title for s in sugs]
            scores.append(sc)
            if on_item:
                on_item(sc)
    return EvalReport(scores, total_cost)
