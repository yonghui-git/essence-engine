from essence.capture.base import CapturedItem, SourceStore
from essence.config import Config
from essence.distill.apply import accept, reject
from essence.distill.inbox import pending
from essence.distill.pipeline import distill
from essence.evaluation.from_reviews import export_reviews, keywords_for
from essence.evaluation.run import score_item
from essence.index import open_index, reindex
from essence.llm.base import LLMClient
from essence.vault.io import write_note
from tests.helpers import LONG_CHAT, FakeProvider, cand


def test_keywords_for_titles():
    assert keywords_for("Build the review inbox first") == ["review", "build", "inbox"]
    assert keywords_for("Distilled notes compound") == ["distilled|distil", "compound|compou", "notes"]
    assert keywords_for("A forum's three-step method") == ["method", "forum", "three"]
    assert keywords_for("参会单位三层匹配法：直接客户、渠道伙伴") == ["参会|配法", "直接|客户", "渠道|伙伴"]
    assert keywords_for("a b") == []


def test_export_from_reviews_round_trips_into_scoring(vault):
    store = SourceStore(vault, Config(min_chars=100))
    store.store(CapturedItem("chat", "chat:1", "AI and my notes", LONG_CHAT))
    fake = FakeProvider(extract=lambda s, u: {"candidates": [
        cand("Review inbox protects ownership", "Suggestions go to an inbox.", excerpt_ids=["exc_1"]),
        cand("Chat about weather", "Small talk.", excerpt_ids=["exc_2"]),
    ], "skip_reason": ""})
    with open_index(vault.index_path) as conn:
        reindex(vault, conn)
        distill(vault, conn, LLMClient(fake, conn), Config(min_chars=100))
        keep, drop = sorted(pending(vault)[0], key=lambda s: s.note.title.startswith("Chat"))
        edited = keep.note.model_copy(update={"title": "Review inbox protects ownership and recall"})
        write_note(edited, keep.path)
        accept(vault, conn, next(s for s in pending(vault)[0] if s.id == keep.id))  # re-read after the edit
        reject(vault, conn, drop)
        reindex(vault, conn)
        (item,) = export_reviews(vault, conn)
    assert item["kind"] == "chat" and "## User" in item["body"]
    assert [e["title"] for e in item["expected"]] == ["Review inbox protects ownership and recall"]  # final, edited form
    assert [a["title"] for a in item["avoid"]] == ["Chat about weather"]
    good = score_item(item, [("insight", "review inbox protects ownership and recall")])
    bad = score_item(item, [("insight", "chat about the weather today")])
    assert good.found == 1 and good.rejected_again == 0
    assert bad.found == 0 and bad.rejected_again == 1


def test_semantic_matching_beats_keywords_for_rewording():
    from essence.evaluation.run import semantic_matcher
    from tests.helpers import HashEmbedder

    item = {"id": "t", "expected": [{"type": "insight", "title": "x", "text": "review inbox protects ownership",
                          "keywords": ["zzz", "yyy"]}], "avoid": []}
    cands = [("concept", "review inbox protects ownership")]
    assert score_item(item, cands).found == 0  # keywords miss
    assert score_item(item, cands, semantic_matcher(HashEmbedder(), item, cands)).found == 1
    assert semantic_matcher(None, item, cands) is None
