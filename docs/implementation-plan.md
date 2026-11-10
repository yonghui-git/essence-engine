# Essence Engine — implementation plan (tier A MVP)

Status: draft, 2026-11-10. Built from two artifacts made during the research:

- Research report: [Personal knowledge accumulation systems: research report](https://claude.ai/artifact/2LeSktLjPxocAnyhndw9om) (doc, rev 14)
- Deck: [Personal knowledge accumulation: ideas, architecture and opportunity](https://claude.ai/artifact/TsrKsZbLcD8NEREyz6oxYL) (21 slides)

## 0. Implementation status (2026-11-10)

Weeks 1–6 are implemented and covered by offline tests (`pytest`: fake LLM, fake embedder).

| Week | Built | Where |
| --- | --- | --- |
| 1 | Vault layout, typed notes with lossless round-trip, rebuildable index, full-text search, CLI | `vault/`, `index/db.py`, `index/reindex.py` |
| 2 | ChatGPT / Claude export importers (zip, folder or json; active branch only), web-clip and document importers, idempotent re-import, excerpt segmentation, redaction, `never_cloud` | `capture/`, `segment.py`, `privacy/redact.py` |
| 3 | Provider-neutral LLM client with outbound-call log, cost table and daily token budget; Anthropic (structured outputs, refusal fallback) and Ollama providers; triage → extract → reconcile → inbox, with every model output validated | `llm/`, `distill/` |
| 4 | Review inbox (CLI and editable files), accept / edit / reject with edit distance, versions in `notes/_history`, supersede / update / confirm / link / merge; hybrid search (FTS + vectors + 1-hop links, RRF) | `distill/inbox.py`, `distill/apply.py`, `index/search.py`, `index/vectors.py` |
| 5 | MCP server (stdio; HTTP on localhost with a bearer token), weekly digest, "changed my mind" log, near-duplicate merge proposals | `mcp_server/`, `consolidate/` |
| 6 | Propagating delete (orphans go to the inbox), SQLCipher index encryption and migration, metrics, cost report, golden-set eval including a prompt-injection case, `doctor` | `ops.py`, `evaluation/` |

Deviations from the plan, and why:

- **Vectors:** exact numpy cosine search over cached float32 vectors instead of sqlite-vec. That is milliseconds at personal scale and needs no SQLite extension, which also keeps SQLCipher simple. `index/vectors.py` is the swap point.
- **Batch API:** not used yet. Digest and consolidation calls are few, so real-time calls cost little. Revisit if `essence cost` shows nightly jobs matter.
- **Triage:** heuristic only (length with CJK weighting, code share, small talk), with no LLM classifier. Measure with `essence cost` before adding one.
- **Docling:** optional (`[docs]` extra, heavy). Built-in parsers (pypdf, python-docx, plain HTML) are used when it is absent.
- **Eval set:** 6 hand-written items, not 50. Grow it from pilot data, as planned.
- **Vault encryption (§8 decision 2):** only the index is encrypted. Notes stay plain Markdown so Obsidian keeps working; rely on disk encryption for the vault.

Not yet verified live: calls to the Claude API. The first live run reached the API, but every request was rejected because the API key is not scoped to a workspace. Setting `ANTHROPIC_WORKSPACE_ID` in `.env` fixes it. Then run `essence eval` and a first `essence distill` on real exports.

## 1. What we are building

A local-first engine that reads a person's daily inputs (AI chat exports, web clips, documents). For each one it asks **"what is new here compared with what I already know?"** and suggests typed, linked notes. The person accepts, rewords or rejects them in a review inbox. The knowledge base is served to every AI assistant over MCP.

Design rules from the research, which the code must enforce:

1. **The AI keeps the books; the user keeps the judgment.** Nothing becomes canonical without passing through the review inbox. Bookkeeping (links, dedup, contradictions, provenance) is automatic.
2. **Markdown is the source of truth.** The vault stays readable and Obsidian-compatible without the app. The SQLite index can always be rebuilt from it.
3. **Grow by updating, not by piling up.** Extraction compares each input with existing notes (confirms / refines / contradicts / extends). It does not summarize documents.
4. **Success means reuse.** From day one, log reuse (search hits, MCP reads, digest opens) and revision, not just how many notes were captured.
5. **Privacy is part of v1**, not something added later.

### MVP scope (tier A, 4–6 weeks, one developer)

| Area | In | Deferred |
| --- | --- | --- |
| Capture | ChatGPT + Claude export zips; Obsidian Web Clipper folder; PDF/DOCX via Docling; MCP `save` tool | Gemini Takeout, email, screen capture, browser extension |
| Distillation | Triage → typed extraction with excerpt provenance → reconcile → review inbox | Temporal graph (Graphiti) |
| Index | SQLite FTS5 + sqlite-vec + nodes/edges tables, hybrid search (RRF) | Graph ranking, global summaries |
| Use | Local MCP server (search, read, save); CLI; weekly "what I learned and decided" digest; "changed my mind" log | Spaced resurfacing (FSRS), chat UI polish, CRDT sync, teams |
| Trust | Encryption at rest, redaction, outbound-call log, propagating deletion, injection-safe extraction | Hosted OAuth MCP, per-tenant encryption |

The report recommends three first-version features: the weekly learning report, the "changed my mind" log and decision support. In the MVP these are the digest, the versioned-insight history and `Decision` notes with expected outcomes plus a review date.

## 2. Key technical decisions (proposed)

| Decision | Choice | Reason |
| --- | --- | --- |
| Language | Python 3.12, `uv` for env/deps | Docling, sqlite-vec and the MCP SDK are Python-first |
| CLI | `typer` | Simple and scriptable |
| Storage | Markdown vault + `.essence/index.db` (SQLite) | Matches the report; portable and rebuildable |
| Vectors | `sqlite-vec` behind a `VectorStore` interface | Pre-v1, so keep LanceDB as a drop-in fallback |
| Embeddings | Local (`fastembed`, small multilingual model) by default | Avoids a cloud call for every chunk; cheap |
| LLM | `LLMProvider` interface. Default Claude API (`claude-haiku-5-5` for triage/extraction, `claude-sonnet-5-5` for weekly synthesis). Bring-your-own-key. Optional local model via Ollama | Cost tiering from the report. Prices need confirming (open item) |
| Structured output | Pydantic schemas + tool/JSON-schema output, with no tools exposed to the model | Typed notes; blocks prompt injection |
| MCP | Official Python MCP SDK, localhost only, bearer token | Neutral access for every assistant |
| Encryption | SQLCipher for the index; vault left to the OS/disk encryption in v1, with optional age-encrypted `raw/` | Keeps Obsidian compatibility for notes. **Open decision** (see §8) |
| Sync | None built-in; document Git/Syncthing; index rebuilt per device | Single user needs no CRDTs |

## 3. Data model

### Vault layout

```
vault/
  raw/                      # immutable captured sources (never edited)
    chats/<provider>/<conversation-id>.md
    web/<slug>.md
    docs/<slug>.md          # Docling output; original file kept alongside
  notes/                    # canonical, user-approved knowledge
    concepts/  claims/  insights/  decisions/  questions/
    _history/<id>/v<N>.md   # earlier versions of a note; the "changed my mind" record lives in the vault
  inbox/                    # suggested changes awaiting review (one file per suggestion)
  digests/2026-W46.md
  schema.md                 # the "schema file" (Karpathy): note types + extraction rules, user-editable
  .essence/                 # index.db, config, outbound-call log, eval data (git-ignored)
```

### Note types (YAML frontmatter + Markdown body)

`Source`, `Excerpt` (stored in the index and pointed to from notes), `Concept`, `Claim`, `Insight`, `Decision`, `Question`.

Common frontmatter:

```yaml
id: ins_01JC...            # ULID
type: insight
title: Review inbox is core, not optional
status: active             # suggested | active | superseded | invalid
applies_when: designing AI-assisted note tools
not_when: ...
sources: [src_01J..#exc_3, src_01J..#exc_7]   # excerpt-level provenance
links: [con_01J.., dec_01J..]
valid_from: 2026-11-09
valid_to: null
version: 3                 # insights are versioned; earlier versions in notes/_history/
created_by: ai | user | ai+edited
```

`Decision` adds `expected_outcome`, `review_on`, `actual_outcome`, `lesson`. `Claim` adds `confidence`. A note that is contradicted is marked `superseded` or `invalid`, never deleted. This history drives the "changed my mind" log.

### SQLite index

Derived tables are dropped and rebuilt from the vault by `essence reindex`. Operational tables (`suggestions`, `events`, `llm_calls`) hold state that is not in the vault, and reindex never touches them.

- `sources`, `excerpts` (text, offsets, source id)
- `notes` (id, type, status, title, path, hash, version, timestamps), `note_versions`
- `edges` (from, to, kind: `links|supports|contradicts|refines|derived_from`)
- `notes_fts` / `excerpts_fts` (FTS5), `vec_notes` / `vec_excerpts` (sqlite-vec)
- `suggestions` (inbox items: op, payload, status, reviewer action, edit distance)
- `events` (reuse telemetry: search, mcp_read, digest_open, accept/edit/reject). Local only.
- `llm_calls` (outbound log: model, tokens, cost, redactions applied, source ids)

## 4. Pipeline

```
capture → normalize → redact → triage → extract → reconcile → inbox → (user) → apply → index
                                                                  ↘ nightly: consolidate → digest
```

1. **Capture / normalize.** Each importer writes an immutable `raw/` Markdown file plus a `Source` record. Chat importers must tolerate schema changes: there is no documented format, so fixture-test them against real exports.
2. **Redact.** Regex + secret scanners (keys, emails, phone numbers, IBAN). Per-source `never_cloud` flag sends the item to the local model or skips it.
3. **Triage** (cheap model or heuristics). Scores each item or chat segment: skip / light / full distillation. This protects the margin and is the main cost lever.
4. **Extract.** Retrieve the top-k similar active notes. The LLM returns typed candidates, each tagged `new|confirms|refines|contradicts|extends`, with excerpt ids. The model gets no tools, and the raw content is wrapped as data.
5. **Reconcile** (Mem0 pattern). For each candidate against its 10 nearest notes, decide `add | update | link | invalidate | noop`. The result becomes `suggestions` and `inbox/*.md`.
6. **Review inbox.** CLI (`essence review`) plus editable inbox files. Accept, edit or reject. Edits are recorded (the signal for distillation quality). Accepting writes or updates `notes/` and bumps the version.
7. **Consolidate** (nightly, batch API). Clusters, merge proposals, contradiction proposals, decisions due for review. Builds the weekly digest: what I learned, what changed my mind, decisions made or due, open questions.

Deletion propagates: deleting a source marks every derived suggestion or note. A note left without sources goes back to the inbox for a keep/delete decision.

## 5. Package layout

```
src/essence/
  cli.py                 # typer app: init, import, clip, review, search, digest, serve, reindex, delete
  config.py
  vault/      model.py (pydantic note types), io.py (frontmatter read/write), schema.md template
  capture/    chatgpt.py, claude.py, webclip.py, docling_docs.py, base.py
  privacy/    redact.py, crypto.py, outbound_log.py
  llm/        provider.py, anthropic.py, ollama.py, prompts/, cost.py
  distill/    triage.py, extract.py, reconcile.py, inbox.py, apply.py
  index/      db.py (migrations), fts.py, vectors.py, graph.py, search.py (hybrid + RRF + 1–2 hop)
  consolidate/ cluster.py, digest.py
  mcp_server/ server.py   # tools: search, read_note, save, list_inbox; resources: notes
  telemetry/  events.py, metrics.py
tests/  fixtures/ (anonymized real exports), golden/ (eval set)
```

## 6. Milestones

| Week | Milestone | Done when |
| --- | --- | --- |
| 0 | Prerequisites | Real ChatGPT and Claude exports downloaded and anonymized into fixtures; API prices confirmed; open decisions in §8 settled |
| 1 | Skeleton + vault + index | `essence init`, note model round-trips Markdown losslessly, `reindex` rebuilds `index.db` from the vault, FTS search works |
| 2 | Capture | ChatGPT/Claude zip importers, web-clip folder watcher and Docling importer write `raw/` + `Source`; redaction + outbound log in place |
| 3 | Distillation core | Triage + extract + reconcile produce inbox suggestions with excerpt provenance on fixture data; cost per item logged |
| 4 | Review + apply + search | `essence review` accept/edit/reject; versions and supersession work; hybrid search (FTS + vectors + link hop, RRF) |
| 5 | MCP + digest | Localhost MCP server with token auth works from Claude Desktop/Code; weekly digest and "changed my mind" log generated |
| 6 | Hardening + dogfood | SQLCipher on, propagating delete, prompt-injection tests pass; author uses it daily; first eval set (~50 labelled items) and cost-per-active-user report |

Weeks 6–12 follow the report's plan: a pilot with 10–20 design partners, tracking weekly return, digest opens and the accept/edit/reject share, then a go/no-go at week 12.

## 7. Quality, cost and testing

- **Unit/fixture tests** for importers (schema drift), frontmatter round-trip, reindex determinism and RRF ranking.
- **Injection tests:** captured pages and chats containing instructions must not change the model's output schema or trigger actions.
- **Distillation eval set** (`tests/golden/`): input → expected candidate types/relations. Score precision of accepted-as-is, edit distance and false-contradiction rate. Use it to compare models and prompts. No public benchmark exists, so this is our own.
- **Cost guardrails:** per-day token budget, batch API for nightly jobs, `essence cost` report. Target: heavy user (~30 items/day) well under €10/month in inference.
- **Pilot metrics** (local `events`, opt-in export): weekly return, inbox throughput, accept/edit/reject %, reuse events per week.

## 8. Open decisions and items to verify before coding

Decisions for you:

1. **Language/stack.** Python as proposed, or something else (e.g. TypeScript to sit closer to Obsidian plugins)?
2. **Vault encryption.** Plain Markdown on an encrypted disk (keeps Obsidian working), or encrypted vault files with the app as the only reader?
3. **Default LLM.** Claude API by default with BYOK, or local-first (Ollama) by default with the cloud opt-in?
4. **Review UI.** CLI + inbox files only for the MVP, or a minimal local web UI from the start?
5. **Open source.** Is the capture tool + MCP server open source from day one (the report's go-to-market), and under which licence?

To verify (from the report's open items):

- ChatGPT and Claude export schemas: get current real exports.
- Claude Haiku 5.5 / Sonnet 5.5 and current OpenAI/Gemini API prices (the report flags the Haiku price as unusually low).
- Whether sqlite-vec is still pre-v1 and works with SQLCipher on Windows. If not, use the LanceDB fallback.
- MCP spec 2026-07-28 features used (elicitation for "is this a duplicate?" in chat) are supported by the Python SDK and the target clients.
