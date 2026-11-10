# Essence Engine

A local-first engine that turns daily inputs (AI chat sessions, web pages, documents) into a long-term knowledge base you own. For each input it asks *what is new here compared with what I already know?* and suggests typed, linked notes. You accept, reword or reject them in a review inbox, and every AI assistant can read the result over MCP.

- **Markdown is the source of truth.** The vault opens in Obsidian. The SQLite index can always be rebuilt from it.
- **The AI keeps the books; you keep the judgment.** Nothing becomes a note without your review.
- **Every note is traceable** to the source excerpts it came from. Contradicted notes are superseded, never deleted, which gives you a "changed my mind" log.
- **Privacy first.** Secrets and personal data are redacted before any LLM call, `never_cloud` sources only go to a local model, every outbound call is logged, deletion propagates, and the index can be encrypted.

Design and milestones: [docs/implementation-plan.md](docs/implementation-plan.md).

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[all]"      # macOS/Linux: .venv/bin/python
cp .env.example .env                                 # then fill in ANTHROPIC_API_KEY
```

Extras: `vectors` (local embeddings via fastembed), `encrypt` (SQLCipher), `docs` (Docling for PDF/DOCX/PPTX; built-in parsers are used without it), `all` = vectors + encrypt.

`.env` is read from the current directory and from the vault root. If your API key is not scoped to a workspace, set `ANTHROPIC_WORKSPACE_ID` too.

## Daily use

```bash
essence init ~/knowledge
export ESSENCE_VAULT=~/knowledge          # or pass --vault each time

essence import chat ~/Downloads/chatgpt-export.zip   # ChatGPT or Claude export (auto-detected)
essence import web  ~/ObsidianVault/Clippings        # Obsidian Web Clipper folder
essence import docs ~/Documents/papers               # PDF, DOCX, Markdown, text, HTML

essence distill --estimate   # what the next run would cost (no LLM calls)
essence distill        # new or changed sources -> suggestions in inbox/
essence review         # accept / edit / reject one by one (or edit inbox/*.md, then `essence accept --all`)

essence search "how do I avoid forgetting what I learn"
essence show ins_01J...
essence digest         # weekly "what I learned and decided", saved in digests/
essence changes        # the "changed my mind" log
essence consolidate    # propose merges of near-duplicate notes
```

## Use it from any AI assistant (MCP)

```bash
essence serve                      # stdio, for Claude Desktop / Claude Code
essence serve --http               # http://127.0.0.1:8765/mcp, bearer token: essence serve --show-token
```

Claude Code: `claude mcp add essence -- essence --vault ~/knowledge serve`.
Tools: `search_knowledge`, `read_note_by_id`, `save_to_knowledge_base` (the saved text goes through the review inbox before it becomes a note), `list_inbox`.

## Trust and measurement

```bash
essence delete src_01J... # remove a source from everything derived from it
essence encrypt-index     # SQLCipher-encrypt the index (passphrase: ESSENCE_DB_KEY)
essence doctor            # check credentials, encryption, embeddings, MCP
essence metrics           # accept / edit / reject rates, active days, reuse
essence cost              # spend from the outbound-call log, monthly projection
essence eval              # distillation quality on a golden set (makes real LLM calls)
```

Settings live in `.essence/config.toml`: models, effort, daily token budget, `never_cloud_patterns`, embeddings, encryption and the MCP port. Set `[llm] provider = "ollama"` to keep everything on your machine.

## Vault layout

```
raw/                 captured sources (chats/, web/, docs/; originals in raw/_originals/)
notes/               approved knowledge: concepts/ claims/ insights/ decisions/ questions/
notes/_history/      earlier versions of each note
inbox/               suggestions waiting for review (one editable file each)
digests/             weekly digests
schema.md            what counts as essence, read by the extractor; edit it
.essence/            index.db, config.toml, mcp_token (per device, not synced)
```

## Development

```bash
.venv/Scripts/python -m pytest
```

Tests run offline. The LLM and the embedder are replaced with deterministic fakes.
