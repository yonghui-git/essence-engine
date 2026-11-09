# Essence Engine

A local-first engine that turns daily inputs (AI chat sessions, web pages, documents) into a long-term, user-owned knowledge base. The AI suggests typed, linked notes; you approve them. Plain Markdown is the source of truth; a SQLite index can always be rebuilt from it.

See [docs/implementation-plan.md](docs/implementation-plan.md) for the design and milestones.

## Status

Week 1: vault layout, note model, rebuildable index, full-text search.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"   # on macOS/Linux: .venv/bin/python
```

(`uv sync --extra dev` works too.)

## Usage

```bash
essence init ~/knowledge                      # create a vault
essence --vault ~/knowledge new concept "Evergreen notes" --body "Organize by concept, not by source."
essence --vault ~/knowledge reindex           # rebuild .essence/index.db from the vault
essence --vault ~/knowledge search "concept"
essence --vault ~/knowledge stats
```

`--vault` can also be set with the `ESSENCE_VAULT` environment variable; it defaults to the current directory.

## Vault layout

```
raw/          immutable captured sources
notes/        approved knowledge: concepts/ claims/ insights/ decisions/ questions/
notes/_history/<id>/v<N>.md   earlier versions of a note
inbox/        AI suggestions awaiting review
digests/      weekly digests
schema.md     note types and extraction rules (editable)
.essence/     index.db and config (derived/operational, not synced)
```
