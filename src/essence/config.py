"""Per-device configuration (.essence/config.toml) and environment (.env).

Secrets (API keys, the index passphrase, the MCP token) come from the
environment or a .env file, never from the vault.
"""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CONFIG = """\
# Essence Engine configuration (per device; not synced).

[llm]
# anthropic | ollama | none
provider = "anthropic"
# Leave empty to use ESSENCE_ANTHROPIC_MODEL from the environment / .env.
extract_model = ""
synth_model = ""
# low | medium | high (empty = model default)
extract_effort = "medium"
synth_effort = "high"
# Stop distilling for the day after this many tokens (input + output).
daily_token_budget = 2000000
ollama_url = "http://localhost:11434"
ollama_model = "qwen3:8b"

[embed]
# fastembed | none
provider = "fastembed"
model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

[privacy]
redact = true
# Sources whose origin (URL, file name, conversation title) contains any of these
# are marked never_cloud and only go to a local model.
never_cloud_patterns = []

[distill]
# Items shorter than this (characters) are skipped by triage.
min_chars = 150
# Most similar existing notes shown to the extractor / reconciler.
context_notes = 10

[index]
# true = encrypt index.db with SQLCipher; the passphrase comes from ESSENCE_DB_KEY.
encrypt = false

[mcp]
host = "127.0.0.1"
port = 8765
"""

_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_.\-]*)\s*=\s*(.*?)\s*$")


def load_dotenv(*paths: Path) -> dict[str, str]:
    """Load KEY=VALUE lines into os.environ (existing variables win). Returns what was set."""
    loaded: dict[str, str] = {}
    for path in paths:
        if not path or not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            m = _ENV_LINE.match(line)
            if not m:
                continue
            key, value = m.groups()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            if key not in os.environ:
                os.environ[key] = value
                loaded[key] = value
    return loaded


def env_model() -> str:
    # The hyphenated spelling is accepted for an existing .env.
    for key in ("ESSENCE_ANTHROPIC_MODEL", "KG-ESSENCE_ANTHROPIC_MODEL", "ANTHROPIC_MODEL"):
        if os.environ.get(key):
            return os.environ[key].strip()
    return "claude-opus-5-5"


@dataclass
class Config:
    llm_provider: str = "anthropic"
    extract_model: str = ""
    synth_model: str = ""
    extract_effort: str = "medium"
    synth_effort: str = "high"
    daily_token_budget: int = 2_000_000
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    embed_provider: str = "fastembed"
    embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    redact: bool = True
    never_cloud_patterns: list[str] = field(default_factory=list)
    min_chars: int = 150
    context_notes: int = 10
    encrypt: bool = False
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8765

    @property
    def model_extract(self) -> str:
        return self.extract_model or env_model()

    @property
    def model_synth(self) -> str:
        return self.synth_model or env_model()

    def is_never_cloud(self, *texts: str | None) -> bool:
        hay = " ".join(t for t in texts if t).lower()
        return any(p.lower() in hay for p in self.never_cloud_patterns if p)


def load_config(path: Path) -> Config:
    config = _load_file(path)
    # Environment overrides (handy for CI, tests and one-off runs).
    if os.environ.get("ESSENCE_LLM_PROVIDER"):
        config.llm_provider = os.environ["ESSENCE_LLM_PROVIDER"]
    if os.environ.get("ESSENCE_EMBED_PROVIDER"):
        config.embed_provider = os.environ["ESSENCE_EMBED_PROVIDER"]
    return config


def _load_file(path: Path) -> Config:
    if not path.is_file():
        return Config()
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    llm, emb = data.get("llm", {}), data.get("embed", {})
    priv, dist = data.get("privacy", {}), data.get("distill", {})
    idx, mcp = data.get("index", {}), data.get("mcp", {})
    d = Config()
    return Config(
        llm_provider=llm.get("provider", d.llm_provider),
        extract_model=llm.get("extract_model", d.extract_model),
        synth_model=llm.get("synth_model", d.synth_model),
        extract_effort=llm.get("extract_effort", d.extract_effort),
        synth_effort=llm.get("synth_effort", d.synth_effort),
        daily_token_budget=int(llm.get("daily_token_budget", d.daily_token_budget)),
        ollama_url=llm.get("ollama_url", d.ollama_url),
        ollama_model=llm.get("ollama_model", d.ollama_model),
        embed_provider=emb.get("provider", d.embed_provider),
        embed_model=emb.get("model", d.embed_model),
        redact=bool(priv.get("redact", d.redact)),
        never_cloud_patterns=list(priv.get("never_cloud_patterns", [])),
        min_chars=int(dist.get("min_chars", d.min_chars)),
        context_notes=int(dist.get("context_notes", d.context_notes)),
        encrypt=bool(idx.get("encrypt", d.encrypt)),
        mcp_host=mcp.get("host", d.mcp_host),
        mcp_port=int(mcp.get("port", d.mcp_port)),
    )
