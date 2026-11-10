from pathlib import Path

import pytest

from essence.vault.layout import Vault, init_vault


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """No network in unit tests: no embedding model download, no live LLM."""
    monkeypatch.setenv("ESSENCE_EMBED_PROVIDER", "none")
    monkeypatch.setenv("ESSENCE_LLM_PROVIDER", "none")
    monkeypatch.delenv("ESSENCE_DB_KEY", raising=False)


@pytest.fixture
def vault(tmp_path: Path) -> Vault:
    v, _ = init_vault(tmp_path / "vault")
    return v
