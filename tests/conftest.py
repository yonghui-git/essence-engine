from pathlib import Path

import pytest

from essence.vault.layout import Vault, init_vault


@pytest.fixture
def vault(tmp_path: Path) -> Vault:
    v, _ = init_vault(tmp_path / "vault")
    return v
