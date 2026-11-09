"""Sortable, prefixed identifiers (ULID-style: 48-bit ms timestamp + 80 random bits)."""

from __future__ import annotations

import os
import time

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def ulid(timestamp_ms: int | None = None) -> str:
    """Return a 26-character Crockford base32 ULID."""
    ts = int(time.time() * 1000) if timestamp_ms is None else timestamp_ms
    value = (ts << 80) | int.from_bytes(os.urandom(10), "big")
    chars = []
    for _ in range(26):
        chars.append(_CROCKFORD[value & 0x1F])
        value >>= 5
    return "".join(reversed(chars))


def new_id(prefix: str) -> str:
    return f"{prefix}_{ulid()}"
