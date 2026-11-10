"""Redact secrets and personal data before any text leaves the machine.

Placeholders are stable within one call (the same email becomes the same
``[EMAIL_1]`` everywhere), so the model can still reason about who said what.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# Order matters: secrets first (they may contain digits that look like phones).
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("PRIVATE_KEY", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S)),
    ("API_KEY", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}")),
    ("API_KEY", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}")),
    ("API_KEY", re.compile(r"\b(?:ghp|gho|ghu|ghs|github_pat)_[A-Za-z0-9_]{20,}")),
    ("API_KEY", re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}")),
    ("API_KEY", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("API_KEY", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}")),
    ("SECRET", re.compile(r"(?i)\b(password|passwd|secret|token|api[_-]?key)\b(\s*[:=]\s*)(\S{6,})")),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("PHONE", re.compile(r"(?<![\w+])\+?\d{1,3}[ .\-]?\(?\d{1,4}\)?(?:[ .\-]?\d{2,4}){2,4}(?!\w)")),
]


@dataclass
class Redaction:
    text: str
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def redact(text: str) -> Redaction:
    mapping: dict[tuple[str, str], str] = {}
    counts: dict[str, int] = {}

    def placeholder(kind: str, value: str) -> str:
        key = (kind, value)
        if key not in mapping:
            n = sum(1 for k in mapping if k[0] == kind) + 1
            mapping[key] = f"[{kind}_{n}]"
        counts[kind] = counts.get(kind, 0) + 1
        return mapping[key]

    for kind, pattern in _PATTERNS:
        def sub(m: re.Match[str], kind: str = kind) -> str:
            value = m.group(0)
            if kind == "SECRET":
                return f"{m.group(1)}{m.group(2)}{placeholder(kind, m.group(3))}"
            if kind == "CARD":
                digits = re.sub(r"\D", "", value)
                if not (13 <= len(digits) <= 19 and _luhn_ok(digits)):
                    return value
            if kind == "PHONE":
                digits = re.sub(r"\D", "", value)
                if len(digits) < 9 or re.fullmatch(r"(19|20)\d{2}[01]\d[0-3]\d", digits[:8]):
                    return value  # too short, or looks like a date
            return placeholder(kind, value)

        text = pattern.sub(sub, text)
    return Redaction(text, counts)
