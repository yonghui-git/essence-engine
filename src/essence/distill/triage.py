"""Cheap, local triage: decide how much distillation an item deserves before any LLM call.

This is the main cost lever. Levels:
- skip:  too short, small talk, or nothing but code/tool output
- light: worth a look, but ask for at most 2 candidates
- full:  substantive; ask for up to 6
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_CODE_FENCE = re.compile(r"```.*?```", re.S)
_ROLE = re.compile(r"^## (User|Assistant)\b", re.M)
_SMALL_TALK = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|ok|okay|great|cool|test|你好|谢谢|好的)[\s!.?]*$", re.I
)


@dataclass
class Triage:
    level: str  # skip | light | full
    reason: str

    @property
    def max_candidates(self) -> int:
        return {"skip": 0, "light": 2, "full": 6}[self.level]


_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]")


def weighted_len(text: str) -> int:
    """Length in Latin-character equivalents: one CJK character carries about as much as three."""
    cjk = len(_CJK.findall(text))
    return len(text) + 2 * cjk


def triage(body: str, kind: str, min_chars: int = 150) -> Triage:
    text = body.strip()
    if weighted_len(text) < min_chars:
        return Triage("skip", f"shorter than {min_chars} characters")

    prose = _CODE_FENCE.sub("", text)
    if len(prose.strip()) < 0.15 * len(text) and weighted_len(prose.strip()) < min_chars:
        return Triage("skip", "almost entirely code or tool output")
    plen = weighted_len(prose)

    if kind == "chat":
        turns = _ROLE.findall(text)
        user_parts = re.split(r"^## User.*$", text, flags=re.M)[1:]
        user_text = " ".join(p.split("## Assistant")[0] for p in user_parts).strip()
        if turns and all(_SMALL_TALK.match(line) for line in user_text.splitlines() if line.strip()):
            return Triage("skip", "small talk only")
        if len(turns) <= 2 and plen < 3 * min_chars:
            return Triage("light", "short exchange")

    if plen < 3 * min_chars:
        return Triage("light", "short text")
    return Triage("full", "substantive")
