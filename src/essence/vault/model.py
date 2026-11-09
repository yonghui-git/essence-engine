"""Typed notes: the units of the knowledge base.

A note is YAML frontmatter plus a Markdown body. Known fields are validated;
any other frontmatter keys (for example ones a user adds in Obsidian) are kept
in ``extra`` so that reading and writing a note never loses data.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from essence.ids import new_id


class NoteType(StrEnum):
    SOURCE = "source"
    CONCEPT = "concept"
    CLAIM = "claim"
    INSIGHT = "insight"
    DECISION = "decision"
    QUESTION = "question"

    @property
    def prefix(self) -> str:
        return _PREFIX[self]

    @property
    def folder(self) -> str:
        """Folder for this type, relative to the vault root."""
        if self is NoteType.SOURCE:
            return "raw"
        return f"notes/{self.value}s"


_PREFIX = {
    NoteType.SOURCE: "src",
    NoteType.CONCEPT: "con",
    NoteType.CLAIM: "clm",
    NoteType.INSIGHT: "ins",
    NoteType.DECISION: "dec",
    NoteType.QUESTION: "qst",
}

KNOWLEDGE_TYPES = [t for t in NoteType if t is not NoteType.SOURCE]


class NoteStatus(StrEnum):
    SUGGESTED = "suggested"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    INVALID = "invalid"


class Origin(StrEnum):
    AI = "ai"
    USER = "user"
    AI_EDITED = "ai+edited"


# Frontmatter key order when writing. Type-specific keys follow the common ones.
FIELD_ORDER = [
    "id",
    "type",
    "title",
    "status",
    "applies_when",
    "not_when",
    "sources",
    "links",
    "valid_from",
    "valid_to",
    "version",
    "created_by",
    "created",
    "updated",
    # source
    "source_kind",
    "origin_ref",
    "captured",
    "never_cloud",
    # claim
    "confidence",
    # decision
    "expected_outcome",
    "review_on",
    "actual_outcome",
    "lesson",
]

_NO_EMIT_IF_NONE = set(FIELD_ORDER) - {"id", "type", "title", "status", "version"}


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


class Note(BaseModel):
    model_config = ConfigDict(use_enum_values=False, validate_assignment=True)

    id: str = ""
    type: NoteType
    title: str
    status: NoteStatus = NoteStatus.ACTIVE
    applies_when: str | None = None
    not_when: str | None = None
    # Provenance: "src_<id>" or "src_<id>#exc_<n>" (excerpt-level).
    sources: list[str] = Field(default_factory=list)
    # Ids of related notes.
    links: list[str] = Field(default_factory=list)
    valid_from: date | None = None
    valid_to: date | None = None
    version: int = 1
    created_by: Origin = Origin.USER
    created: datetime = Field(default_factory=_now)
    updated: datetime | None = None

    # Source-only
    source_kind: str | None = None  # chat | web | doc
    origin_ref: str | None = None  # URL, file name or conversation id
    captured: datetime | None = None
    never_cloud: bool | None = None

    # Claim-only
    confidence: float | None = None

    # Decision-only
    expected_outcome: str | None = None
    review_on: date | None = None
    actual_outcome: str | None = None
    lesson: str | None = None

    body: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _assign_id(self) -> Note:
        if not self.id:
            # Bypass validate_assignment recursion.
            object.__setattr__(self, "id", new_id(self.type.prefix))
        return self

    @field_validator("confidence")
    @classmethod
    def _confidence_range(cls, v: float | None) -> float | None:
        if v is not None and not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        return v

    @field_validator("version")
    @classmethod
    def _version_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("version must be >= 1")
        return v

    def frontmatter(self) -> dict[str, Any]:
        """Ordered frontmatter dict, as written to disk."""
        data = self.model_dump(mode="json", exclude={"body", "extra"})
        out: dict[str, Any] = {}
        for key in FIELD_ORDER:
            value = data.get(key)
            if key in _NO_EMIT_IF_NONE and (value is None or value == []):
                continue
            out[key] = value
        for key, value in self.extra.items():
            if key not in out:
                out[key] = value
        return out
