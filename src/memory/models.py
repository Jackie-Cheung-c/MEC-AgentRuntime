"""持久、用户作用域的长期记忆的领域契约"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import Field, JsonValue, model_validator

from src.domain.base import DomainModel


class MemoryCategory(StrEnum):
    PROFILE = "profile"
    PREFERENCE = "preference"
    CONSTRAINT = "constraint"
    EPISODIC = "episodic"


class MemoryScope(StrEnum):
    GLOBAL = "global"
    TRIP = "trip"
    THREAD = "thread"


class MemoryStatus(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DELETED = "deleted"
    REJECTED = "rejected"


class MemorySourceType(StrEnum):
    USER_EXPLICIT = "user_explicit"
    CONFIRMED_EVIDENCE = "confirmed_evidence"
    CORRECTION = "correction"
    SYSTEM_INFERENCE = "system_inference"


class MemoryItem(DomainModel):
    """可审计的长期记忆
    `scope` 控制条目何时可被检索
    - global:同一user的所有会话都可访问
    - trip: 仅trip_id匹配时可用
    - thread: 仅thread_id匹配时可用
    """

    memory_id: UUID = Field(default_factory=uuid4)
    user_id: str = Field(min_length=1, max_length=128)
    category: MemoryCategory
    memory_key: str = Field(min_length=1, max_length=128)
    value: JsonValue
    scope: MemoryScope
    trip_id: UUID | None = None
    thread_id: str | None = Field(default=None, min_length=1, max_length=255)
    confidence: float = Field(ge=0.0, le=1.0)
    source_type: MemorySourceType
    source_ref: str = Field(min_length=1, max_length=255)
    evidence_quote: str | None = Field(default=None, max_length=2000)
    status: MemoryStatus = MemoryStatus.ACTIVE
    supersedes_id: UUID | None = None
    valid_from: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_scope_and_lifetime(self) -> "MemoryItem":
        if self.scope == MemoryScope.GLOBAL:
            if self.trip_id is not None or self.thread_id is not None:
                raise ValueError(
                    "global memory cannot be scoped to a trip_id or thread_id"
                )
        elif self.scope == MemoryScope.TRIP:
            if self.trip_id is None:
                raise ValueError("trip-scoped memory requires trip_id")
        elif self.scope == MemoryScope.THREAD:
            if self.thread_id is None:
                raise ValueError("thread-scoped memory requires thread_id")

        if self.expires_at is not None and self.expires_at <= self.valid_from:
            raise ValueError("expires_at must be later than valid_from")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot be earlier than created_at")
        return self

    def is_active_at(self, moment: datetime) -> bool:
        """Return whether this item is currently eligible for retrieval."""
        if self.status != MemoryStatus.ACTIVE:
            return False
        if moment < self.valid_from:
            return False
        return self.expires_at is None or moment < self.expires_at


class MemorySearchResult(DomainModel):
    """A memory item returned from a candidate or ranked search."""

    item: MemoryItem
    score: float


class MemoryCandidate(DomainModel):
    """One validated extraction candidate produced from a user message."""

    category: MemoryCategory
    memory_key: str = Field(min_length=1, max_length=128)
    value: JsonValue
    scope: MemoryScope
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_quote: str = Field(min_length=1, max_length=2000)


class MemoryExtraction(DomainModel):
    """Structured response expected from the long-term memory extractor."""

    memories: list[MemoryCandidate] = Field(default_factory=list)
