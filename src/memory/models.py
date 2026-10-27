"""持久、用户作用域的长期记忆的领域契约"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import Field, JsonValue, field_validator, model_validator

from src.domain.base import DomainModel


class MemoryCategory(StrEnum):
    PROFILE = "profile"
    PREFERENCE = "preference"
    CONSTRAINT = "constraint"
    EPISODIC = "episodic"


class MemoryScope(StrEnum):
    GLOBAL = "global"
    CONTEXT = "context"
    THREAD = "thread"
    # Legacy persisted value. New writes normalize it to CONTEXT.
    TRIP = "trip"


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


class MemoryContext(DomainModel):
    """Retrieval dimensions for one Agent turn."""

    domain: str = Field(
        default="general",
        pattern=r"^[a-z][a-z0-9_-]*$",
        max_length=100,
    )
    workflow_id: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_-]*$",
        max_length=100,
    )
    scope_id: str | None = Field(default=None, min_length=1, max_length=255)
    thread_id: str | None = Field(default=None, min_length=1, max_length=255)


class MemoryItem(DomainModel):
    """可审计的长期记忆
    `scope` 控制条目何时可被检索
        - global:同一user的所有会话都可访问
        - context: 仅scope_id匹配时可用
        - thread: 仅thread_id匹配时可用
    """

    memory_id: UUID = Field(default_factory=uuid4)
    user_id: str = Field(min_length=1, max_length=128)
    category: MemoryCategory
    memory_key: str = Field(min_length=1, max_length=128)
    value: JsonValue
    scope: MemoryScope
    scope_id: str | None = Field(default=None, min_length=1, max_length=255)
    domain: str = Field(
        default="general",
        pattern=r"^[a-z][a-z0-9_-]*$",
        max_length=100,
    )
    workflow_id: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_-]*$",
        max_length=100,
    )
    # Deprecated travel alias. New code reads scope_id.
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

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_context(cls, value):
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        scope = data.get("scope")
        if scope == MemoryScope.TRIP or (
            isinstance(scope, str) and scope.strip().casefold() == "trip"
        ):
            data["scope"] = MemoryScope.CONTEXT
        elif str(scope).strip().casefold() == "global":
            data["scope"] = MemoryScope.GLOBAL
            data["workflow_id"] = None

        scope_id = data.get("scope_id")
        trip_id = data.get("trip_id")
        if scope_id is None and trip_id is not None:
            scope_id = str(trip_id)
        if scope_id is not None:
            data["scope_id"] = str(scope_id)
            if trip_id is None:
                try:
                    data["trip_id"] = UUID(str(scope_id))
                except (ValueError, TypeError):
                    pass

        memory_key = str(data.get("memory_key") or "").strip()
        if memory_key:
            from src.memory.key_policy import memory_domain_for_key

            key_domain = memory_domain_for_key(memory_key)
            if key_domain is not None:
                data["domain"] = key_domain
                if key_domain == "general":
                    data["workflow_id"] = None
        return data

    @model_validator(mode="after")
    def validate_scope_and_lifetime(self) -> "MemoryItem":
        canonical_scope_id = self.scope_id or (
            str(self.trip_id) if self.trip_id is not None else None
        )
        if (
            self.scope_id is not None
            and self.trip_id is not None
            and self.scope_id != str(self.trip_id)
        ):
            raise ValueError("scope_id and legacy trip_id must match")
        if self.scope == MemoryScope.GLOBAL:
            if (
                self.scope_id is not None
                or self.trip_id is not None
                or self.thread_id is not None
            ):
                raise ValueError(
                    "global memory cannot have scope_id, trip_id, or thread_id"
                )
        elif self.scope == MemoryScope.CONTEXT:
            if canonical_scope_id is None:
                raise ValueError("context memory requires scope_id")
            if self.thread_id is not None:
                raise ValueError("context memory cannot have thread_id")
        elif self.scope == MemoryScope.THREAD:
            if self.thread_id is None:
                raise ValueError("thread-scoped memory requires thread_id")
            if canonical_scope_id is not None:
                raise ValueError("thread memory cannot have scope_id or trip_id")

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

    @field_validator("scope", mode="before")
    @classmethod
    def normalize_legacy_scope(cls, value):
        if isinstance(value, str) and value.strip().casefold() == "trip":
            return MemoryScope.CONTEXT
        return value


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

    @field_validator("scope", mode="before")
    @classmethod
    def normalize_legacy_scope(cls, value):
        if isinstance(value, str) and value.strip().casefold() == "trip":
            return MemoryScope.CONTEXT
        return value


class MemoryExtraction(DomainModel):
    """Structured response expected from the long-term memory extractor."""

    memories: list[MemoryCandidate] = Field(default_factory=list)
