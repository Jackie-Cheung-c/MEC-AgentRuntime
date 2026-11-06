# 定义通用状态信封、事件和产物

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from pydantic import Field, JsonValue

from src.domain.base import DomainModel


class StateEnvelope(DomainModel):
    """Scenario-neutral wrapper around a persisted state payload."""

    adapter: str = Field(min_length=1, max_length=100)
    state_id: UUID
    version: int | None = Field(default=None, ge=1)
    payload: JsonValue
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class AgentEvent(DomainModel):
    """Generic event emitted by a state adapter or tool."""

    event_id: UUID = Field(default_factory=uuid4)
    event_type: str = Field(min_length=1, max_length=100)
    severity: str = Field(default="warning", max_length=50)
    status: str = Field(default="open", max_length=50)
    summary: str = Field(min_length=1, max_length=1000)
    occurred_at: datetime | None = None
    detected_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    affected_refs: list[str] = Field(default_factory=list)
    affected_item_ids: list[UUID] = Field(default_factory=list)
    evidence_ids: list[UUID] = Field(default_factory=list)
    details: dict[str, JsonValue] = Field(default_factory=dict)


class AgentArtifact(DomainModel):
    """Generic files, reports, plans, or other outputs."""

    artifact_id: UUID = Field(default_factory=uuid4)
    kind: str = Field(min_length=1, max_length=100)
    name: str | None = Field(default=None, max_length=255)
    uri: str | None = Field(default=None, max_length=2000)
    payload: JsonValue | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

class StateMutation(DomainModel):
    """Structured state change returned by a stateful tool."""

    adapter: str = Field(min_length=1, max_length=100)
    operation: str = Field(min_length=1, max_length=100)
    state_id: UUID | None = None
    version: int | None = Field(default=None, ge=1)
    reference_id: UUID | None = None
    data: dict[str, JsonValue] = Field(default_factory=dict)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)