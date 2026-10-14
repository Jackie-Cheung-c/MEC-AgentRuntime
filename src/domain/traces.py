"""Structured execution traces shared by runtime and evaluation code."""
# 运行时及离线评测结构化执行追踪
from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import Field, JsonValue, model_validator

from src.domain.base import DomainModel


class TraceSpanKind(StrEnum):
    AGENT = "agent"
    CHAIN = "chain"
    MODEL = "model"
    TOOL = "tool"
    RETRIEVER = "retriever"
    OTHER = "other"


class TraceSpanStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"

# 记录某一具体步骤
class TraceSpan(DomainModel):
    span_id: str = Field(min_length=1, max_length=255)
    parent_span_ids: list[str] = Field(default_factory=list)
    name: str = Field(min_length=1, max_length=255)
    kind: TraceSpanKind
    status: TraceSpanStatus
    input: JsonValue | None = None
    output: JsonValue | None = None
    error: str | None = Field(default=None, max_length=4000)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    duration_ms: int | None = Field(default=None, ge=0)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_outcome(self) -> "TraceSpan":
        if self.status == TraceSpanStatus.FAILED and not self.error:
            raise ValueError("failed trace spans must include an error")
        if self.completed_at and self.completed_at < self.started_at:
            raise ValueError("completed_at cannot be before started_at")
        return self

# 打包TraceSpan，形成一个完整的执行追踪
class ExecutionTrace(DomainModel):
    trace_id: UUID = Field(default_factory=uuid4)
    spans: list[TraceSpan] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

    @model_validator(mode="after")
    def validate_spans(self) -> "ExecutionTrace":
        span_ids = [span.span_id for span in self.spans]
        if len(span_ids) != len(set(span_ids)):
            raise ValueError("trace span IDs must be unique")
        if self.completed_at and self.completed_at < self.started_at:
            raise ValueError("completed_at cannot be before started_at")
        return self
