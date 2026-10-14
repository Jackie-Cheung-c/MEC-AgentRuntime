"""应用与评测共用的统一请求和响应契约"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import Field, JsonValue, model_validator

from src.domain.attachments import Attachment
from src.domain.base import DomainModel
from src.domain.evidence import Evidence, ExtractedDocument
from src.domain.traces import ExecutionTrace
from src.domain.travel import TripState, TravelEvent


class RunStatus(StrEnum):
    RUNNING = "running"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"


class ToolCallStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ToolCallRecord(DomainModel):
    call_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1, max_length=255)
    name: str = Field(min_length=1, max_length=255)
    status: ToolCallStatus
    input: JsonValue = Field(default_factory=dict)
    output: JsonValue | None = None
    error: str | None = Field(default=None, max_length=4000)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_outcome(self) -> "ToolCallRecord":
        if self.status == ToolCallStatus.FAILED and not self.error:
            raise ValueError("failed tool calls must include an error")
        if self.started_at and self.completed_at and self.completed_at < self.started_at:
            raise ValueError("completed_at cannot be before started_at")
        return self


class ConfirmationRequest(DomainModel):
    confirmation_id: UUID = Field(default_factory=uuid4)
    action: str = Field(min_length=1, max_length=255)
    prompt: str = Field(min_length=1, max_length=2000)
    payload: JsonValue = Field(default_factory=dict)
    evidence_ids: list[UUID] = Field(default_factory=list)
    expires_at: datetime | None = None


class RunMetrics(DomainModel):
    duration_ms: int = Field(default=0, ge=0)
    model_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    retry_count: int = Field(default=0, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    estimated_cost: Decimal | None = Field(default=None, ge=0, decimal_places=6)
    cost_currency: str = Field(default="USD", pattern=r"^[A-Z]{3}$")


class AgentRunRequest(DomainModel):
    request_id: UUID = Field(default_factory=uuid4)
    user_id: str = Field(default="local_user", min_length=1, max_length=128)
    thread_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1, max_length=255)
    query: str = Field(default="", max_length=4000)
    trip_id: UUID | None = None
    attachments: list[Attachment] = Field(default_factory=list, max_length=10)
    locale: str = Field(default="zh-CN", min_length=2, max_length=35)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_input(self) -> "AgentRunRequest":
        if not self.query and not self.attachments:
            raise ValueError("query or at least one attachment is required")
        attachment_ids = [item.attachment_id for item in self.attachments]
        if len(attachment_ids) != len(set(attachment_ids)):
            raise ValueError("attachment IDs must be unique")
        return self


class AgentRunResult(DomainModel):
    run_id: UUID = Field(default_factory=uuid4)
    request_id: UUID
    thread_id: str = Field(min_length=1, max_length=255)
    status: RunStatus
    answer: str = ""
    selected_skill: str | None = Field(default=None, max_length=255)
    trip_state: TripState | None = None
    events: list[TravelEvent] = Field(default_factory=list)
    extracted_documents: list[ExtractedDocument] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    trace: ExecutionTrace | None = None
    pending_confirmations: list[ConfirmationRequest] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = Field(default=None, max_length=4000)
    metrics: RunMetrics = Field(default_factory=RunMetrics)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

    @model_validator(mode="after")
    def validate_result(self) -> "AgentRunResult":
        if self.completed_at and self.completed_at < self.started_at:
            raise ValueError("completed_at cannot be before started_at")
        if self.status == RunStatus.FAILED and not self.error:
            raise ValueError("failed runs must include an error")
        if self.status == RunStatus.AWAITING_CONFIRMATION and not self.pending_confirmations:
            raise ValueError("awaiting_confirmation runs must include a confirmation request")
        if self.status in {RunStatus.SUCCEEDED, RunStatus.PARTIAL} and not self.answer:
            raise ValueError("successful or partial runs must include an answer")

        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):  # 全局唯一evidence_id
            raise ValueError("evidence IDs must be unique")
        available = set(evidence_ids)
        referenced = {   ## 提取的文档字段里引用的证据
            evidence_id
            for document in self.extracted_documents
            for evidence_id in document.evidence_ids
        }
        referenced.update(  ## 事件中引用的证据
            evidence_id for event in self.events for evidence_id in event.evidence_ids
        )
        referenced.update(  ## 待确认请求里引用的证据
            evidence_id    
            for confirmation in self.pending_confirmations
            for evidence_id in confirmation.evidence_ids
        )
        if referenced - available:
            raise ValueError("result contains references to missing evidence")
        return self

class ThreadTurn(DomainModel):
    """One persisted question/answer pair inside a conversation thread."""

    run_id: str | None = None
    query: str = ""
    answer: str = ""
    status: str = "succeeded"
    started_at: datetime | None = None


class ThreadSummary(DomainModel):
    """One conversation thread as shown in the sidebar history list."""

    thread_id: str = Field(min_length=1, max_length=255)
    title: str = Field(default="", max_length=200)
    turns: int = Field(default=0, ge=0)
    trip_id: UUID | None = None
    updated_at: datetime | None = None