"""证据与多模态提取契约"""

### -----------------标准化坐标、页码/时间定位、抽取字段和文档结果----------------------------
### 防止幻觉，可审计    evidence-->extracted_field-->extracted_document
from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import Field, JsonValue, model_validator

from src.domain.base import DomainModel

# 证据来源
class EvidenceSource(StrEnum):
    ATTACHMENT = "attachment"
    TOOL = "tool"
    USER = "user"
    AGENT = "agent"


class BoundingBox(DomainModel):
    """基于左上角的归一化边界框（坐标值域 [0, 1]）"""

    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def stay_inside_source(self) -> "BoundingBox":
        if self.x + self.width > 1 or self.y + self.height > 1:
            raise ValueError("bounding box must remain inside the source")
        return self

## 支持页码（PDF）、归一化边界框（图像）、时间戳毫秒（视频/音频）、字符偏移（纯文本）。
class EvidenceLocator(DomainModel):
    page_number: int | None = Field(default=None, ge=1)
    bounding_box: BoundingBox | None = None
    start_ms: int | None = Field(default=None, ge=0)
    end_ms: int | None = Field(default=None, ge=0)
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_ranges(self) -> "EvidenceLocator":
        self._validate_pair("start_ms", self.start_ms, "end_ms", self.end_ms)
        self._validate_pair("char_start", self.char_start, "char_end", self.char_end)
        return self

    @staticmethod
    def _validate_pair(start_name: str, start: int | None, end_name: str, end: int | None) -> None:
        if (start is None) != (end is None):
            raise ValueError(f"{start_name} and {end_name} must be provided together")
        if start is not None and end is not None and end <= start:
            raise ValueError(f"{end_name} must be greater than {start_name}")

## 精准定位结果来源，可分辨是“用户输入误导”、“工具返回错误”还是“Agent 编造”
class Evidence(DomainModel):
    evidence_id: UUID = Field(default_factory=uuid4)
    source: EvidenceSource
    source_id: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=1, max_length=8000)
    confidence: float = Field(default=1.0, ge=0, le=1)
    locator: EvidenceLocator | None = None
    observed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

# 提取字段
class ExtractedField(DomainModel):
    path: str = Field(min_length=1, max_length=255)
    value: JsonValue
    confidence: float = Field(ge=0, le=1)
    evidence_ids: list[UUID] = Field(min_length=1)
    requires_confirmation: bool = False

# 文档类型
class DocumentType(StrEnum):
    FLIGHT_TICKET = "flight_ticket"
    TRAIN_TICKET = "train_ticket"
    HOTEL_BOOKING = "hotel_booking"
    ATTRACTION_BOOKING = "attraction_booking"
    CANCELLATION_NOTICE = "cancellation_notice"
    DELAY_NOTICE = "delay_notice"
    RECEIPT = "receipt"
    MAP = "map"
    MENU = "menu"
    SIGN = "sign"
    OTHER = "other"


class ExtractionStatus(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"

# 提取文档（基于证据有来源）
class ExtractedDocument(DomainModel):
    document_id: UUID = Field(default_factory=uuid4)
    attachment_id: UUID
    document_type: DocumentType = DocumentType.OTHER
    status: ExtractionStatus = ExtractionStatus.PENDING
    fields: list[ExtractedField] = Field(default_factory=list)
    evidence_ids: list[UUID] = Field(default_factory=list)
    raw_text: str | None = Field(default=None, max_length=50000)
    extractor: str | None = Field(default=None, max_length=100)
    extractor_version: str | None = Field(default=None, max_length=100)
    warnings: list[str] = Field(default_factory=list)
    ### 跨字段校验,检查ExtractedField 里LLM给的evidence_ids，是否全部声明在agent工具提取的ExtractedDocument的 evidence_ids 列表中
    @model_validator(mode="after")
    def validate_evidence_references(self) -> "ExtractedDocument":
        declared = set(self.evidence_ids)
        referenced = {item for field in self.fields for item in field.evidence_ids}
        missing = referenced - declared
        if missing:
            raise ValueError("field evidence must be declared by the document")
        return self
