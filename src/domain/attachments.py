"""用户上传文件的元数据规范"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import Field, JsonValue, model_validator

from src.domain.base import DomainModel

## 附件类型、来源、媒体信息、存储引用和尺寸校验


class AttachmentKind(StrEnum):
    IMAGE = "image"
    DOCUMENT = "document"
    AUDIO = "audio"
    VIDEO = "video"
    OTHER = "other"


class AttachmentSource(StrEnum):
    UPLOAD = "upload"
    CAMERA = "camera"
    MESSAGE = "message"
    IMPORT = "import"


class AttachmentStatus(StrEnum):
    """Lifecycle state of an attachment and its downstream extraction work."""

    AVAILABLE = "available"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    DELETED = "deleted"


class Attachment(DomainModel):
    """Reference to an uploaded asset; binary content lives in object storage."""

    attachment_id: UUID = Field(default_factory=uuid4)
    kind: AttachmentKind
    source: AttachmentSource = AttachmentSource.UPLOAD
    filename: str = Field(min_length=1, max_length=255)
    media_type: str = Field(
        min_length=3,
        max_length=127,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*/[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*$",
    )# 类型校验
    size_bytes: int = Field(gt=0)
    storage_uri: str = Field(min_length=1, max_length=2048)
    sha256: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")  # 去重
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    page_count: int | None = Field(default=None, gt=0)
    duration_ms: int | None = Field(default=None, gt=0)
    status: AttachmentStatus = AttachmentStatus.AVAILABLE
    status_reason: str | None = Field(default=None, max_length=1000)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_media_metadata(self) -> "Attachment":
        if (self.width is None) != (self.height is None):
            raise ValueError("width and height must be provided together")
        if self.kind == AttachmentKind.IMAGE and not self.media_type.startswith("image/"):
            raise ValueError("image attachments must use an image media type")
        if self.kind == AttachmentKind.AUDIO and not self.media_type.startswith("audio/"):
            raise ValueError("audio attachments must use an audio media type")
        if self.kind == AttachmentKind.VIDEO and not self.media_type.startswith("video/"):
            raise ValueError("video attachments must use a video media type")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot be earlier than created_at")
        if self.status == AttachmentStatus.FAILED and not self.status_reason:
            raise ValueError("failed attachments must include status_reason")
        return self
