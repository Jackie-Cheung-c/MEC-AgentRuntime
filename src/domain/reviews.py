"""Persisted image extraction and human review contracts."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from pydantic import Field, JsonValue, model_validator

from src.domain.base import DomainModel
from src.domain.evidence import Evidence, ExtractedDocument


class FieldCorrectionInput(DomainModel):
    field_path: str = Field(min_length=1, max_length=255)
    original_value: JsonValue
    corrected_value: JsonValue
    note: str | None = Field(default=None, max_length=1000)


class FieldCorrection(DomainModel):
    correction_id: UUID = Field(default_factory=uuid4)
    attachment_id: UUID
    document_id: UUID
    field_path: str = Field(min_length=1, max_length=255)
    original_value: JsonValue
    corrected_value: JsonValue
    note: str | None = Field(default=None, max_length=1000)
    corrected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CorrectionBatch(DomainModel):
    document_id: UUID
    corrections: list[FieldCorrectionInput] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def reject_duplicate_paths(self) -> "CorrectionBatch":
        paths = [item.field_path for item in self.corrections]
        if len(paths) != len(set(paths)):
            raise ValueError("a correction batch cannot repeat a field path")
        return self


class AttachmentExtractionRecord(DomainModel):
    attachment_id: UUID
    document: ExtractedDocument
    evidence: list[Evidence]
    corrections: list[FieldCorrection] = Field(default_factory=list)
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def validate_links(self) -> "AttachmentExtractionRecord":
        if self.document.attachment_id != self.attachment_id:
            raise ValueError("document must belong to the extraction attachment")
        available = {item.evidence_id for item in self.evidence}
        if set(self.document.evidence_ids) - available:
            raise ValueError("document references evidence outside this extraction")
        return self
