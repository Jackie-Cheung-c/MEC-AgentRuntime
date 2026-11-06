
"""Policy contracts for domain-agnostic attachment extraction."""

from __future__ import annotations

import hashlib
import json

from pydantic import Field, model_validator

from src.domain.base import DomainModel
from src.domain.evidence import DocumentType


class ExtractionDocumentType(DomainModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$", max_length=100)
    label: str | None = Field(default=None, max_length=80)
    description: str = Field(min_length=1, max_length=500)
    persisted_type: DocumentType = DocumentType.OTHER
    field_hints: dict[str, str] = Field(default_factory=dict)


class ExtractionPolicy(DomainModel):
    policy_id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$", max_length=100)
    version: str = Field(default="1", min_length=1, max_length=50)
    document_types: list[ExtractionDocumentType] = Field(min_length=1, max_length=50)
    default_document_type: str = Field(
        default="other",
        pattern=r"^[a-z][a-z0-9_-]*$",
        max_length=100,
    )
    instructions: list[str] = Field(default_factory=list, max_length=50)
    max_fields: int = Field(default=100, ge=1, le=100)

    @model_validator(mode="after")
    def validate_types(self) -> "ExtractionPolicy":
        ids = [item.id for item in self.document_types]
        if len(ids) != len(set(ids)):
            raise ValueError("document type ids must be unique")
        if self.default_document_type not in ids:
            raise ValueError("default_document_type is not defined")
        return self

    @property
    def document_type_ids(self) -> set[str]:
        return {item.id for item in self.document_types}

    def document_type(self, type_id: str) -> ExtractionDocumentType:
        for item in self.document_types:
            if item.id == type_id:
                return item
        raise ValueError(f"unknown document type: {type_id}")

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def general_extraction_policy() -> ExtractionPolicy:
    return ExtractionPolicy(
        policy_id="general-document",
        document_types=[
            ExtractionDocumentType(
                id="document",
                label="文档",
                description="General document with visible text or structured fields.",
                persisted_type=DocumentType.OTHER,
                field_hints={
                    "document.title": "Document title.",
                    "document.date": "Date shown on the document.",
                },
            ),
            ExtractionDocumentType(
                id="receipt",
                label="收据",
                description="Receipt, invoice, or payment record.",
                persisted_type=DocumentType.RECEIPT,
                field_hints={
                    "receipt.merchant": "Merchant or issuer.",
                    "receipt.total": "Total amount.",
                    "receipt.currency": "Currency.",
                    "receipt.paid_at": "Payment time.",
                },
            ),
            ExtractionDocumentType(
                id="other",
                label="其他",
                description="A document that does not match another allowed type.",
                persisted_type=DocumentType.OTHER,
            ),
        ],
        default_document_type="other",
        instructions=[
            "Extract only content visible in the image.",
            "Do not assume a travel, finance, or other business context.",
        ],
    )
