"""Object storage and metadata persistence for uploaded attachments."""

from __future__ import annotations

import sqlite3
import threading
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Protocol
from uuid import UUID

from src.domain import (
    Attachment,
    AttachmentExtractionRecord,
    AttachmentStatus,
    CorrectionBatch,
    Evidence,
    ExtractedDocument,
    ExtractionStatus,
    FieldCorrection,
)
from src.paths import DATA_DIR

# 协议类，类似接口
class ObjectStorage(Protocol):
    bucket: str

    def put_object(
        self,
        *,
        object_name: str,
        data: BinaryIO,
        length: int,
        content_type: str,
        metadata: dict[str, str] | None = None,
    ) -> str: ...

    def remove_object(self, object_name: str) -> None: ...

    def get_object(self, object_name: str) -> bytes: ...


class MinioConfigurationError(RuntimeError):
    """Raised when required MinIO connection settings are absent."""


class AttachmentStateConflictError(RuntimeError):
    """Raised when an optimistic attachment status update loses a race."""


class ExtractionNotFoundError(LookupError):
    """Raised when an owned attachment has no matching extraction snapshot."""


class CorrectionConflictError(RuntimeError):
    """Raised when a correction targets stale or unknown extracted data."""


class MinioObjectStorage:
    """Store attachment binaries in a private MinIO bucket."""

    def __init__(
        self,
        *,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = False,
        client: Any | None = None,
    ) -> None:
        if not endpoint.strip():
            raise MinioConfigurationError("MINIO_ENDPOINT is required.")
        if not bucket.strip():
            raise MinioConfigurationError("MINIO_BUCKET is required.")
        if client is None:
            if not access_key or not secret_key:
                raise MinioConfigurationError(
                    "MINIO_ACCESS_KEY and MINIO_SECRET_KEY are required."
                )
            from minio import Minio

            try:
                client = Minio(
                    endpoint,
                    access_key=access_key,
                    secret_key=secret_key,
                    secure=secure,
                )
            except ValueError as error:
                raise MinioConfigurationError(
                    f"Invalid MinIO configuration: {error}"
                ) from error
        self.client = client
        self.bucket = bucket
        self._bucket_ready = False
        # 多线程环境下只加载一次bucket
        self._bucket_lock = threading.Lock()

    def _ensure_bucket(self) -> None:
        if self._bucket_ready:
            return
        with self._bucket_lock:
            if self._bucket_ready:
                return
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
            self._bucket_ready = True

    def put_object(
        self,
        *,
        object_name: str,
        data: BinaryIO,
        length: int,
        content_type: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        self._ensure_bucket()
        data.seek(0)
        self.client.put_object(
            self.bucket,
            object_name,
            data,
            length,
            content_type=content_type,
            metadata=metadata,
        )
        return f"minio://{self.bucket}/{object_name}"

    def remove_object(self, object_name: str) -> None:
        self.client.remove_object(self.bucket, object_name)

    def get_object(self, object_name: str) -> bytes:
        response = self.client.get_object(self.bucket, object_name)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

# 将 Attachment 对象（Pydantic 模型）序列化为 JSON 存入 SQLite
class SQLiteAttachmentRepository:
    """Persist attachment metadata and enforce user ownership."""

    def __init__(self, path: str | Path = DATA_DIR / "attachments.db") -> None:
        self.path = Path(path)
        self._initialize()

    @contextmanager
    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS attachments (
                    attachment_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    storage_uri TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL DEFAULT 'available',
                    attachment_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_attachments_user_created
                    ON attachments(user_id, created_at DESC);
                CREATE UNIQUE INDEX IF NOT EXISTS uq_attachments_user_sha256
                    ON attachments(user_id, sha256);
                CREATE TABLE IF NOT EXISTS attachment_extractions (
                    document_id TEXT PRIMARY KEY,
                    attachment_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    document_json TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    extracted_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (attachment_id) REFERENCES attachments(attachment_id)
                );
                CREATE INDEX IF NOT EXISTS idx_extractions_attachment_created
                    ON attachment_extractions(user_id, attachment_id, extracted_at DESC);
                CREATE TABLE IF NOT EXISTS attachment_corrections (
                    correction_id TEXT PRIMARY KEY,
                    attachment_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    correction_json TEXT NOT NULL,
                    corrected_at TEXT NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES attachment_extractions(document_id)
                );
                CREATE INDEX IF NOT EXISTS idx_corrections_document_created
                    ON attachment_corrections(user_id, document_id, corrected_at ASC);
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(attachments)").fetchall()
            }
            if "status" not in columns:
                connection.execute(
                    "ALTER TABLE attachments ADD COLUMN status TEXT "
                    "NOT NULL DEFAULT 'available'"
                )
            if "updated_at" not in columns:
                connection.execute(
                    "ALTER TABLE attachments ADD COLUMN updated_at TEXT NOT NULL DEFAULT ''"
                )
                connection.execute(
                    "UPDATE attachments SET updated_at = created_at WHERE updated_at = ''"
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_attachments_user_status "
                "ON attachments(user_id, status, created_at DESC)"
            )

    def save(self, user_id: str, attachment: Attachment) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO attachments (
                    attachment_id, user_id, sha256, storage_uri,
                    status, attachment_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(attachment.attachment_id),
                    user_id,
                    attachment.sha256,
                    attachment.storage_uri,
                    attachment.status.value,
                    attachment.model_dump_json(),
                    attachment.created_at.isoformat(),
                    attachment.updated_at.isoformat(),
                ),
            )

    def get(
        self,
        user_id: str,
        attachment_id: UUID | str,
        *,
        include_deleted: bool = False,
    ) -> Attachment | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT attachment_json FROM attachments
                WHERE attachment_id = ? AND user_id = ?
                  AND (? = 1 OR status <> ?)
                """,
                (
                    str(attachment_id),
                    user_id,
                    int(include_deleted),
                    AttachmentStatus.DELETED.value,
                ),
            ).fetchone()
        return Attachment.model_validate_json(row["attachment_json"]) if row else None

    def get_by_sha256(
        self,
        user_id: str,
        sha256: str,
        *,
        include_deleted: bool = False,
    ) -> Attachment | None:
        """
        按用户和文件 SHA-256 查找已有附件。同一用户上传同一文件时，返回已有附件。
        """
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT attachment_json
                FROM attachments
                WHERE user_id = ?
                AND sha256 = ?
                AND (? = 1 OR status <> ?)
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (
                    user_id,
                    sha256.lower(),
                    int(include_deleted),
                    AttachmentStatus.DELETED.value,
                ),
            ).fetchone()

        return Attachment.model_validate_json(row["attachment_json"]) if row else None


    def update_status(
        self,
        user_id: str,
        attachment: Attachment,
        *,
        expected_status: AttachmentStatus,
    ) -> None:
        """Atomically persist a status transition for one owning user."""
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE attachments
                SET status = ?, attachment_json = ?, updated_at = ?
                WHERE attachment_id = ? AND user_id = ? AND status = ?
                """,
                (
                    attachment.status.value,
                    attachment.model_dump_json(),
                    attachment.updated_at.isoformat(),
                    str(attachment.attachment_id),
                    user_id,
                    expected_status.value,
                ),
            )
            if cursor.rowcount != 1:
                raise AttachmentStateConflictError(
                    "Attachment status changed before the update completed."
                )

    def list_for_user(self, user_id: str, limit: int = 50) -> list[Attachment]:
        if limit < 1:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT attachment_json FROM attachments
                WHERE user_id = ? AND status <> ?
                ORDER BY created_at DESC LIMIT ?
                """,
                (user_id, AttachmentStatus.DELETED.value, limit),
            ).fetchall()
        return [Attachment.model_validate_json(row["attachment_json"]) for row in rows]

    def save_extraction(
        self,
        user_id: str,
        document: ExtractedDocument,
        evidence: list[Evidence],
    ) -> AttachmentExtractionRecord:
        """Persist one immutable model extraction before human review begins."""
        record = AttachmentExtractionRecord(
            attachment_id=document.attachment_id,
            document=document,
            evidence=evidence,
        )
        with self._connect() as connection:
            owned = connection.execute(
                """
                SELECT 1 FROM attachments
                WHERE attachment_id = ? AND user_id = ? AND status <> ?
                """,
                (
                    str(document.attachment_id),
                    user_id,
                    AttachmentStatus.DELETED.value,
                ),
            ).fetchone()
            if owned is None:
                raise ExtractionNotFoundError("Attachment not found.")
            connection.execute(
                """
                INSERT INTO attachment_extractions (
                    document_id, attachment_id, user_id, document_json,
                    evidence_json, extracted_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(document.document_id),
                    str(document.attachment_id),
                    user_id,
                    document.model_dump_json(),
                    json.dumps(
                        [item.model_dump(mode="json") for item in evidence],
                        ensure_ascii=False,
                    ),
                    record.extracted_at.isoformat(),
                    record.updated_at.isoformat(),
                ),
            )
        return record

    def get_latest_extraction(
        self, user_id: str, attachment_id: UUID | str
    ) -> AttachmentExtractionRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM attachment_extractions
                WHERE attachment_id = ? AND user_id = ?
                ORDER BY extracted_at DESC LIMIT 1
                """,
                (str(attachment_id), user_id),
            ).fetchone()
            if row is None:
                return None
            correction_rows = connection.execute(
                """
                SELECT correction_json FROM attachment_corrections
                WHERE document_id = ? AND user_id = ?
                ORDER BY corrected_at ASC
                """,
                (row["document_id"], user_id),
            ).fetchall()
        return self._record_from_rows(row, correction_rows)

    def apply_corrections(
        self,
        user_id: str,
        attachment_id: UUID | str,
        batch: CorrectionBatch,
    ) -> AttachmentExtractionRecord:
        """Apply optimistic field corrections and append their audit records."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT extraction.* FROM attachment_extractions AS extraction
                JOIN attachments AS attachment
                  ON attachment.attachment_id = extraction.attachment_id
                WHERE extraction.document_id = ?
                  AND extraction.attachment_id = ?
                  AND extraction.user_id = ?
                  AND attachment.user_id = ?
                  AND attachment.status <> ?
                """,
                (
                    str(batch.document_id),
                    str(attachment_id),
                    user_id,
                    user_id,
                    AttachmentStatus.DELETED.value,
                ),
            ).fetchone()
            if row is None:
                raise ExtractionNotFoundError("Extraction not found.")

            document = ExtractedDocument.model_validate_json(row["document_json"])
            fields = {field.path: field for field in document.fields}
            now = datetime.now(timezone.utc)
            corrections: list[FieldCorrection] = []
            for item in batch.corrections:
                field = fields.get(item.field_path)
                if field is None:
                    raise CorrectionConflictError(
                        f"Unknown extracted field: {item.field_path}."
                    )
                if field.value != item.original_value:
                    raise CorrectionConflictError(
                        f"Field {item.field_path} changed after it was loaded."
                    )
                updated_field = field.model_copy(
                    update={
                        "value": item.corrected_value,
                        "requires_confirmation": False,
                    }
                )
                fields[item.field_path] = updated_field
                correction = FieldCorrection(
                    attachment_id=document.attachment_id,
                    document_id=document.document_id,
                    field_path=item.field_path,
                    original_value=item.original_value,
                    corrected_value=item.corrected_value,
                    note=item.note,
                    corrected_at=now,
                )
                corrections.append(correction)

            updated_fields = [fields[field.path] for field in document.fields]
            updated_document = ExtractedDocument.model_validate(
                {
                    **document.model_dump(),
                    "fields": updated_fields,
                    "status": (
                        ExtractionStatus.NEEDS_REVIEW
                        if any(field.requires_confirmation for field in updated_fields)
                        else ExtractionStatus.COMPLETED
                    ),
                }
            )
            connection.execute(
                """
                UPDATE attachment_extractions
                SET document_json = ?, updated_at = ?
                WHERE document_id = ? AND user_id = ?
                """,
                (
                    updated_document.model_dump_json(),
                    now.isoformat(),
                    str(batch.document_id),
                    user_id,
                ),
            )
            for correction in corrections:
                connection.execute(
                    """
                    INSERT INTO attachment_corrections (
                        correction_id, attachment_id, document_id, user_id,
                        correction_json, corrected_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(correction.correction_id),
                        str(correction.attachment_id),
                        str(correction.document_id),
                        user_id,
                        correction.model_dump_json(),
                        correction.corrected_at.isoformat(),
                    ),
                )
            updated_row = dict(row)
            updated_row["document_json"] = updated_document.model_dump_json()
            updated_row["updated_at"] = now.isoformat()
            prior_rows = connection.execute(
                """
                SELECT correction_json FROM attachment_corrections
                WHERE document_id = ? AND user_id = ? ORDER BY corrected_at ASC
                """,
                (str(batch.document_id), user_id),
            ).fetchall()
        return self._record_from_rows(updated_row, prior_rows)

    @staticmethod
    def _record_from_rows(row, correction_rows) -> AttachmentExtractionRecord:
        return AttachmentExtractionRecord(
            attachment_id=row["attachment_id"],
            document=ExtractedDocument.model_validate_json(row["document_json"]),
            evidence=[
                Evidence.model_validate(item)
                for item in json.loads(row["evidence_json"])
            ],
            corrections=[
                FieldCorrection.model_validate_json(item["correction_json"])
                for item in correction_rows
            ],
            extracted_at=row["extracted_at"],
            updated_at=row["updated_at"],
        )
