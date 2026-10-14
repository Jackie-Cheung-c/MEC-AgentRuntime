"""Validated image upload workflow for TravelOps attachments."""

# ------------负责处理用户上传的图片文件，生成标准的 Attachment 元数据对象--------------
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import BinaryIO, Protocol
from uuid import UUID, uuid4

from src.domain import Attachment, AttachmentKind, AttachmentSource, AttachmentStatus
from src.storage.attachment_store import ObjectStorage


class AttachmentRepository(Protocol):
    def save(self, user_id: str, attachment: Attachment) -> None: ...

    def get(
        self,
        user_id: str,
        attachment_id: UUID | str,
        *,
        include_deleted: bool = False,
    ) -> Attachment | None: ...
    # 哈希路径存储图片名称
    def get_by_sha256(
        self,
        user_id: str,
        sha256: str,
        *,
        include_deleted: bool = False,
    ) -> Attachment | None: ...

    def update_status(
        self,
        user_id: str,
        attachment: Attachment,
        *,
        expected_status: AttachmentStatus,
    ) -> None: ...

    def list_for_user(self, user_id: str, limit: int = 50) -> list[Attachment]: ...


class InvalidAttachmentError(ValueError):
    """Raised when uploaded bytes are not a supported, valid image."""


class AttachmentTooLargeError(InvalidAttachmentError):
    """Raised when an image exceeds the configured upload limit."""


class AttachmentNotFoundError(LookupError):
    """Raised when an attachment is absent or belongs to another user."""


class AttachmentIntegrityError(RuntimeError):
    """Raised when stored bytes no longer match persisted metadata."""


class InvalidAttachmentStatusTransition(ValueError):
    """Raised when an attachment lifecycle transition is not allowed."""


class AttachmentUploadService:
    """Validate, store, and register image attachments as one workflow."""

    _FORMAT_MEDIA_TYPES = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
        "JPG": "image/jpg",
    } 
    _FORMAT_EXTENSIONS = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp", "JPG": "jpg"}
    _STATUS_TRANSITIONS = {
        AttachmentStatus.AVAILABLE: {
            AttachmentStatus.PROCESSING,
            AttachmentStatus.DELETED,
        },
        AttachmentStatus.PROCESSING: {
            AttachmentStatus.PROCESSED,
            AttachmentStatus.FAILED,
            AttachmentStatus.DELETED,
        },
        AttachmentStatus.PROCESSED: {
            AttachmentStatus.PROCESSING,
            AttachmentStatus.DELETED,
        },
        AttachmentStatus.FAILED: {
            AttachmentStatus.PROCESSING,
            AttachmentStatus.DELETED,
        },
        AttachmentStatus.DELETED: set(),
    }

    def __init__(
        self,
        *,
        object_storage: ObjectStorage,
        repository: AttachmentRepository,
        max_bytes: int = 10 * 1024 * 1024,
        max_pixels: int = 50_000_000,
        chunk_size: int = 1024 * 1024,
    ) -> None:
        self.object_storage = object_storage
        self.repository = repository
        self.max_bytes = max_bytes
        self.max_pixels = max_pixels
        self.chunk_size = chunk_size

    def upload_image(
        self,
        *,
        user_id: str,
        filename: str,
        declared_media_type: str | None,
        data: BinaryIO,
        source: AttachmentSource = AttachmentSource.UPLOAD,
    ) -> Attachment:
        user_id = self._normalize_user_id(user_id)
        safe_filename = self._safe_filename(filename)
        size_bytes, sha256 = self._measure(data)
        image_format, width, height = self._inspect_image(data)
        media_type = self._FORMAT_MEDIA_TYPES[image_format]
        declared = (declared_media_type or "").lower().strip()
        if declared not in {"", "application/octet-stream", media_type}:
            raise InvalidAttachmentError(
                f"Declared media type {declared!r} does not match {media_type!r}."
            )
        # 幂等检查：同一用户上传相同内容时，直接返回已有附件，软删除的附件也会被忽略
        existing = self.repository.get_by_sha256(user_id, sha256, include_deleted=False)
        if existing:
            print(f"当前附件已上传过，直接返回已有附件: {existing.attachment_id}")
            return existing
        # 生成唯一的Attachment ID
        attachment_id = uuid4()
        # 通过对 user_id 进行哈希处理，生成一个安全的用户范围目录，避免泄露原始 user_id
        user_scope = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:16]
        extension = self._FORMAT_EXTENSIONS[image_format]
        # 对象名称使用内容哈希，而不是 UUID，同一用户上传相同文件时，MinIO object key 相同
        object_name = str(
            PurePosixPath("users", user_scope, "attachments", f"{sha256}.{extension}")
        )
        storage_uri = self.object_storage.put_object(
            object_name=object_name,
            data=data,
            length=size_bytes,
            content_type=media_type,
            metadata={"attachment-id": str(attachment_id), "sha256": sha256},
        )
        # 自动回滚，如果保存 Attachment 元数据失败，则删除已上传的对象
        try:
            attachment = Attachment(
                attachment_id=attachment_id,
                kind=AttachmentKind.IMAGE,
                source=source,
                filename=safe_filename,
                media_type=media_type,
                size_bytes=size_bytes,
                storage_uri=storage_uri,
                sha256=sha256,
                width=width,
                height=height,
            )
            self.repository.save(user_id, attachment)
        except Exception:
            #  处理并发重复上传,另一个请求可能已经先完成数据库写入
            existing = self.repository.get_by_sha256(user_id, sha256, include_deleted=False)
            if existing:
                # 不删除，可能另一个请求已经成功保存了 Attachment 元数据，MinIO 对象也已经存在
                return existing
            # 如果不是，那就是数据库保存失败，回滚
            self.object_storage.remove_object(object_name)
            raise
        return attachment

    def get(self, *, user_id: str, attachment_id: UUID | str) -> Attachment:
        user_id = self._normalize_user_id(user_id)
        attachment = self.repository.get(user_id, attachment_id)
        if attachment is None:
            raise AttachmentNotFoundError("Attachment not found.")
        return attachment

    def list_for_user(self, *, user_id: str, limit: int = 50) -> list[Attachment]:
        user_id = self._normalize_user_id(user_id)
        return self.repository.list_for_user(user_id, limit=limit)

    def update_status(
        self,
        *,
        user_id: str,
        attachment_id: UUID | str,
        status: AttachmentStatus,
        reason: str | None = None,
    ) -> Attachment:
        """Move an owned attachment through its processing lifecycle."""
        user_id = self._normalize_user_id(user_id)
        current = self.repository.get(
            user_id, attachment_id, include_deleted=True
        )
        if current is None:
            raise AttachmentNotFoundError("Attachment not found.")
        if status == current.status:
            return current
        if status not in self._STATUS_TRANSITIONS[current.status]:
            raise InvalidAttachmentStatusTransition(
                f"Cannot transition attachment from {current.status.value} "
                f"to {status.value}."
            )
        if status == AttachmentStatus.FAILED and not (reason or "").strip():
            raise InvalidAttachmentStatusTransition(
                "A failure reason is required when marking an attachment failed."
            )
        updated = Attachment.model_validate(
            {
                **current.model_dump(),
                "status": status,
                "status_reason": reason.strip()
                if status == AttachmentStatus.FAILED and reason
                else None,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        self.repository.update_status(
            user_id,
            updated,
            expected_status=current.status,
        )
        return updated

    def read(
        self, *, user_id: str, attachment_id: UUID | str
    ) -> tuple[Attachment, bytes]:
        attachment = self.get(user_id=user_id, attachment_id=attachment_id)
        prefix = f"minio://{self.object_storage.bucket}/"
        if not attachment.storage_uri.startswith(prefix):
            raise InvalidAttachmentError("Attachment storage reference is invalid.")
        object_name = attachment.storage_uri[len(prefix) :]
        if not object_name:
            raise InvalidAttachmentError("Attachment object name is missing.")
        content = self.object_storage.get_object(object_name)
        if len(content) != attachment.size_bytes:
            raise AttachmentIntegrityError(
                "Stored attachment size does not match its metadata."
            )
        if not attachment.sha256:
            raise AttachmentIntegrityError("Attachment SHA-256 metadata is missing.")
        actual_sha256 = hashlib.sha256(content).hexdigest()
        if actual_sha256 != attachment.sha256.lower():
            raise AttachmentIntegrityError(
                "Stored attachment SHA-256 does not match its metadata."
            )
        return attachment, content

    @staticmethod
    def _normalize_user_id(user_id: str) -> str:
        normalized = user_id.strip()
        if not normalized or len(normalized) > 128:
            raise InvalidAttachmentError("user_id must contain 1 to 128 characters.")
        return normalized

    # 去掉路径、控制字符，防止注入攻击
    @staticmethod
    def _safe_filename(filename: str) -> str:
        name = filename.replace("\\", "/").rsplit("/", 1)[-1].strip()
        if not name or name in {".", ".."}:
            raise InvalidAttachmentError("A valid filename is required.")
        if len(name) > 255:
            raise InvalidAttachmentError("Filename exceeds 255 characters.")
        if any(ord(character) < 32 for character in name):
            raise InvalidAttachmentError("Filename contains control characters.")
        return name

    # 计算 SHA256 哈希，校验文件大小限制（默认 10MB）
    def _measure(self, data: BinaryIO) -> tuple[int, str]:
        data.seek(0)
        digest = hashlib.sha256()
        total = 0
        while chunk := data.read(self.chunk_size):
            total += len(chunk)
            if total > self.max_bytes:
                raise AttachmentTooLargeError(
                    f"Image exceeds the {self.max_bytes} byte upload limit."
                )
            digest.update(chunk)
        data.seek(0)
        if total == 0:
            raise InvalidAttachmentError("Uploaded image is empty.")
        return total, digest.hexdigest()

    # 使用 Pillow 库解码图片，限制尺寸（默认 5000 万像素）
    def _inspect_image(self, data: BinaryIO) -> tuple[str, int, int]:
        import warnings

        from PIL import Image, UnidentifiedImageError

        try:
            data.seek(0)
            with warnings.catch_warnings():
                # 防止图像炸弹攻击，Pillow 会在解码大图时发出警告，这里将其视为错误
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(data) as image:
                    image_format = str(image.format or "").upper()
                    width, height = image.size
                    image.verify()
            if image_format not in self._FORMAT_MEDIA_TYPES:
                raise InvalidAttachmentError(
                    "Only JPEG, PNG, JPG and WebP images are supported."
                )
            if width * height > self.max_pixels:
                raise InvalidAttachmentError("Image dimensions exceed the pixel limit.")
            return image_format, width, height
        except (
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
            UnidentifiedImageError,
            OSError,
            SyntaxError,
        ) as error:
            raise InvalidAttachmentError("Uploaded bytes are not a valid image.") from error
        finally:
            data.seek(0)
