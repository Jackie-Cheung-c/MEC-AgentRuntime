"""Production attachment service wiring."""
#----------------生产AttachmentUploadService附件服务配置----------------
from __future__ import annotations

import os
from functools import lru_cache

from langchain.chat_models import init_chat_model

from config import settings
from src.attachments.service import AttachmentUploadService
from src.attachments.vlm_processor import ImageVLMProcessor
from src.storage.attachment_store import MinioObjectStorage, SQLiteAttachmentRepository


class VLMConfigurationError(RuntimeError):
    """Raised when the default image VLM is not configured."""

# 单例模式 ，第一次调用时创建服务实例，后续调用直接返回缓存的同一个对象     懒加载单例模式（懒汉式）
@lru_cache(maxsize=1)
def get_default_attachment_service() -> AttachmentUploadService:
    """Build the process-level MinIO attachment service on first upload."""
    object_storage = MinioObjectStorage(
        endpoint=settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        bucket=settings.MINIO_BUCKET,
        secure=settings.MINIO_SECURE,
    )
    return AttachmentUploadService(
        object_storage=object_storage,
        repository=SQLiteAttachmentRepository(),
        max_bytes=settings.MAX_IMAGE_UPLOAD_BYTES,
        max_pixels=settings.MAX_IMAGE_PIXELS,
    )

# 负责加载图片、调用 VLM、生成证据和文件的服务。它依赖于VLM和附件服务（AttachmentUploadService）。
# 在第一次调用时，它会根据环境变量配置聊天模型，并创建一个 ImageVLMProcessor 实例。之后的调用将返回缓存的同一个实例，确保性能和资源利用率。
@lru_cache(maxsize=1)
def get_default_image_vlm_processor() -> ImageVLMProcessor:
    """Build the production image processor lazily on its first image request."""
    api_key = os.getenv("VLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise VLMConfigurationError("VLM_API_KEY or OPENAI_API_KEY is required.")
    model = init_chat_model(
        model=settings.VLM_MODEL,
        model_provider="openai",
        api_key=api_key,
        base_url=settings.VLM_BASE_URL,
        disabled_params={"prompt_cache_retention": None},
    )
    attachment_service = get_default_attachment_service()
    return ImageVLMProcessor(
        model=model,
        attachment_service=attachment_service,
        extraction_repository=attachment_service.repository,
        extractor_name=settings.VLM_MODEL,
    )
