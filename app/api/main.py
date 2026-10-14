"""FastAPI application for TravelOps."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated
from uuid import UUID

from dotenv import load_dotenv
BASE_DIR = Path(__file__).resolve().parent.parent.parent
dotenv_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=dotenv_path)   # 显式指定完整路径


from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.api.auth import (
    get_current_user,
    resolve_user_id,
    router as auth_router,
)
from config import settings
from src.agent.service import UnsupportedAttachmentError
from src.agent.travel_agent import close_default_checkpointer
from src.attachments.factory import (
    VLMConfigurationError,
    get_default_attachment_service,
    get_default_image_vlm_processor,
)
from src.attachments.service import (
    AttachmentIntegrityError,
    AttachmentNotFoundError,
    AttachmentTooLargeError,
    AttachmentUploadService,
    InvalidAttachmentError,
)
from src.attachments.vlm_processor import ImageVLMProcessingError, ImageVLMProcessor
from src.auth import AuthService, Principal
from src.domain import (
    AgentRunRequest,
    AgentRunResult,
    Attachment,
    AttachmentExtractionRecord,
    CorrectionBatch,
    ReplanProposal,
    ReplanRequest,
    ThreadSummary,
    ThreadTurn,
    TripState,
)
from src.memory import (
    ChromaMemoryVectorIndex,
    MemoryCategory,
    MemoryItem,
    MemoryNotFoundError,
    MemoryScope,
    MemoryStatus,
    SQLiteMemoryRepository,
)
from src.planning import (
    ItineraryPlanningService,
    PlanningError,
    ProposalConflictError,
    TripNotFoundError,
    TripVersionConflictError,
)
from src.storage.attachment_store import (
    CorrectionConflictError,
    ExtractionNotFoundError,
    MinioConfigurationError,
)
from src.storage.trace_store import SQLiteTraceStore
from src.storage.trip_store import SQLiteTripRepository
from src.tools.expenses_tool import (
    get_expense_summary_for_user,
    record_expenses_for_user,
)
from src.tools.hotel_tool import search_nearby_hotels
from src.tools.idempotency import payload_fingerprint
from src.tools.search_tool import recommend_attractions
from src.tools.traffic_recommand_tool import plan_route
from src.tools.weather_tool import get_weather

APP_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = APP_DIR / "web"
@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    await close_default_checkpointer()


app = FastAPI(title="TravelOps Control Room", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.state.auth_service = AuthService.with_defaults()
app.state.trip_repository = SQLiteTripRepository()
app.state.planning_service = ItineraryPlanningService(app.state.trip_repository)
app.state.attachment_service = None
app.state.image_vlm_processor = None
# 会话历史读取：与 service 内部的 trace 写入共用同一份 data/traces.db
app.state.trace_store = SQLiteTraceStore()
app.state.memory_repository = None
app.state.memory_vector_index = None
app.include_router(auth_router)


class DestinationRequest(BaseModel):
    destination: str = Field(min_length=1, max_length=100)


class RouteRequest(BaseModel):
    origin: str = Field(min_length=1, max_length=100)
    destination: str = Field(min_length=1, max_length=100)


class ExpenseRequest(BaseModel):
    expense_text: str = Field(min_length=1, max_length=500)


class TripOwnerRequest(BaseModel):
    user_id: str = Field(default="local_user", min_length=1, max_length=128)


@app.get("/")
async def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "workspace.html")


async def _call_tool(tool, payload: dict) -> dict[str, str]:
    try:
        result = await asyncio.to_thread(tool.invoke, payload)
        return {"result": str(result)}
    except Exception as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@app.post("/api/weather")
async def weather(
    request: DestinationRequest,
    principal: CurrentUser,
) -> dict[str, str]:
    return await _call_tool(get_weather, {"city": request.destination})


@app.post("/api/attractions")
async def attractions(
    request: DestinationRequest,
    principal: CurrentUser,
) -> dict[str, str]:
    return await _call_tool(recommend_attractions, {"destination": request.destination})


@app.post("/api/hotels")
async def hotels(
    request: DestinationRequest,
    principal: CurrentUser,
) -> dict[str, str]:
    return await _call_tool(search_nearby_hotels, {"attraction": request.destination})


@app.post("/api/route")
async def route(
    request: RouteRequest,
    principal: CurrentUser,
) -> dict[str, str]:
    return await _call_tool(plan_route, request.model_dump())


@app.post("/api/expenses")
async def expenses(
    request: ExpenseRequest,
    principal: CurrentUser,
    idempotency_key: Annotated[
        str | None,
        Header(alias="Idempotency-Key", min_length=1, max_length=255),
    ] = None,
) -> dict[str, str]:
    effective_user_id = resolve_user_id(principal)
    scoped_key = (
        f"record_expenses:{effective_user_id}:{idempotency_key}:"
        + payload_fingerprint({"expense_text": request.expense_text.strip()})
        if idempotency_key
        else None
    )
    result = await asyncio.to_thread(
        record_expenses_for_user,
        request.expense_text,
        effective_user_id,
        idempotency_key=scoped_key,
    )
    return {"result": str(result)}


@app.get("/api/expenses")
async def expense_summary(principal: CurrentUser) -> dict[str, str]:
    result = await asyncio.to_thread(
        get_expense_summary_for_user,
        resolve_user_id(principal),
    )
    return {"result": str(result)}


def get_trace_store(request: Request) -> SQLiteTraceStore:
    """Resolve the conversation-history store; tests may replace it on app.state."""
    return request.app.state.trace_store


def get_attachment_service(request: Request) -> AttachmentUploadService:
    """Resolve an override or lazily build the production MinIO service."""
    if request.app.state.attachment_service is not None:
        return request.app.state.attachment_service
    try:
        return get_default_attachment_service()
    except MinioConfigurationError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


def get_image_vlm_processor(request: Request) -> ImageVLMProcessor:
    if request.app.state.image_vlm_processor is not None:
        return request.app.state.image_vlm_processor
    try:
        return get_default_image_vlm_processor()
    except VLMConfigurationError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


def get_planning_service(request: Request) -> ItineraryPlanningService:
    return request.app.state.planning_service


CurrentUser = Annotated[Principal, Depends(get_current_user)]


def ensure_memory_backend(current_app: FastAPI) -> SQLiteMemoryRepository:
    """Create one shared SQLite + Chroma memory backend per application."""
    if current_app.state.memory_repository is not None:
        return current_app.state.memory_repository
    vector_index = ChromaMemoryVectorIndex()
    repository = SQLiteMemoryRepository(vector_index=vector_index)
    current_app.state.memory_vector_index = vector_index
    current_app.state.memory_repository = repository
    return repository


def get_memory_repository(
    request: Request,
) -> SQLiteMemoryRepository:
    return ensure_memory_backend(request.app)


def _sse(event: str, payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {data}\n\n"


def _agent_result_payload(result: Any) -> dict:
    if hasattr(result, "model_dump"):
        return result.model_dump(
            mode="json",
            include={
                "thread_id",
                "status",
                "answer",
                "error",
                "trip_state",
            },
            exclude_none=True,
        )
    return {
        key: result.get(key)
        for key in (
            "thread_id",
            "status",
            "answer",
            "error",
            "trip_state",
        )
        if result.get(key) is not None
    }


async def _enqueue_agent_task(
    store: SQLiteTraceStore,
    request: AgentRunRequest,
) -> dict:
    try:
        return await asyncio.to_thread(
            store.enqueue_run,
            request,
            max_attempts=settings.AGENT_TASK_MAX_ATTEMPTS,
        )
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to enqueue Agent task: {error}",
        ) from error


async def _wait_for_agent_result(
    store: SQLiteTraceStore,
    *,
    user_id: str,
    request_id: UUID,
    timeout_seconds: float,
) -> dict:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(1.0, timeout_seconds)
    while True:
        result = await asyncio.to_thread(
            store.get_result_by_request_id,
            user_id=user_id,
            request_id=request_id,
        )
        if result is not None:
            return result
        task = await asyncio.to_thread(store.get_task, request_id)
        if task is not None and task["user_id"] != user_id:
            raise HTTPException(status_code=404, detail="Agent task not found.")
        if task is not None and task["status"] == "failed":
            raise HTTPException(
                status_code=500,
                detail=task.get("last_error") or "Agent task failed.",
            )
        if loop.time() >= deadline:
            raise HTTPException(
                status_code=504,
                detail="Agent task is still running.",
            )
        await asyncio.sleep(
            max(0.05, settings.AGENT_TASK_POLL_INTERVAL_SECONDS)
        )


@app.get("/api/trips/{trip_id}", response_model=TripState)
async def get_trip(
    trip_id: UUID,
    principal: CurrentUser,
    planner: Annotated[ItineraryPlanningService, Depends(get_planning_service)],
    user_id: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
) -> TripState:
    effective_user_id = resolve_user_id(principal, user_id)
    try:
        return await asyncio.to_thread(
            planner.repository.get,
            effective_user_id,
            trip_id,
        )
    except TripNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post("/api/trips/{trip_id}/replan", response_model=ReplanProposal)
async def propose_trip_replan(
    trip_id: UUID,
    request: ReplanRequest,
    principal: CurrentUser,
    planner: Annotated[ItineraryPlanningService, Depends(get_planning_service)],
    idempotency_key: Annotated[
        str | None,
        Header(alias="Idempotency-Key", min_length=1, max_length=255),
    ] = None,
) -> ReplanProposal:
    secured_request = request.model_copy(
        update={"user_id": resolve_user_id(principal, request.user_id)}
    )
    scoped_key = (
        f"propose_trip_replan:{secured_request.user_id}:{idempotency_key}"
        if idempotency_key
        else None
    )
    try:
        return await asyncio.to_thread(
            planner.propose_replan,
            trip_id,
            secured_request,
            idempotency_key=scoped_key,
        )
    except TripNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except TripVersionConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except PlanningError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post(
    "/api/trips/{trip_id}/replan/{proposal_id}/confirm",
    response_model=TripState,
)
async def confirm_trip_replan(
    trip_id: UUID,
    proposal_id: UUID,
    request: TripOwnerRequest,
    principal: CurrentUser,
    planner: Annotated[ItineraryPlanningService, Depends(get_planning_service)],
    idempotency_key: Annotated[
        str | None,
        Header(alias="Idempotency-Key", min_length=1, max_length=255),
    ] = None,
) -> TripState:
    effective_user_id = resolve_user_id(principal, request.user_id)
    scoped_key = (
        f"confirm_trip_replan:{effective_user_id}:{idempotency_key}:"
        + payload_fingerprint(
            {"trip_id": str(trip_id), "proposal_id": str(proposal_id)}
        )
        if idempotency_key
        else None
    )
    try:
        return await asyncio.to_thread(
            planner.confirm_replan,
            effective_user_id,
            trip_id,
            proposal_id,
            idempotency_key=scoped_key,
        )
    except TripNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (TripVersionConflictError, ProposalConflictError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error

# 上传附件的API端点
@app.post(
    "/api/attachments",
    response_model=Attachment,
    status_code=status.HTTP_201_CREATED,
)
async def upload_attachment(
    file: Annotated[UploadFile, File(description="JPEG, PNG, or WebP image")],
    service: Annotated[AttachmentUploadService, Depends(get_attachment_service)],
    principal: CurrentUser,
    user_id: Annotated[str, Form(min_length=1, max_length=128)] = "local_user",
) -> Attachment:
    """Validate and store one private image attachment in MinIO."""
    effective_user_id = resolve_user_id(principal, user_id)
    try:
        return await asyncio.to_thread(
            service.upload_image,
            user_id=effective_user_id,
            filename=file.filename or "upload",
            declared_media_type=file.content_type,
            data=file.file,
        )
    except AttachmentTooLargeError as error:
        raise HTTPException(status_code=413, detail=str(error)) from error
    except InvalidAttachmentError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except MinioConfigurationError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=502, detail=f"Attachment storage failed: {error}"
        ) from error
    finally:
        await file.close()

# 分页列出指定用户的所有附件元数据
@app.get("/api/attachments", response_model=list[Attachment])
async def list_attachments(
    principal: CurrentUser,
    service: Annotated[AttachmentUploadService, Depends(get_attachment_service)],
    user_id: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 30,
) -> list[Attachment]:
    effective_user_id = resolve_user_id(principal, user_id)
    return await asyncio.to_thread(
        service.list_for_user, user_id=effective_user_id, limit=limit
    )

# 拿到某个附件的元数据，前提是请求的用户拥有该附件
@app.get("/api/attachments/{attachment_id}", response_model=Attachment)
async def get_attachment(
    attachment_id: UUID,
    principal: CurrentUser,
    service: Annotated[AttachmentUploadService, Depends(get_attachment_service)],
    user_id: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
) -> Attachment:
    """Return metadata only when the requesting user owns the attachment."""
    effective_user_id = resolve_user_id(principal, user_id)
    try:
        return await asyncio.to_thread(
            service.get,
            user_id=effective_user_id,
            attachment_id=attachment_id,
        )
    except AttachmentNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=502, detail=f"Attachment storage failed: {error}"
        ) from error

# 获得某个附件的二进制内容，前提是请求的用户拥有该附件
@app.get("/api/attachments/{attachment_id}/content")
async def get_attachment_content(
    attachment_id: UUID,
    principal: CurrentUser,
    service: Annotated[AttachmentUploadService, Depends(get_attachment_service)],
    user_id: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
) -> Response:
    """Stream private image bytes through the ownership-checked API."""
    effective_user_id = resolve_user_id(principal, user_id)
    try:
        attachment, content = await asyncio.to_thread(
            service.read,
            user_id=effective_user_id,
            attachment_id=attachment_id,
        )
        return Response(
            content=content,
            media_type=attachment.media_type,
            headers={"Cache-Control": "private, max-age=300"},
        )
    except AttachmentNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except AttachmentIntegrityError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=502, detail=f"Attachment storage failed: {error}"
        ) from error

# 信息提取
@app.post(
    "/api/attachments/{attachment_id}/extract",
    response_model=AttachmentExtractionRecord,
)
async def extract_attachment(
    attachment_id: UUID,
    principal: CurrentUser,
    service: Annotated[AttachmentUploadService, Depends(get_attachment_service)],
    processor: Annotated[ImageVLMProcessor, Depends(get_image_vlm_processor)],
    user_id: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
    force: bool = Query(
        default=False,
        description="是否强制重新执行 VLM 提取",
    ),
) -> AttachmentExtractionRecord:
    """
    提取图片信息。

    默认行为：
    - 已经有解析记录：直接返回数据库记录
    - 没有解析记录：调用 VLM

    force=true：
    - 忽略已有解析记录
    - 强制重新执行 VLM
    """
    effective_user_id = resolve_user_id(principal, user_id)
    try:
        # 第一步：验证附件归属
        attachment = await asyncio.to_thread(
            service.get,
            user_id=effective_user_id,
            attachment_id=attachment_id,
        )
        # 第二步：默认优先复用已有解析结果
        if not force:
            existing  = await asyncio.to_thread(
                service.repository.get_latest_extraction,
                effective_user_id,
                attachment_id,
            )
            if existing  is not None:
                return existing 
        # 第三步：没有解析记录，或者 force=true，才调用 VLM
        documents, evidence = await processor.process(
            [attachment],
            user_id=effective_user_id,
            force=force,
        )
        # 优先读取数据库中已有的解析记录，避免重复写入
        # 兼容某些只返回结果、但没有配置 extraction_repository 的处理器
        if not documents:
            raise HTTPException(
                status_code=502,
                detail="Image processor returned no extracted document.",
            )
        record = await asyncio.to_thread(
            service.repository.get_latest_extraction,
            effective_user_id,
            attachment_id,
        )
        return record or AttachmentExtractionRecord(
            attachment_id=attachment_id,
            document=documents[0],
            evidence=evidence,
        )
    except AttachmentNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (ImageVLMProcessingError, VLMConfigurationError) as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

# 拿到提取的信息
@app.get(
    "/api/attachments/{attachment_id}/extraction",
    response_model=AttachmentExtractionRecord,
)
async def get_attachment_extraction(
    attachment_id: UUID,
    principal: CurrentUser,
    service: Annotated[AttachmentUploadService, Depends(get_attachment_service)],
    user_id: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
) -> AttachmentExtractionRecord:
    effective_user_id = resolve_user_id(principal, user_id)
    try:
        await asyncio.to_thread(
            service.get,
            user_id=effective_user_id,
            attachment_id=attachment_id,
        )
        record = await asyncio.to_thread(
            service.repository.get_latest_extraction,
            effective_user_id,
            attachment_id,
        )
        if record is None:
            raise HTTPException(status_code=404, detail="Extraction not found.")
        return record
    except AttachmentNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

# 附件的字段修正
@app.post(
    "/api/attachments/{attachment_id}/corrections",
    response_model=AttachmentExtractionRecord,
)
async def correct_attachment_fields(
    attachment_id: UUID,
    batch: CorrectionBatch,
    principal: CurrentUser,
    service: Annotated[AttachmentUploadService, Depends(get_attachment_service)],
    user_id: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
) -> AttachmentExtractionRecord:
    """Persist human field corrections with an append-only audit trail."""
    effective_user_id = resolve_user_id(principal, user_id)
    try:
        return await asyncio.to_thread(
            service.repository.apply_corrections,
            effective_user_id,
            attachment_id,
            batch,
        )
    except ExtractionNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except CorrectionConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error

# ---------- 会话历史 ----------
@app.get("/api/threads", response_model=list[ThreadSummary])
async def list_threads(
    principal: CurrentUser,
    store: Annotated[SQLiteTraceStore, Depends(get_trace_store)],
    user_id: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[ThreadSummary]:
    """List one user's conversation threads for the sidebar history."""
    effective_user_id = resolve_user_id(principal, user_id)
    rows = await asyncio.to_thread(
        store.list_threads,
        user_id=effective_user_id,
        limit=limit,
    )
    return [ThreadSummary.model_validate(row) for row in rows]

# 查看对话
@app.get("/api/threads/{thread_id}/turns", response_model=list[ThreadTurn])
async def list_thread_turns(
    thread_id: str,
    principal: CurrentUser,
    store: Annotated[SQLiteTraceStore, Depends(get_trace_store)],
    user_id: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
) -> list[ThreadTurn]:
    """Replay one conversation so the client can restore its transcript."""
    effective_user_id = resolve_user_id(principal, user_id)
    rows = await asyncio.to_thread(
        store.list_thread_turns,
        user_id=effective_user_id,
        thread_id=thread_id,
        limit=limit,
    )
    return [ThreadTurn.model_validate(row) for row in rows]


@app.delete("/api/threads/{thread_id}")
async def delete_thread(
    thread_id: str,
    principal: CurrentUser,
    store: Annotated[SQLiteTraceStore, Depends(get_trace_store)],
    user_id: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
) -> dict[str, int]:
    """Delete one conversation, its spans, and its persisted session memory."""
    from src.session_memory.memory import delete_thread_memory

    effective_user_id = resolve_user_id(principal, user_id)
    memory_deleted = await asyncio.to_thread(
        delete_thread_memory,
        effective_user_id,
        thread_id,
    )
    deleted = await asyncio.to_thread(
        store.delete_thread,
        user_id=effective_user_id,
        thread_id=thread_id,
    )
    return {
        "deleted": deleted,
        "session_memory_deleted": memory_deleted,
    }


# ---------- 长期记忆管理 ----------
@app.get("/api/memories", response_model=list[MemoryItem])
async def list_memories(
    principal: CurrentUser,
    repository: Annotated[
        SQLiteMemoryRepository,
        Depends(get_memory_repository),
    ],
    user_id: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
    status: Annotated[
        str,
        Query(pattern=r"^(active|pending|superseded|deleted|rejected|all)$"),
    ] = "active",
    category: MemoryCategory | None = None,
    scope: MemoryScope | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[MemoryItem]:
    effective_user_id = resolve_user_id(principal, user_id)
    return await asyncio.to_thread(
        repository.list_items,
        effective_user_id,
        status=None if status == "all" else MemoryStatus(status),
        category=category,
        scope=scope,
        limit=limit,
        offset=offset,
    )


@app.get("/api/memories/export")
async def export_memories(
    principal: CurrentUser,
    repository: Annotated[
        SQLiteMemoryRepository,
        Depends(get_memory_repository),
    ],
    user_id: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
) -> Response:
    effective_user_id = resolve_user_id(principal, user_id)
    items = await asyncio.to_thread(
        repository.list_items,
        effective_user_id,
        status=MemoryStatus.ACTIVE,
        limit=10000,
    )
    payload = {
        "user_id": effective_user_id,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "memories": [item.model_dump(mode="json") for item in items],
    }
    filename = (
        f"memories-{effective_user_id}-"
        f"{datetime.now(timezone.utc):%Y%m%d}.json"
    )
    return Response(
        content=json.dumps(payload, ensure_ascii=False, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.delete("/api/memories/{memory_id}", response_model=MemoryItem)
async def delete_memory(
    memory_id: UUID,
    principal: CurrentUser,
    repository: Annotated[
        SQLiteMemoryRepository,
        Depends(get_memory_repository),
    ],
    user_id: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
) -> MemoryItem:
    effective_user_id = resolve_user_id(principal, user_id)
    try:
        return await asyncio.to_thread(
            repository.forget,
            effective_user_id,
            memory_id,
        )
    except MemoryNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


# 流式输出Agent执行的API端点
@app.post("/api/agent/stream")
async def assistant_stream(
    request: AgentRunRequest,
    principal: CurrentUser,
) -> StreamingResponse:
    """Queue work and stream durable deltas, not Web-process memory."""
    secured_request = request.model_copy(
        update={"user_id": resolve_user_id(principal, request.user_id)}
    )
    store: SQLiteTraceStore = app.state.trace_store
    stored_result = await asyncio.to_thread(
        store.get_result_by_request_id,
        user_id=secured_request.user_id,
        request_id=secured_request.request_id,
    )
    if stored_result is not None:
        async def completed_stream():
            yield _sse("done", {"result": _agent_result_payload(stored_result)})

        return StreamingResponse(
            completed_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    task = await _enqueue_agent_task(store, secured_request)
    if task.get("user_id") != secured_request.user_id:
        raise HTTPException(status_code=404, detail="Agent task not found.")

    async def event_stream():
        loop = asyncio.get_running_loop()
        stream_started = loop.time()
        last_heartbeat = loop.time()
        after_event_id = 0
        try:
            while True:
                events = await asyncio.to_thread(
                    store.list_events,
                    secured_request.request_id,
                    after_id=after_event_id,
                )
                for event in events:
                    after_event_id = event["event_id"]
                    if event["event_type"] == "delta":
                        yield _sse("delta", event["payload"])
                    elif event["event_type"] == "reset":
                        yield _sse("reset", event["payload"])
                    elif event["event_type"] == "done":
                        result = await asyncio.to_thread(
                            store.get_result_by_request_id,
                            user_id=secured_request.user_id,
                            request_id=secured_request.request_id,
                        )
                        if result is None:
                            yield _sse(
                                "error",
                                {"error": "Agent task completed without a persisted result."},
                            )
                        else:
                            yield _sse(
                                "done",
                                {"result": _agent_result_payload(result)},
                            )
                        return
                    elif event["event_type"] == "error":
                        yield _sse(
                            "error",
                            {
                                "error": event["payload"].get(
                                    "error", "Agent task failed."
                                )
                            },
                        )
                        return
                if events:
                    continue
                current_task = await asyncio.to_thread(
                    store.get_task,
                    secured_request.request_id,
                )
                if current_task is not None and current_task["status"] in {
                    "succeeded",
                    "failed",
                }:
                    result = await asyncio.to_thread(
                        store.get_result_by_request_id,
                        user_id=secured_request.user_id,
                        request_id=secured_request.request_id,
                    )
                    if result is not None:
                        yield _sse(
                            "done",
                            {"result": _agent_result_payload(result)},
                        )
                    elif current_task["status"] == "succeeded":
                        yield _sse(
                            "error",
                            {"error": "Agent task completed without a persisted result."},
                        )
                    else:
                        yield _sse(
                            "error",
                            {
                                "error": current_task.get("last_error")
                                or "Agent task failed."
                            },
                        )
                    return
                await asyncio.sleep(
                    max(0.05, settings.AGENT_TASK_EVENT_POLL_INTERVAL_SECONDS)
                )
                if (
                    loop.time() - stream_started
                    >= settings.AGENT_TASK_RESULT_WAIT_TIMEOUT_SECONDS
                ):
                    yield _sse(
                        "error",
                        {"error": "Agent task is still running."},
                    )
                    return
                now = loop.time()
                if now - last_heartbeat >= 15:
                    last_heartbeat = now
                    yield ": keep-alive\n\n"
        except asyncio.CancelledError:
            raise
        except UnsupportedAttachmentError as error:
            yield _sse("error", {"error": str(error)})
        except Exception as error:
            yield _sse("error", {"error": str(error)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/agent", response_model=AgentRunResult, response_model_exclude_none=True)
async def assistant(
    request: AgentRunRequest,
    principal: CurrentUser,
) -> AgentRunResult:
    """Queue the request and wait for the durable worker's persisted result."""
    secured_request = request.model_copy(
        update={"user_id": resolve_user_id(principal, request.user_id)}
    )
    store: SQLiteTraceStore = app.state.trace_store
    task = await _enqueue_agent_task(store, secured_request)
    if task.get("user_id") != secured_request.user_id:
        raise HTTPException(status_code=404, detail="Agent task not found.")
    result = await _wait_for_agent_result(
        store,
        user_id=secured_request.user_id,
        request_id=secured_request.request_id,
        timeout_seconds=settings.AGENT_TASK_RESULT_WAIT_TIMEOUT_SECONDS,
    )
    return AgentRunResult.model_validate(result)
