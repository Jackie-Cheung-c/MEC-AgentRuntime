from langchain_chroma import Chroma
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4
from langchain_community.embeddings import DashScopeEmbeddings
from src.paths import DATA_DIR
import os


@dataclass(frozen=True)
class ConversationTurn:
    turn_id: str
    user_id: str
    thread_id: str
    user_message: str
    assistant_message: str
    created_at: datetime


embeddings = DashScopeEmbeddings(
    model="qwen3.7-text-embedding",
    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY"),
)

memory_store = Chroma(
    collection_name="travel_memories",
    embedding_function=embeddings,
    persist_directory=str(DATA_DIR / "chroma"),
)

# 保存本轮会话记忆
def remember_conversation(
    *,
    user_id: str,
    thread_id: str,
    user_message: str,
    assistant_message: str,
    turn_id: str | None = None,
    created_at: datetime | None = None,
) -> None:
    """Persist one completed conversation turn as a searchable local memory."""
    moment = created_at or datetime.now(timezone.utc)
    resolved_turn_id = turn_id or str(uuid4())
    text = f"User request:\n{user_message}\n\nAssistant response:\n{assistant_message}"
    memory_store.add_texts(
        texts=[text],
        metadatas=[
            {
                "user_id": user_id,
                "thread_id": thread_id,
                "turn_id": resolved_turn_id,
                "user_message": user_message,
                "assistant_message": assistant_message,
                "created_at": moment.isoformat(),
                "type": "conversation",
            }
        ],
        ids=[resolved_turn_id],
    )


def _parse_legacy_document(document: str) -> tuple[str, str]:
    marker = "\n\nAssistant response:\n"
    if not document.startswith("User request:\n") or marker not in document:
        return document, ""
    user_message, assistant_message = document[len("User request:\n") :].split(
        marker,
        1,
    )
    return user_message, assistant_message


def _parse_created_at(value: object) -> datetime:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
            return (
                parsed
                if parsed.tzinfo is not None
                else parsed.replace(tzinfo=timezone.utc)
            )
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def list_recent_turns(
    user_id: str,
    thread_id: str,
    *,
    limit: int = 12,
) -> list[ConversationTurn]:
    """Load one thread's recent persisted turns in chronological order."""
    if limit < 1:
        return []
    records = memory_store.get(
        where={
            "$and": [
                {"user_id": user_id},
                {"thread_id": thread_id},
            ]
        },
        include=["documents", "metadatas"],
    )
    turns: list[ConversationTurn] = []
    for document_id, document, metadata in zip(
        records.get("ids", []),
        records.get("documents", []),
        records.get("metadatas", []),
        strict=False,
    ):
        if (
            not document
            or not metadata
            or metadata.get("user_id") != user_id
            or metadata.get("thread_id") != thread_id
        ):
            continue
        user_message = str(metadata.get("user_message") or "")
        assistant_message = str(metadata.get("assistant_message") or "")
        if not user_message and not assistant_message:
            user_message, assistant_message = _parse_legacy_document(document)
        turns.append(
            ConversationTurn(
                turn_id=str(metadata.get("turn_id") or document_id or uuid4()),
                user_id=user_id,
                thread_id=thread_id,
                user_message=user_message,
                assistant_message=assistant_message,
                created_at=_parse_created_at(metadata.get("created_at")),
            )
        )
    turns.sort(key=lambda item: item.created_at)
    return turns[-limit:]

# 删除选中的会话
def delete_thread_memory(user_id: str, thread_id: str) -> int:
    """Delete every persisted session-memory turn for one user and thread."""
    records = memory_store.get(
        where={
            "$and": [
                {"user_id": user_id},
                {"thread_id": thread_id},
            ]
        },
        include=[],
    )
    ids = [
        str(document_id)
        for document_id in records.get("ids", [])
        if document_id
    ]
    if not ids:
        return 0
    memory_store.delete(ids=ids)
    return len(ids)
