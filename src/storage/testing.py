"""Disposable storage bundle for tests and offline evaluations."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

from langgraph.checkpoint.memory import InMemorySaver

from src.domain import AgentRunRequest, AgentRunResult
from src.memory import SQLiteMemoryRepository, retrieve_long_term
from src.session_memory.context_compressor import compress_checkpoint

if TYPE_CHECKING:
    from src.agent.service import ServiceDependencies


class _NullMemoryVectorIndex:
    def upsert(self, _item) -> None:
        return None

    def delete(self, _memory_id) -> None:
        return None

    def search(self, **_kwargs) -> list[Any]:
        return []


class TemporaryTravelOpsStorage:
    """Keep checkpoint, memory, trace, and compression artifacts out of `data/`."""

    def __init__(self) -> None:
        self._directory = TemporaryDirectory(prefix="travelops-test-")
        self.root = Path(self._directory.name)
        self.checkpoint_store = InMemorySaver()
        self.conversation_path = self.root / "conversations.jsonl"
        self.trace_path = self.root / "traces.jsonl"
        self.compression_log_path = self.root / "context_compression.jsonl"
        self.long_term_memory_repository = SQLiteMemoryRepository(
            self.root / "memories.db"
        )
        self.long_term_memory_vector_index = _NullMemoryVectorIndex()
        self.long_term_memory_repository.vector_index = (
            self.long_term_memory_vector_index
        )
        self._conversations: list[dict[str, Any]] = []

    def configure(self, dependencies: "ServiceDependencies") -> "ServiceDependencies":
        """Attach every storage edge to an existing dependency bundle."""
        dependencies.checkpoint_store = self.checkpoint_store
        dependencies.memory_retriever = self.retrieve_memories
        dependencies.conversation_recorder = self.record_conversation
        dependencies.long_term_memory_repository = self.long_term_memory_repository
        dependencies.long_term_memory_vector_index = (
            self.long_term_memory_vector_index
        )
        dependencies.long_term_memory_retriever = self.retrieve_long_term_memories
        dependencies.long_term_memory_recorder = self.record_long_term_memory
        dependencies.trace_recorder = self.record_trace
        dependencies.trace_store = self
        dependencies.context_compressor = self.compress_checkpoint
        return dependencies

    def retrieve_memories(
        self,
        user_id: str,
        thread_id: str,
        query: str = "",
        limit: int = 3,
    ) -> str:
        del query

        records = [
            item
            for item in self._conversations
            if item["user_id"] == user_id
            and item["thread_id"] == thread_id
        ][-limit:]

        return "\n\n".join(
            "User request:\n"
            + item["user_message"]
            + "\n\nAssistant response:\n"
            + item["assistant_message"]
            for item in records
        )

    def retrieve_long_term_memories(
        self,
        user_id: str,
        query: str,
        trip_id=None,
    ) -> str:
        return retrieve_long_term(
            self.long_term_memory_repository,
            user_id=user_id,
            query=query,
            trip_id=trip_id,
        )

    def record_long_term_memory(
        self,
        _request: AgentRunRequest,
        _result: AgentRunResult,
    ) -> list[Any]:
        """Keep test extraction deterministic and outside the production model."""
        return []

    def record_conversation(
        self, request: AgentRunRequest, result: AgentRunResult
    ) -> None:
        record = {
            "request_id": str(request.request_id),
            "user_id": request.user_id,
            "thread_id": request.thread_id,
            "user_message": request.query,
            "assistant_message": result.answer,
            "created_at": result.completed_at.isoformat() if result.completed_at else None,
        }
        self._conversations.append(record)
        self._append_json(self.conversation_path, record)

    def record_trace(self, request: AgentRunRequest, result: AgentRunResult) -> None:
        if result.trace is None:
            return
        self._append_json(
            self.trace_path,
            {
                "request_id": str(request.request_id),
                "user_id": request.user_id,
                "thread_id": request.thread_id,
                "trace": result.trace.model_dump(mode="json"),
            },
        )

    async def compress_checkpoint(self, agent, model, config) -> None:
        await compress_checkpoint(
            agent,
            model,
            config,
            log_path=self.compression_log_path,
        )

    def close(self) -> None:
        self._directory.cleanup()

    def __enter__(self) -> "TemporaryTravelOpsStorage":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    @staticmethod
    def _append_json(path: Path, record: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
