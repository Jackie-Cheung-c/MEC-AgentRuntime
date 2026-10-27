"""长期记忆向量索引做召回"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

from src.memory.models import (
    MemoryContext,
    MemoryItem,
    MemoryScope,
    MemoryStatus,
)
from src.paths import DATA_DIR


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemoryVectorHit:
    memory_id: str
    score: float


def _memory_text(item: MemoryItem) -> str:
    return "\n".join(
        (
            f"category: {item.category.value}",
            f"key: {item.memory_key}",
            f"value: {item.value}",
            f"domain: {item.domain}",
            f"workflow: {item.workflow_id or ''}",
            f"scope: {item.scope.value}",
            f"scope_id: {item.scope_id or ''}",
            f"evidence: {item.evidence_quote or ''}",
        )
    )


class ChromaMemoryVectorIndex:
    """Mirror active SQLite memory rows into a dedicated Chroma collection."""

    def __init__(
        self,
        persist_directory: str | Path = DATA_DIR / "chroma",
        collection_name: str = "long_term_memories",
    ) -> None:
        embeddings = DashScopeEmbeddings(
            model="qwen3.7-text-embedding",
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        )
        self.store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(persist_directory),
        )
    # 只要不是active就删 存储记忆项到chroma中
    def upsert(self, item: MemoryItem) -> None:
        if item.status != MemoryStatus.ACTIVE:
            self.delete(item.memory_id)
            return
        self.store.delete(ids=[str(item.memory_id)])
        self.store.add_texts(
            texts=[_memory_text(item)],
            metadatas=[
                {
                    "memory_id": str(item.memory_id),
                    "user_id": item.user_id,
                    "category": item.category.value,
                    "domain": item.domain,
                    "workflow_id": item.workflow_id or "",
                    "scope": item.scope.value,
                    "scope_id": item.scope_id or "",
                    "thread_id": item.thread_id or "",
                    "status": item.status.value,
                }
            ],
            ids=[str(item.memory_id)],
        )
    # 从chroma中删除记忆项
    def delete(self, memory_id: UUID | str) -> None:
        try:
            self.store.delete(ids=[str(memory_id)])
        except Exception as error:
            logger.warning("Failed to delete vector memory %s: %s", memory_id, error)

    def count(self) -> int:
        try:
            return int(self.store._collection.count())
        except Exception:
            return 0

    def search(
        self,
        *,
        user_id: str,
        query: str,
        context: MemoryContext,
        limit: int,
    ) -> list[MemoryVectorHit]:
        if not query.strip() or limit <= 0:
            return []

        filters: list[dict[str, Any]] = [
            {"user_id": user_id},
            {"status": MemoryStatus.ACTIVE.value},
        ]
        scope_filters: list[dict[str, Any]] = [
            {"scope": MemoryScope.GLOBAL.value}
        ]
        if context.scope_id is not None:
            scope_filters.append(
                {
                    "$and": [
                        {"scope": MemoryScope.CONTEXT.value},
                        {"scope_id": context.scope_id},
                    ]
                }
            )
        if context.thread_id is not None:
            scope_filters.append(
                {
                    "$and": [
                        {"scope": MemoryScope.THREAD.value},
                        {"thread_id": context.thread_id},
                    ]
                }
            )
        filters.append({"$or": scope_filters})
        filters.append(
            {
                "$or": [
                    {"domain": "general"},
                    {
                        "$and": [
                            {"domain": context.domain},
                            {
                                "$or": [
                                    {"scope": MemoryScope.GLOBAL.value},
                                    {"workflow_id": ""},
                                    {
                                        "workflow_id": (
                                            context.workflow_id
                                            or "__no_workflow__"
                                        )
                                    },
                                ]
                            },
                        ]
                    },
                ]
            }
        )

        try:
            results = self.store.similarity_search_with_score(
                query,
                k=limit,
                filter={"$and": filters},
            )
        except Exception as error:
            logger.warning("Long-term vector retrieval failed: %s", error)
            return []

        hits: list[MemoryVectorHit] = []
        for document, distance in results:
            memory_id = str((document.metadata or {}).get("memory_id") or "")
            if memory_id:
                hits.append(
                    MemoryVectorHit(
                        memory_id=memory_id,
                        score=1.0 / (1.0 + max(float(distance), 0.0)),
                    )
                )
        return hits
