"""用户长期记忆的SQLite唯一事实来源"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from src.memory.models import (
    MemoryCategory,
    MemoryItem,
    MemoryScope,
    MemoryStatus,
)
from src.memory.key_policy import (
    MemoryValueCardinality,
    memory_value_cardinality,
    merge_memory_values,
)
from src.memory.tokenization import fts_match_query, tokenize
from src.paths import DATA_DIR


logger = logging.getLogger(__name__)


class MemoryNotFoundError(LookupError):
    """Raised when a memory item does not exist for the requested user."""


class MemoryStateConflictError(RuntimeError):
    """Raised when a state transition is attempted from an invalid state."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SQLiteMemoryRepository:
    """持久化并管理长期记忆，不与检索耦合
    SQLite 仍是规范存储。
    向量/关键词索引可以稍后这些行进行镜像，但每个读或写决策都必须先在这里验证
    """

    def __init__(
        self,
        path: str | Path = DATA_DIR / "memories.db",
        *,
        vector_index: Any | None = None,
    ) -> None:
        self.path = Path(path)
        self.vector_index = vector_index
        self._initialize()
        self._reconcile_vector_index()

    @contextmanager
    def _connect(self, *, immediate: bool = False):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            if immediate:
                connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS memory_items (
                    memory_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    memory_key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    trip_id TEXT,
                    thread_id TEXT,
                    confidence REAL NOT NULL,
                    source_type TEXT NOT NULL,
                    source_ref TEXT NOT NULL,
                    evidence_quote TEXT,
                    status TEXT NOT NULL,
                    supersedes_id TEXT REFERENCES memory_items(memory_id),
                    valid_from TEXT NOT NULL,
                    expires_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_memory_user_status_updated
                    ON memory_items(user_id, status, updated_at DESC);

                CREATE INDEX IF NOT EXISTS idx_memory_user_scope
                    ON memory_items(user_id, scope, trip_id, thread_id);

                CREATE INDEX IF NOT EXISTS idx_memory_user_key
                    ON memory_items(user_id, category, memory_key, updated_at DESC);

                DROP INDEX IF EXISTS idx_memory_active_profile_key;

                CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_active_single_key
                    ON memory_items(
                        user_id,
                        category,
                        memory_key,
                        scope,
                        COALESCE(trip_id, ''),
                        COALESCE(thread_id, '')
                    )
                    WHERE status = 'active'
                      AND memory_key IN (
                          'name',
                          'home_city',
                          'preferred_departure_city',
                          'preferred_transport',
                          'preferred_hotel_level',
                          'seat_preference',
                          'travel_pace'
                      );
                
                -- fts5全文检索表
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_items_fts
                    USING fts5(memory_id UNINDEXED, content);

                -- 记忆抽取状态表，记录当前抽取记忆抽到哪一轮会话，避免重复
                CREATE TABLE IF NOT EXISTS memory_extraction_state (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    last_turn_id TEXT,
                    last_extracted_at TEXT,
                    extraction_count INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_memory_extraction_user_thread
                    ON memory_extraction_state(user_id, thread_id);
                """
            )
            self._rebuild_fts(connection)

    def _rebuild_fts(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute("SELECT * FROM memory_items").fetchall()
        for row in rows:
            self._sync_fts(connection, self._row_to_memory(row))

    def _reconcile_vector_index(self) -> None:
        if self.vector_index is None or not hasattr(self.vector_index, "count"):
            return
        try:
            with self._connect() as connection:
                now = _utc_now().astimezone(timezone.utc).isoformat()
                sqlite_count = int(
                    connection.execute(
                        """
                        SELECT COUNT(*) FROM memory_items
                        WHERE status = ?
                          AND valid_from <= ?
                          AND (expires_at IS NULL OR expires_at > ?)
                        """,
                        (MemoryStatus.ACTIVE.value, now, now),
                    ).fetchone()[0]
                )
            if self.vector_index.count() != sqlite_count:
                self.rebuild_vector_index()
        except Exception as error:
            logger.warning("Failed to reconcile memory vector index: %s", error)

    # 存记忆，同步存入FTS5索引和向量库
    def save(self, item: MemoryItem) -> MemoryItem:
        """Insert one memory item.
        Callers should use `supersede()` when replacing an existing value.
        """
        with self._connect() as connection:
            self._insert(connection, item)
            self._sync_fts(connection, item)
        self._sync_vector(item)
        return item

    def upsert_active(self, item: MemoryItem) -> MemoryItem:
        """Insert or refresh one active logical memory without duplicating it.

        Profile, preference, and constraint items are unique by
        user/category/key/scope. Episodic items are append-only because the
        same event key can legitimately occur more than once.
        """
        if item.status != MemoryStatus.ACTIVE:
            raise ValueError("upsert_active() requires an active memory item.")
        if item.category == MemoryCategory.EPISODIC:
            return self.save(item)

        with self._connect(immediate=True) as connection:
            row = connection.execute(
                """
                SELECT * FROM memory_items
                WHERE user_id = ?
                  AND category = ?
                  AND memory_key = ?
                  AND scope = ?
                  AND COALESCE(trip_id, '') = COALESCE(?, '')
                  AND COALESCE(thread_id, '') = COALESCE(?, '')
                  AND status = ?
                LIMIT 1
                """,
                (
                    item.user_id,
                    item.category.value,
                    item.memory_key,
                    item.scope.value,
                    str(item.trip_id) if item.trip_id else None,
                    item.thread_id,
                    MemoryStatus.ACTIVE.value,
                ),
            ).fetchone()

            if row is None:
                self._insert(connection, item)
                self._sync_fts(connection, item)
                self._sync_vector(item)
                return item

            current = self._row_to_memory(row)
            cardinality = memory_value_cardinality(
                item.category,
                item.memory_key,
            )
            if cardinality == MemoryValueCardinality.MULTI:
                refreshed = item.model_copy(
                    update={
                        "memory_id": current.memory_id,
                        "value": merge_memory_values(
                            current.value,
                            item.value,
                        ),
                        "supersedes_id": current.supersedes_id,
                        "created_at": current.created_at,
                        "updated_at": _utc_now(),
                    }
                )
                connection.execute(
                    """
                    UPDATE memory_items
                    SET value_json = ?, confidence = ?, source_type = ?,
                        source_ref = ?, evidence_quote = ?, valid_from = ?,
                        expires_at = ?, updated_at = ?, metadata_json = ?
                    WHERE memory_id = ? AND user_id = ?
                    """,
                    (
                        json.dumps(
                            refreshed.value,
                            ensure_ascii=False,
                            default=str,
                        ),
                        refreshed.confidence,
                        refreshed.source_type.value,
                        refreshed.source_ref,
                        refreshed.evidence_quote,
                        refreshed.valid_from.isoformat(),
                        (
                            refreshed.expires_at.isoformat()
                            if refreshed.expires_at
                            else None
                        ),
                        refreshed.updated_at.isoformat(),
                        json.dumps(
                            refreshed.metadata,
                            ensure_ascii=False,
                            default=str,
                        ),
                        str(refreshed.memory_id),
                        refreshed.user_id,
                    ),
                )
                self._sync_fts(connection, refreshed)
                self._sync_vector(refreshed)
                return refreshed

            if current.value == item.value:
                refreshed = item.model_copy(
                    update={
                        "memory_id": current.memory_id,
                        "supersedes_id": current.supersedes_id,
                        "created_at": current.created_at,
                        "updated_at": _utc_now(),
                    }
                )
                connection.execute(
                    """
                    UPDATE memory_items
                    SET value_json = ?, confidence = ?, source_type = ?,
                        source_ref = ?, evidence_quote = ?, valid_from = ?,
                        expires_at = ?, updated_at = ?, metadata_json = ?
                    WHERE memory_id = ? AND user_id = ?
                    """,
                    (
                        json.dumps(
                            refreshed.value, ensure_ascii=False, default=str
                        ),
                        refreshed.confidence,
                        refreshed.source_type.value,
                        refreshed.source_ref,
                        refreshed.evidence_quote,
                        refreshed.valid_from.isoformat(),
                        (
                            refreshed.expires_at.isoformat()
                            if refreshed.expires_at
                            else None
                        ),
                        refreshed.updated_at.isoformat(),
                        json.dumps(
                            refreshed.metadata,
                            ensure_ascii=False,
                            default=str,
                        ),
                        str(refreshed.memory_id),
                        refreshed.user_id,
                    ),
                )
                self._sync_fts(connection, refreshed)
                self._sync_vector(refreshed)
                return refreshed

            now = _utc_now()
            replacement = item.model_copy(
                update={"supersedes_id": current.memory_id, "updated_at": now}
            )
            connection.execute(
                """
                UPDATE memory_items
                SET status = ?, updated_at = ?
                WHERE memory_id = ? AND user_id = ?
                """,
                (
                    MemoryStatus.SUPERSEDED.value,
                    now.isoformat(),
                    str(current.memory_id),
                    item.user_id,
                ),
            )
            self._insert(connection, replacement)
            self._sync_fts(connection, replacement)
            self._sync_vector(current, delete_only=True)
            self._sync_vector(replacement)
            return replacement

    def get(self, user_id: str, memory_id: UUID | str) -> MemoryItem:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM memory_items
                WHERE memory_id = ? AND user_id = ?
                """,
                (str(memory_id), user_id),
            ).fetchone()
        if row is None:
            raise MemoryNotFoundError(
                f"Memory {memory_id} was not found for this user."
            )
        return self._row_to_memory(row)

    def list_active(
        self,
        user_id: str,
        *,
        categories: Iterable[MemoryCategory] | None = None,
        scopes: Iterable[MemoryScope] | None = None,
        trip_id: UUID | str | None = None,
        thread_id: str | None = None,
        limit: int = 100,
        now: datetime | None = None,
    ) -> list[MemoryItem]:
        """List currently active memories that match the supplied scope.

        Passing a scope does not automatically expand it. Callers must query
        global items and the current trip/thread separately, then combine them
        explicitly so trip memory cannot leak into other trips.
        """
        if limit < 1:
            return []

        clauses = [
            "user_id = ?",
            "status = ?",
            "valid_from <= ?",
            "(expires_at IS NULL OR expires_at > ?)",
        ]
        moment = (now or _utc_now()).astimezone(timezone.utc).isoformat()
        values: list[Any] = [user_id, MemoryStatus.ACTIVE.value, moment, moment]

        category_values = [item.value for item in categories or ()]
        if category_values:
            placeholders = ", ".join("?" for _ in category_values)
            clauses.append(f"category IN ({placeholders})")
            values.extend(category_values)

        scope_values = [item.value for item in scopes or ()]
        if scope_values:
            placeholders = ", ".join("?" for _ in scope_values)
            clauses.append(f"scope IN ({placeholders})")
            values.extend(scope_values)

        if trip_id is not None:
            clauses.append("trip_id = ?")
            values.append(str(trip_id))
        if thread_id is not None:
            clauses.append("thread_id = ?")
            values.append(thread_id)

        values.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM memory_items
                WHERE {" AND ".join(clauses)}
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                values,
            ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def list_history(
        self,
        user_id: str,
        *,
        category: MemoryCategory,
        memory_key: str,
        limit: int = 50,
    ) -> list[MemoryItem]:
        """Return every version of one logical memory key, newest first."""
        if limit < 1:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM memory_items
                WHERE user_id = ? AND category = ? AND memory_key = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (user_id, category.value, memory_key, limit),
            ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def supersede(
        self,
        user_id: str,
        old_memory_id: UUID | str,
        replacement: MemoryItem,
    ) -> MemoryItem:
        """Atomically mark the old item superseded and insert its replacement."""
        if replacement.user_id != user_id:
            raise ValueError("replacement memory belongs to a different user.")

        with self._connect(immediate=True) as connection:
            row = connection.execute(
                """
                SELECT status, category, memory_key, scope, trip_id, thread_id
                FROM memory_items
                WHERE memory_id = ? AND user_id = ?
                """,
                (str(old_memory_id), user_id),
            ).fetchone()
            if row is None:
                raise MemoryNotFoundError(
                    f"Memory {old_memory_id} was not found for this user."
                )
            if row["status"] != MemoryStatus.ACTIVE.value:
                raise MemoryStateConflictError(
                    "Only an active memory item can be superseded."
                )

            identity = (
                row["category"],
                row["memory_key"],
                row["scope"],
                row["trip_id"],
                row["thread_id"],
            )
            replacement_identity = (
                replacement.category.value,
                replacement.memory_key,
                replacement.scope.value,
                str(replacement.trip_id) if replacement.trip_id else None,
                replacement.thread_id,
            )
            if identity != replacement_identity:
                raise ValueError(
                    "replacement must keep the same category, key, and scope."
                )

            now = _utc_now()
            connection.execute(
                """
                UPDATE memory_items
                SET status = ?, updated_at = ?
                WHERE memory_id = ? AND user_id = ?
                """,
                (
                    MemoryStatus.SUPERSEDED.value,
                    now.isoformat(),
                    str(old_memory_id),
                    user_id,
                ),
            )
            stored = replacement.model_copy(
                update={
                    "supersedes_id": UUID(str(old_memory_id)),
                    "updated_at": now,
                }
            )
            self._insert(connection, stored)
            self._sync_fts(connection, stored)
        if self.vector_index is not None:
            self.vector_index.delete(old_memory_id)
        self._sync_vector(stored)
        return stored

    def set_status(
        self,
        user_id: str,
        memory_id: UUID | str,
        status: MemoryStatus,
    ) -> MemoryItem:
        """Soft-delete, reject, activate, or otherwise change one item."""
        if status == MemoryStatus.SUPERSEDED:
            raise ValueError("Use supersede() to replace an active memory.")

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM memory_items
                WHERE memory_id = ? AND user_id = ?
                """,
                (str(memory_id), user_id),
            ).fetchone()
            if row is None:
                raise MemoryNotFoundError(
                    f"Memory {memory_id} was not found for this user."
                )
            item = self._row_to_memory(row)
            if item.status == MemoryStatus.SUPERSEDED:
                raise MemoryStateConflictError(
                    "A superseded memory cannot be activated again."
                )

            updated = item.model_copy(
                update={"status": status, "updated_at": _utc_now()}
            )
            connection.execute(
                """
                UPDATE memory_items
                SET status = ?, updated_at = ?
                WHERE memory_id = ? AND user_id = ?
                """,
                (
                    updated.status.value,
                    updated.updated_at.isoformat(),
                    str(memory_id),
                    user_id,
                ),
            )
            self._sync_fts(connection, updated)
        self._sync_vector(
            updated,
            delete_only=updated.status != MemoryStatus.ACTIVE,
        )
        return updated

    def forget(self, user_id: str, memory_id: UUID | str) -> MemoryItem:
        """Soft-delete one memory so an audit record remains available."""
        return self.set_status(user_id, memory_id, MemoryStatus.DELETED)

    def delete(self, user_id: str, memory_id: UUID | str) -> bool:
        """Permanently remove one memory item.

        Prefer `forget()` for normal user-facing deletion.
        """
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM memory_items WHERE memory_id = ? AND user_id = ?",
                (str(memory_id), user_id),
            )
            connection.execute(
                "DELETE FROM memory_items_fts WHERE memory_id = ?",
                (str(memory_id),),
            )
        if self.vector_index is not None:
            self.vector_index.delete(memory_id)
        return bool(cursor.rowcount)

    def find_active(
        self,
        user_id: str,
        *,
        category: MemoryCategory,
        memory_key: str,
        scope: MemoryScope,
        trip_id: UUID | str | None = None,
        thread_id: str | None = None,
    ) -> MemoryItem | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM memory_items
                WHERE user_id = ?
                  AND category = ?
                  AND memory_key = ?
                  AND scope = ?
                  AND COALESCE(trip_id, '') = COALESCE(?, '')
                  AND COALESCE(thread_id, '') = COALESCE(?, '')
                  AND status = ?
                LIMIT 1
                """,
                (
                    user_id,
                    category.value,
                    memory_key,
                    scope.value,
                    str(trip_id) if trip_id else None,
                    thread_id,
                    MemoryStatus.ACTIVE.value,
                ),
            ).fetchone()
        return self._row_to_memory(row) if row else None

    def get_many(
        self,
        user_id: str,
        memory_ids: Sequence[UUID | str],
    ) -> dict[str, MemoryItem]:
        if not memory_ids:
            return {}
        values = [str(memory_id) for memory_id in memory_ids]
        placeholders = ", ".join("?" for _ in values)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM memory_items
                WHERE user_id = ? AND memory_id IN ({placeholders})
                """,
                (user_id, *values),
            ).fetchall()
        return {
            str(item.memory_id): item
            for item in (self._row_to_memory(row) for row in rows)
        }

    def list_items(
        self,
        user_id: str | None = None,
        *,
        status: MemoryStatus | None = MemoryStatus.ACTIVE,
        category: MemoryCategory | None = None,
        scope: MemoryScope | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[MemoryItem]:
        if limit < 1 or offset < 0:
            return []
        clauses: list[str] = []
        values: list[Any] = []
        if user_id is not None:
            clauses.append("user_id = ?")
            values.append(user_id)
        if status is not None:
            clauses.append("status = ?")
            values.append(status.value)
            if status == MemoryStatus.ACTIVE:
                moment = _utc_now().astimezone(timezone.utc).isoformat()
                clauses.extend(
                    [
                        "valid_from <= ?",
                        "(expires_at IS NULL OR expires_at > ?)",
                    ]
                )
                values.extend([moment, moment])
        if category is not None:
            clauses.append("category = ?")
            values.append(category.value)
        if scope is not None:
            clauses.append("scope = ?")
            values.append(scope.value)
        values.extend([limit, offset])
        with self._connect() as connection:
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            rows = connection.execute(
                f"""
                SELECT * FROM memory_items
                {where}
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
                """,
                values,
            ).fetchall()
        return [self._row_to_memory(row) for row in rows]
    # FTS5关键词检索
    def search_fts(
        self,
        user_id: str,
        query: str,
        *,
        trip_id: UUID | str | None = None,
        limit: int = 20,
        now: datetime | None = None,
    ) -> list[tuple[str, float]]:
        match_query = fts_match_query(query)
        if not match_query or limit < 1:
            return []
        moment = (now or _utc_now()).astimezone(timezone.utc).isoformat()
        scope_clause = "m.scope = ?"
        scope_values: list[Any] = [MemoryScope.GLOBAL.value]
        if trip_id is not None:
            scope_clause = (
                "(m.scope = ? OR (m.scope = ? AND m.trip_id = ?))"
            )
            scope_values = [
                MemoryScope.GLOBAL.value,
                MemoryScope.TRIP.value,
                str(trip_id),
            ]
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT m.memory_id, bm25(memory_items_fts) AS rank
                FROM memory_items_fts
                JOIN memory_items AS m
                  ON m.memory_id = memory_items_fts.memory_id
                WHERE memory_items_fts MATCH ?
                  AND m.user_id = ?
                  AND m.status = ?
                  AND m.valid_from <= ?
                  AND (m.expires_at IS NULL OR m.expires_at > ?)
                  AND {scope_clause}
                ORDER BY rank
                LIMIT ?
                """,
                (
                    match_query,
                    user_id,
                    MemoryStatus.ACTIVE.value,
                    moment,
                    moment,
                    *scope_values,
                    limit,
                ),
            ).fetchall()
        return [
            (str(row["memory_id"]), -float(row["rank"] or 0.0))
            for row in rows
        ]

    def get_extraction_state(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM memory_extraction_state WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return dict(row) if row else None

    def mark_extracted(
        self,
        *,
        session_id: str,
        user_id: str,
        thread_id: str,
        last_turn_id: str,
        extracted_at: datetime | None = None,
    ) -> None:
        moment = extracted_at or _utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO memory_extraction_state (
                    session_id, user_id, thread_id, last_turn_id,
                    last_extracted_at, extraction_count, updated_at
                ) VALUES (?, ?, ?, ?, ?, 1, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    user_id = excluded.user_id,
                    thread_id = excluded.thread_id,
                    last_turn_id = excluded.last_turn_id,
                    last_extracted_at = excluded.last_extracted_at,
                    extraction_count = memory_extraction_state.extraction_count + 1,
                    updated_at = excluded.updated_at
                """,
                (
                    session_id,
                    user_id,
                    thread_id,
                    last_turn_id,
                    moment.isoformat(),
                    moment.isoformat(),
                ),
            )

    def rebuild_vector_index(self, *, batch_size: int = 100) -> int:
        if self.vector_index is None:
            return 0
        count = 0
        offset = 0
        while True:
            items = self.list_items(
                user_id=None,
                status=None,
                limit=batch_size,
                offset=offset,
            )
            if not items:
                break
            for item in items:
                self._sync_vector(item)
                count += 1
            offset += len(items)
        return count

    def _sync_vector(
        self,
        item: MemoryItem,
        *,
        delete_only: bool = False,
    ) -> None:
        if self.vector_index is None:
            return
        try:
            if delete_only or item.status != MemoryStatus.ACTIVE:
                self.vector_index.delete(item.memory_id)
            else:
                self.vector_index.upsert(item)
        except Exception as error:
            logger.warning(
                "Failed to sync memory vector %s: %s",
                item.memory_id,
                error,
            )

    @staticmethod
    def _sync_fts(
        connection: sqlite3.Connection,
        item: MemoryItem,
    ) -> None:
        connection.execute(
            "DELETE FROM memory_items_fts WHERE memory_id = ?",
            (str(item.memory_id),),
        )
        content = " ".join(
            tokenize(
                " ".join(
                    (
                        item.category.value,
                        item.memory_key,
                        json.dumps(item.value, ensure_ascii=False, default=str),
                        item.evidence_quote or "",
                    )
                )
            )
        )
        connection.execute(
            "INSERT INTO memory_items_fts(memory_id, content) VALUES (?, ?)",
            (str(item.memory_id), content),
        )

    @staticmethod
    def _insert(connection: sqlite3.Connection, item: MemoryItem) -> None:
        connection.execute(
            """
            INSERT INTO memory_items (
                memory_id, user_id, category, memory_key, value_json, scope,
                trip_id, thread_id, confidence, source_type, source_ref,
                evidence_quote, status, supersedes_id, valid_from, expires_at,
                created_at, updated_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(item.memory_id),
                item.user_id,
                item.category.value,
                item.memory_key,
                json.dumps(item.value, ensure_ascii=False, default=str),
                item.scope.value,
                str(item.trip_id) if item.trip_id else None,
                item.thread_id,
                item.confidence,
                item.source_type.value,
                item.source_ref,
                item.evidence_quote,
                item.status.value,
                str(item.supersedes_id) if item.supersedes_id else None,
                item.valid_from.isoformat(),
                item.expires_at.isoformat() if item.expires_at else None,
                item.created_at.isoformat(),
                item.updated_at.isoformat(),
                json.dumps(item.metadata, ensure_ascii=False, default=str),
            ),
        )

    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> MemoryItem:
        return MemoryItem.model_validate(
            {
                "memory_id": row["memory_id"],
                "user_id": row["user_id"],
                "category": row["category"],
                "memory_key": row["memory_key"],
                "value": json.loads(row["value_json"]),
                "scope": row["scope"],
                "trip_id": row["trip_id"],
                "thread_id": row["thread_id"],
                "confidence": row["confidence"],
                "source_type": row["source_type"],
                "source_ref": row["source_ref"],
                "evidence_quote": row["evidence_quote"],
                "status": row["status"],
                "supersedes_id": row["supersedes_id"],
                "valid_from": row["valid_from"],
                "expires_at": row["expires_at"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "metadata": json.loads(row["metadata_json"]),
            }
        )
