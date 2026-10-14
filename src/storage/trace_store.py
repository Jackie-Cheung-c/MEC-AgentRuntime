"""Durable SQLite persistence for TravelOps execution traces."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from src.domain import AgentRunRequest, AgentRunResult, ExecutionTrace, TraceSpan
from src.paths import DATA_DIR


class SQLiteTraceStore:
    """Persist complete runs and normalized spans in a local SQLite database.

    A connection is opened per operation so the store is safe to use from
    FastAPI worker threads and from multiple service instances in one process.
    SQLite WAL mode keeps trace writes from blocking readers for long periods.
    """

    def __init__(self, path: str | Path = DATA_DIR / "traces.db") -> None:
        self.path = Path(path)
        self._initialize()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
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
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS trace_runs (
                    trace_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    error TEXT,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    trace_json TEXT NOT NULL,
                    result_json TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_trace_runs_request
                    ON trace_runs(request_id);
                CREATE INDEX IF NOT EXISTS idx_trace_runs_session
                    ON trace_runs(user_id, thread_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS trace_spans (
                    trace_id TEXT NOT NULL REFERENCES trace_runs(trace_id) ON DELETE CASCADE,
                    span_id TEXT NOT NULL,
                    parent_span_ids_json TEXT NOT NULL,
                    name TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    input_json TEXT,
                    output_json TEXT,
                    error TEXT,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    duration_ms INTEGER,
                    metadata_json TEXT NOT NULL,
                    PRIMARY KEY (trace_id, span_id)
                );
                CREATE INDEX IF NOT EXISTS idx_trace_spans_kind_status
                    ON trace_spans(kind, status);

                CREATE TABLE IF NOT EXISTS agent_runs (
                    request_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    current_step_id TEXT,
                    started_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    heartbeat_at TEXT,
                    completed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_agent_runs_user_status
                    ON agent_runs(user_id, status, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_agent_runs_recoverable
                    ON agent_runs(status, heartbeat_at);
                CREATE TABLE IF NOT EXISTS agent_steps (
                    request_id TEXT NOT NULL REFERENCES agent_runs(request_id) ON DELETE CASCADE,
                    step_id TEXT NOT NULL,
                    parent_step_ids_json TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    input_json TEXT,
                    output_json TEXT,
                    error TEXT,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    duration_ms INTEGER,
                    metadata_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (request_id, step_id)
                );
                CREATE INDEX IF NOT EXISTS idx_agent_steps_request_updated
                    ON agent_steps(request_id, updated_at ASC);
                CREATE TABLE IF NOT EXISTS agent_tasks (
                    request_id TEXT PRIMARY KEY
                        REFERENCES agent_runs(request_id) ON DELETE CASCADE,
                    status TEXT NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 0,
                    available_at TEXT NOT NULL,
                    lease_owner TEXT,
                    lease_expires_at TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 3,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_agent_tasks_claim
                    ON agent_tasks(status, available_at, priority DESC, created_at ASC);
                CREATE INDEX IF NOT EXISTS idx_agent_tasks_lease
                    ON agent_tasks(status, lease_expires_at);
                CREATE TABLE IF NOT EXISTS agent_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL
                        REFERENCES agent_runs(request_id) ON DELETE CASCADE,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_agent_events_request
                    ON agent_events(request_id, event_id);
                """
            )
            # 会话历史需要的两列：用户问句（列表标题与还原对话）与关联行程。
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(trace_runs)").fetchall()
            }
            if "query" not in columns:
                connection.execute(
                    "ALTER TABLE trace_runs ADD COLUMN query TEXT NOT NULL DEFAULT ''"
                )
            if "trip_id" not in columns:
                connection.execute("ALTER TABLE trace_runs ADD COLUMN trip_id TEXT")
            if "result_json" not in columns:
                connection.execute(
                    "ALTER TABLE trace_runs ADD COLUMN result_json TEXT"
                )
            self._backfill_agent_tasks(connection)

    @staticmethod
    def _backfill_agent_tasks(connection: sqlite3.Connection) -> None:
        """Adopt unfinished runs created before the durable queue existed."""
        now = datetime.now(timezone.utc).isoformat()
        rows = connection.execute(
            """
            SELECT request_id FROM agent_runs AS runs
            WHERE runs.status IN ('queued', 'running')
              AND runs.result_json IS NULL
              AND NOT EXISTS (
                  SELECT 1 FROM agent_tasks AS tasks
                  WHERE tasks.request_id = runs.request_id
              )
            """
        ).fetchall()
        if not rows:
            return
        connection.executemany(
            """
            INSERT INTO agent_tasks (
                request_id, status, priority, available_at,
                attempts, max_attempts, created_at, updated_at
            ) VALUES (?, 'queued', 0, ?, 0, ?, ?, ?)
            """,
            [
                (row["request_id"], now, 3, now, now)
                for row in rows
            ],
        )
        connection.executemany(
            """
            UPDATE agent_runs
            SET status = 'queued', updated_at = ?, heartbeat_at = ?
            WHERE request_id = ?
            """,
            [(now, now, row["request_id"]) for row in rows],
        )
    # 因为 SQLite 默认是延迟锁，先读后写时多个 worker 可能都读到同一个候选任务，然后都去抢。
    # BEGIN IMMEDIATE 一开始就拿写锁，把“读候选 + 更新状态”变成原子操作，避免重复领取。
    @staticmethod
    def _begin_immediate(connection: sqlite3.Connection) -> None:
        """Serialize queue producers and claims before they read state."""
        if not connection.in_transaction:
            connection.execute("BEGIN IMMEDIATE")
    # Web 进程把请求写进 agent_tasks
    def enqueue_run(
        self,
        request: AgentRunRequest,
        *,
        max_attempts: int = 3,
        priority: int = 0,
    ) -> dict[str, Any]:
        """Persist one request in the durable worker queue idempotently."""
        now = datetime.now(timezone.utc).isoformat()
        request_id = str(request.request_id)
        with self._connect() as connection:
            self._begin_immediate(connection)
            existing = self._task_row(connection, request_id)
            if existing is not None:
                payload = dict(existing)
                payload["created"] = False
                return payload

            connection.execute(
                """
                INSERT INTO agent_runs (
                    request_id, user_id, thread_id, status, request_json,
                    started_at, updated_at, heartbeat_at
                ) VALUES (?, ?, ?, 'queued', ?, ?, ?, ?)
                ON CONFLICT(request_id) DO UPDATE SET
                    user_id = excluded.user_id,
                    thread_id = excluded.thread_id,
                    request_json = excluded.request_json,
                    status = CASE
                        WHEN agent_runs.result_json IS NULL THEN 'queued'
                        ELSE agent_runs.status
                    END,
                    updated_at = excluded.updated_at,
                    heartbeat_at = excluded.heartbeat_at
                """,
                (
                    request_id,
                    request.user_id,
                    request.thread_id,
                    request.model_dump_json(),
                    now,
                    now,
                    now,
                ),
            )
            connection.execute(
                """
                INSERT INTO agent_tasks (
                    request_id, status, priority, available_at,
                    attempts, max_attempts, created_at, updated_at
                ) VALUES (?, 'queued', ?, ?, 0, ?, ?, ?)
                """,
                (
                    request_id,
                    priority,
                    now,
                    max(1, max_attempts),
                    now,
                    now,
                ),
            )
            row = self._task_row(connection, request_id)
        payload = dict(row)
        payload["created"] = True
        return payload

    @staticmethod
    def _task_row(
        connection: sqlite3.Connection,
        request_id: UUID | str,
    ) -> sqlite3.Row | None:
        return connection.execute(
            """
            SELECT t.*, r.request_json, r.user_id, r.thread_id,
                   r.result_json, r.error AS run_error
            FROM agent_tasks t
            JOIN agent_runs r ON r.request_id = t.request_id
            WHERE t.request_id = ?
            """,
            (str(request_id),),
        ).fetchone()

    def get_task(self, request_id: UUID | str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = self._task_row(connection, request_id)
        return dict(row) if row is not None else None
    # worker租下最老的可执行任务，状态变为running，设置lease_owner(workerID)和lease_expires_at(到期时间)
    def claim_task(
        self,
        *,
        worker_id: str,
        lease_seconds: int,
    ) -> dict[str, Any] | None:
        """Atomically lease the oldest available task."""
        now = datetime.now(timezone.utc)
        now_text = now.isoformat()
        lease_expires_at = (
            now.timestamp() + max(1, lease_seconds)
        )
        lease_expires_text = datetime.fromtimestamp(
            lease_expires_at, timezone.utc
        ).isoformat()
        with self._connect() as connection:
            self._begin_immediate(connection)
            candidate = connection.execute(
                """
                SELECT request_id FROM agent_tasks
                WHERE status = 'queued' AND available_at <= ?
                ORDER BY priority DESC, created_at ASC
                LIMIT 1
                """,
                (now_text,),
            ).fetchone()
            if candidate is None:
                return None
            request_id = candidate["request_id"]
            connection.execute(
                """
                UPDATE agent_tasks
                SET status = 'running',
                    lease_owner = ?,
                    lease_expires_at = ?,
                    attempts = attempts + 1,
                    updated_at = ?
                WHERE request_id = ? AND status = 'queued'
                """,
                (
                    worker_id,
                    lease_expires_text,
                    now_text,
                    request_id,
                ),
            )
            connection.execute(
                """
                UPDATE agent_runs
                SET status = 'running', heartbeat_at = ?, updated_at = ?
                WHERE request_id = ?
                """,
                (now_text, now_text, request_id),
            )
            row = connection.execute(
                """
                SELECT t.*, r.request_json, r.user_id, r.thread_id
                FROM agent_tasks t
                JOIN agent_runs r ON r.request_id = t.request_id
                WHERE t.request_id = ?
                """,
                (request_id,),
            ).fetchone()
        return dict(row) if row is not None else None
    # 执行期间续租
    def heartbeat_task(
        self,
        *,
        request_id: UUID | str,
        worker_id: str,
        lease_seconds: int,
    ) -> bool:
        now = datetime.now(timezone.utc)
        now_text = now.isoformat()
        lease_expires_text = datetime.fromtimestamp(
            now.timestamp() + max(1, lease_seconds), timezone.utc
        ).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE agent_tasks
                SET lease_expires_at = ?, updated_at = ?
                WHERE request_id = ? AND status = 'running'
                  AND lease_owner = ?
                """,
                (
                    lease_expires_text,
                    now_text,
                    str(request_id),
                    worker_id,
                ),
            )
            if cursor.rowcount != 1:
                return False
            connection.execute(
                """
                UPDATE agent_runs
                SET heartbeat_at = ?, updated_at = ?
                WHERE request_id = ?
                """,
                (now_text, now_text, str(request_id)),
            )
        return True
    # 清空租约
    def finish_task(
        self,
        *,
        request_id: UUID | str,
        worker_id: str,
        status: str,
        error: str | None = None,
    ) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE agent_tasks
                SET status = ?,
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    last_error = ?,
                    updated_at = ?
                WHERE request_id = ? AND status = 'running'
                  AND lease_owner = ?
                """,
                (
                    status,
                    error,
                    now,
                    str(request_id),
                    worker_id,
                ),
            )
            if cursor.rowcount != 1:
                return False
        return True
    # 失败的任务重新入队或标记为失败
    def fail_task(
        self,
        *,
        request_id: UUID | str,
        worker_id: str,
        error: str,
        retry_delay_seconds: float,
        retry: bool = True,
    ) -> bool:
        """Requeue a failed lease when attempts remain, otherwise fail it."""
        now = datetime.now(timezone.utc)
        now_text = now.isoformat()
        with self._connect() as connection:
            self._begin_immediate(connection)
            row = connection.execute(
                """
                SELECT attempts, max_attempts FROM agent_tasks
                WHERE request_id = ? AND status = 'running'
                  AND lease_owner = ?
                """,
                (str(request_id), worker_id),
            ).fetchone()
            if row is None:
                return False
            can_retry = retry and row["attempts"] < row["max_attempts"]
            if can_retry:
                available_at = datetime.fromtimestamp(
                    now.timestamp() + max(0.0, retry_delay_seconds),
                    timezone.utc,
                ).isoformat()
                connection.execute(
                    """
                    UPDATE agent_tasks
                    SET status = 'queued',
                        available_at = ?,
                        lease_owner = NULL,
                        lease_expires_at = NULL,
                        last_error = ?,
                        updated_at = ?
                    WHERE request_id = ?
                    """,
                    (
                        available_at,
                        error[:4000],
                        now_text,
                        str(request_id),
                    ),
                )
                connection.execute(
                    """
                    UPDATE agent_runs
                    SET status = 'queued', error = ?, updated_at = ?,
                        heartbeat_at = ?
                    WHERE request_id = ?
                    """,
                    (error[:4000], now_text, now_text, str(request_id)),
                )
                return True

            connection.execute(
                """
                UPDATE agent_tasks
                SET status = 'failed',
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    last_error = ?,
                    updated_at = ?
                WHERE request_id = ?
                """,
                (error[:4000], now_text, str(request_id)),
            )
            connection.execute(
                """
                UPDATE agent_runs
                SET status = 'failed', error = ?, updated_at = ?,
                    heartbeat_at = ?, completed_at = ?
                WHERE request_id = ?
                """,
                (
                    error[:4000],
                    now_text,
                    now_text,
                    now_text,
                    str(request_id),
                ),
            )
        return False
    # 扫描过期的任务，重新入队或标记为失败
    def recover_expired_tasks(self, *, limit: int = 100) -> list[str]:
        """Requeue tasks whose worker lease expired, or fail exhausted ones."""
        if limit < 1:
            return []
        now = datetime.now(timezone.utc)
        now_text = now.isoformat()
        recovered: list[str] = []
        with self._connect() as connection:
            self._begin_immediate(connection)
            rows = connection.execute(
                """
                SELECT t.request_id, t.attempts, t.max_attempts,
                       r.result_json
                FROM agent_tasks AS t
                JOIN agent_runs AS r ON r.request_id = t.request_id
                WHERE t.status = 'running'
                  AND t.lease_expires_at IS NOT NULL
                  AND t.lease_expires_at < ?
                ORDER BY t.lease_expires_at ASC
                LIMIT ?
                """,
                (now_text, limit),
            ).fetchall()
            for row in rows:
                request_id = row["request_id"]
                if row["result_json"]:
                    result = json.loads(row["result_json"])
                    task_status = (
                        "succeeded"
                        if result.get("status") != "failed"
                        else "failed"
                    )
                    connection.execute(
                        """
                        UPDATE agent_tasks
                        SET status = ?,
                            lease_owner = NULL,
                            lease_expires_at = NULL,
                            updated_at = ?
                        WHERE request_id = ?
                        """,
                        (task_status, now_text, request_id),
                    )
                    continue
                if row["attempts"] < row["max_attempts"]:
                    connection.execute(
                        """
                        UPDATE agent_tasks
                        SET status = 'queued',
                            available_at = ?,
                            lease_owner = NULL,
                            lease_expires_at = NULL,
                            last_error = 'worker lease expired',
                            updated_at = ?
                        WHERE request_id = ?
                        """,
                        (now_text, now_text, request_id),
                    )
                    connection.execute(
                        """
                        UPDATE agent_runs
                        SET status = 'queued', updated_at = ?, heartbeat_at = ?,
                            error = COALESCE(error, 'worker lease expired')
                        WHERE request_id = ?
                        """,
                        (now_text, now_text, request_id),
                    )
                    recovered.append(request_id)
                else:
                    connection.execute(
                        """
                        UPDATE agent_tasks
                        SET status = 'failed',
                            lease_owner = NULL,
                            lease_expires_at = NULL,
                            last_error = 'worker lease expired; attempts exhausted',
                            updated_at = ?
                        WHERE request_id = ?
                        """,
                        (now_text, request_id),
                    )
                    connection.execute(
                        """
                        UPDATE agent_runs
                        SET status = 'failed',
                            error = 'worker lease expired; attempts exhausted',
                            updated_at = ?, heartbeat_at = ?, completed_at = ?
                        WHERE request_id = ?
                        """,
                        (now_text, now_text, now_text, request_id),
                    )
        return recovered
    # 写入事件流，返回 event_id
    def append_event(
        self,
        request_id: UUID | str,
        event_type: str,
        payload: dict[str, Any],
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO agent_events (
                    request_id, event_type, payload_json, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    str(request_id),
                    event_type,
                    json.dumps(payload, ensure_ascii=False, default=str),
                    now,
                ),
            )
            return int(cursor.lastrowid)
    # 支持断点续传的事件流，按 request_id 顺序返回事件
    def list_events(
        self,
        request_id: UUID | str,
        *,
        after_id: int = 0,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        if limit < 1:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT event_id, event_type, payload_json, created_at
                FROM agent_events
                WHERE request_id = ? AND event_id > ?
                ORDER BY event_id ASC
                LIMIT ?
                """,
                (str(request_id), after_id, limit),
            ).fetchall()
        return [
            {
                "event_id": row["event_id"],
                "event_type": row["event_type"],
                "payload": json.loads(row["payload_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def start_run(self, request: AgentRunRequest) -> None:
        """Create an idempotent durable ledger row before Agent execution."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO agent_runs (
                    request_id, user_id, thread_id, status, request_json,
                    started_at, updated_at, heartbeat_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(request_id) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    heartbeat_at = excluded.heartbeat_at
                WHERE agent_runs.status = 'running'
                """,
                (
                    str(request.request_id),
                    request.user_id,
                    request.thread_id,
                    "running",
                    request.model_dump_json(),
                    now,
                    now,
                    now,
                ),
            )

    def record_step(self, request_id: UUID | str, span: TraceSpan) -> None:
        """Upsert one incrementally observed model/tool step."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO agent_steps (
                    request_id, step_id, parent_step_ids_json, kind, name,
                    status, input_json, output_json, error, started_at,
                    completed_at, duration_ms, metadata_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(request_id, step_id) DO UPDATE SET
                    parent_step_ids_json = excluded.parent_step_ids_json,
                    kind = excluded.kind,
                    name = excluded.name,
                    status = excluded.status,
                    input_json = excluded.input_json,
                    output_json = excluded.output_json,
                    error = excluded.error,
                    started_at = excluded.started_at,
                    completed_at = excluded.completed_at,
                    duration_ms = excluded.duration_ms,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    str(request_id),
                    span.span_id,
                    json.dumps(span.parent_span_ids, ensure_ascii=False),
                    span.kind.value,
                    span.name,
                    span.status.value,
                    self._json_dump(span.input),
                    self._json_dump(span.output),
                    span.error,
                    span.started_at.isoformat(),
                    (
                        span.completed_at.isoformat()
                        if span.completed_at
                        else None
                    ),
                    span.duration_ms,
                    json.dumps(span.metadata, ensure_ascii=False),
                    now,
                ),
            )
            connection.execute(
                """
                UPDATE agent_runs
                SET current_step_id = ?, updated_at = ?, heartbeat_at = ?
                WHERE request_id = ?
                """,
                (span.span_id, now, now, str(request_id)),
            )

    def get_run_ledger(
        self,
        *,
        user_id: str,
        request_id: UUID | str,
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM agent_runs
                WHERE request_id = ? AND user_id = ?
                """,
                (str(request_id), user_id),
            ).fetchone()
            if row is None:
                return None
            steps = connection.execute(
                """
                SELECT * FROM agent_steps
                WHERE request_id = ?
                ORDER BY updated_at ASC, started_at ASC
                """,
                (str(request_id),),
            ).fetchall()
        payload = dict(row)
        payload["steps"] = [dict(step) for step in steps]
        return payload

    def list_recoverable_runs(
        self,
        *,
        stale_before: datetime | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List running runs for a future durable worker/resume loop."""
        if limit < 1:
            return []
        with self._connect() as connection:
            if stale_before is None:
                rows = connection.execute(
                    """
                    SELECT * FROM agent_runs
                    WHERE status = 'running'
                    ORDER BY updated_at ASC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM agent_runs
                    WHERE status = 'running'
                      AND COALESCE(heartbeat_at, updated_at) < ?
                    ORDER BY updated_at ASC
                    LIMIT ?
                    """,
                    (stale_before.isoformat(), limit),
                ).fetchall()
        return [dict(row) for row in rows]

    def record(self, request: AgentRunRequest, result: AgentRunResult) -> None:
        """Atomically persist a run and all of its spans."""
        trace = result.trace
        if trace is None:
            return

        trace_id = str(trace.trace_id)
        trace_json = trace.model_dump_json()
        trip_id = result.trip_state.trip_id if result.trip_state else request.trip_id
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO agent_runs (
                    request_id, user_id, thread_id, status, request_json,
                    result_json, error, started_at, updated_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(request_id) DO UPDATE SET
                    user_id = excluded.user_id,
                    thread_id = excluded.thread_id,
                    status = excluded.status,
                    request_json = excluded.request_json,
                    result_json = excluded.result_json,
                    error = excluded.error,
                    updated_at = excluded.updated_at,
                    heartbeat_at = excluded.updated_at,
                    completed_at = excluded.completed_at
                """,
                (
                    str(request.request_id),
                    request.user_id,
                    request.thread_id,
                    result.status.value,
                    request.model_dump_json(),
                    result.model_dump_json(),
                    result.error,
                    trace.started_at.isoformat(),
                    datetime.now(timezone.utc).isoformat(),
                    (
                        result.completed_at.isoformat()
                        if result.completed_at
                        else None
                    ),
                    ),
                )
            # request_id is the business identity of one run; a worker retry
            # replaces the previous attempt's trace instead of violating the
            # unique request index with a new trace_id.
            connection.execute(
                "DELETE FROM trace_runs WHERE request_id = ?",
                (str(request.request_id),),
            )
            connection.execute(
                """
                INSERT INTO trace_runs (
                    trace_id, run_id, request_id, user_id, thread_id, query,
                    trip_id, status, answer, error, started_at, completed_at,
                    trace_json, result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trace_id) DO UPDATE SET
                    run_id=excluded.run_id,
                    request_id=excluded.request_id,
                    user_id=excluded.user_id,
                    thread_id=excluded.thread_id,
                    query=excluded.query,
                    trip_id=excluded.trip_id,
                    status=excluded.status,
                    answer=excluded.answer,
                    error=excluded.error,
                    started_at=excluded.started_at,
                    completed_at=excluded.completed_at,
                    trace_json=excluded.trace_json,
                    result_json=excluded.result_json
                """,
                (
                    trace_id,
                    str(result.run_id),
                    str(request.request_id),
                    request.user_id,
                    request.thread_id,
                    request.query or "",
                    str(trip_id) if trip_id else None,
                    result.status.value,
                    result.answer,
                    result.error,
                    trace.started_at.isoformat(),
                    trace.completed_at.isoformat() if trace.completed_at else None,
                    trace_json,
                    result.model_dump_json(),
                ),
            )
            connection.execute(
                "DELETE FROM trace_spans WHERE trace_id = ?", (trace_id,)
            )
            connection.executemany(
                """
                INSERT INTO trace_spans (
                    trace_id, span_id, parent_span_ids_json, name, kind, status,
                    input_json, output_json, error, started_at, completed_at,
                    duration_ms, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        trace_id,
                        span.span_id,
                        json.dumps(span.parent_span_ids, ensure_ascii=False),
                        span.name,
                        span.kind.value,
                        span.status.value,
                        self._json_dump(span.input),
                        self._json_dump(span.output),
                        span.error,
                        span.started_at.isoformat(),
                        span.completed_at.isoformat() if span.completed_at else None,
                        span.duration_ms,
                        json.dumps(span.metadata, ensure_ascii=False),
                    )
                    for span in trace.spans
                ],
            )
            for span in trace.spans:
                connection.execute(
                    """
                    INSERT INTO agent_steps (
                        request_id, step_id, parent_step_ids_json, kind, name,
                        status, input_json, output_json, error, started_at,
                        completed_at, duration_ms, metadata_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(request_id, step_id) DO UPDATE SET
                        parent_step_ids_json = excluded.parent_step_ids_json,
                        kind = excluded.kind,
                        name = excluded.name,
                        status = excluded.status,
                        input_json = excluded.input_json,
                        output_json = excluded.output_json,
                        error = excluded.error,
                        started_at = excluded.started_at,
                        completed_at = excluded.completed_at,
                        duration_ms = excluded.duration_ms,
                        metadata_json = excluded.metadata_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        str(request.request_id),
                        span.span_id,
                        json.dumps(span.parent_span_ids, ensure_ascii=False),
                        span.kind.value,
                        span.name,
                        span.status.value,
                        self._json_dump(span.input),
                        self._json_dump(span.output),
                        span.error,
                        span.started_at.isoformat(),
                        (
                            span.completed_at.isoformat()
                            if span.completed_at
                            else None
                        ),
                        span.duration_ms,
                        json.dumps(span.metadata, ensure_ascii=False),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )

    def get_trace(self, trace_id: UUID | str) -> ExecutionTrace | None:
        """Load a trace after process restart."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT trace_json FROM trace_runs WHERE trace_id = ?",
                (str(trace_id),),
            ).fetchone()
        return ExecutionTrace.model_validate_json(row["trace_json"]) if row else None

    def get_result_by_request_id(
        self,
        *,
        user_id: str,
        request_id: UUID | str,
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT result_json FROM trace_runs
                WHERE request_id = ? AND user_id = ?
                """,
                (str(request_id), user_id),
            ).fetchone()
        if row is None or not row["result_json"]:
            return None
        return json.loads(row["result_json"])

    def list_traces(
        self,
        *,
        user_id: str | None = None,
        thread_id: str | None = None,
        limit: int = 50,
    ) -> list[ExecutionTrace]:
        """List traces scoped to a user/session, newest first."""
        if limit < 1:
            return []
        clauses: list[str] = []
        values: list[Any] = []
        if user_id is not None:
            clauses.append("user_id = ?")
            values.append(user_id)
        if thread_id is not None:
            clauses.append("thread_id = ?")
            values.append(thread_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT trace_json FROM trace_runs {where} "
                "ORDER BY created_at DESC LIMIT ?",
                (*values, limit),
            ).fetchall()
        return [ExecutionTrace.model_validate_json(row["trace_json"]) for row in rows]

    def list_threads(self, *, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """List conversation threads for one user, newest first."""
        if limit < 1:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT t.thread_id AS thread_id,
                       COUNT(*) AS turns,
                       MAX(t.started_at) AS updated_at,
                       (
                           SELECT i.trip_id FROM trace_runs i
                           WHERE i.user_id = t.user_id
                             AND i.thread_id = t.thread_id
                             AND i.trip_id IS NOT NULL
                           ORDER BY i.started_at DESC LIMIT 1
                       ) AS trip_id
                FROM trace_runs t
                WHERE t.user_id = ?
                GROUP BY t.thread_id
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
            titles = {
                row["thread_id"]: row["title"]
                for row in connection.execute(
                    """
                    SELECT outer_runs.thread_id AS thread_id,
                           outer_runs.query AS title
                    FROM trace_runs outer_runs
                    WHERE outer_runs.user_id = ?
                      AND outer_runs.query <> ''
                      AND outer_runs.started_at = (
                          SELECT MIN(inner_runs.started_at)
                          FROM trace_runs inner_runs
                          WHERE inner_runs.user_id = outer_runs.user_id
                            AND inner_runs.thread_id = outer_runs.thread_id
                      )
                    """,
                    (user_id,),
                ).fetchall()
            }
        return [
            {
                "thread_id": row["thread_id"],
                "title": (titles.get(row["thread_id"]) or "")[:200],
                "turns": row["turns"],
                "trip_id": row["trip_id"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def list_thread_turns(
        self, *, user_id: str, thread_id: str, limit: int = 200
    ) -> list[dict[str, Any]]:
        """Replay one conversation: every question/answer pair in order."""
        if limit < 1:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT run_id, query, answer, status, started_at
                FROM trace_runs
                WHERE user_id = ? AND thread_id = ?
                ORDER BY started_at ASC
                LIMIT ?
                """,
                (user_id, thread_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_thread(self, *, user_id: str, thread_id: str) -> int:
        """Delete one conversation and its spans (ON DELETE CASCADE)."""
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM trace_runs WHERE user_id = ? AND thread_id = ?",
                (user_id, thread_id),
            )
            return cursor.rowcount or 0

    @staticmethod
    def _json_dump(value: Any) -> str | None:
        return None if value is None else json.dumps(value, ensure_ascii=False, default=str)
