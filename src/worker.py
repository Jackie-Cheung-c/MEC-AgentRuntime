"""Durable Agent task worker.

The Web process only writes requests into `agent_tasks`. This worker leases
tasks from SQLite, executes them outside the Web process, renews the lease
while running, and recovers tasks whose previous worker died.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import socket
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from config import settings
from src.agent.service import TravelOpsService, UnsupportedAttachmentError
from src.agent.travel_agent import close_default_checkpointer
from src.domain import AgentRunRequest, RunStatus
from src.storage.trace_store import SQLiteTraceStore
from src.tracing.execution_trace import stream_text


logger = logging.getLogger(__name__)


class AgentWorker:
    def __init__(
        self,
        *,
        store: SQLiteTraceStore | None = None,
        service: TravelOpsService | None = None,
        worker_id: str | None = None,
        poll_interval: float | None = None,
        lease_seconds: int | None = None,
        recovery_interval: float | None = None,
        retry_delay_seconds: float | None = None,
    ) -> None:
        self.store = store or SQLiteTraceStore()
        self.service = service or TravelOpsService.with_defaults()
        dependencies = getattr(self.service, "dependencies", None)
        if dependencies is not None and getattr(
            dependencies, "trace_store", None
        ) is None:
            dependencies.trace_store = self.store
            dependencies.trace_recorder = (
                dependencies.trace_recorder or self.store.record
            )
            dependencies.run_starter = (
                dependencies.run_starter or self.store.start_run
            )
            dependencies.run_step_recorder = (
                dependencies.run_step_recorder or self.store.record_step
            )
        self.worker_id = worker_id or (
            f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        )
        self.poll_interval = (
            settings.AGENT_TASK_POLL_INTERVAL_SECONDS
            if poll_interval is None
            else poll_interval
        )
        self.lease_seconds = (
            settings.AGENT_TASK_LEASE_SECONDS
            if lease_seconds is None
            else lease_seconds
        )
        self.recovery_interval = (
            settings.AGENT_TASK_RECOVERY_INTERVAL_SECONDS
            if recovery_interval is None
            else recovery_interval
        )
        self.retry_delay_seconds = (
            settings.AGENT_TASK_RETRY_DELAY_SECONDS
            if retry_delay_seconds is None
            else retry_delay_seconds
        )

    async def run_forever(self, stop_event: asyncio.Event | None = None) -> None:
        loop = asyncio.get_running_loop()
        next_recovery = 0.0
        while stop_event is None or not stop_event.is_set():
            now = loop.time()
            if now >= next_recovery:
                recovered = await asyncio.to_thread(
                    self.store.recover_expired_tasks,
                    limit=100,
                )
                if recovered:
                    logger.warning(
                        "[恢复过期任务] worker=%s|任务=%s",
                        self.worker_id,
                        recovered,
                    )
                next_recovery = now + max(0.1, self.recovery_interval)
            try:
                processed = await self.run_once()
            except Exception:
                logger.exception("Agent worker loop failed")
                processed = False
            if not processed:
                await asyncio.sleep(max(0.01, self.poll_interval))

    async def run_once(self) -> bool:
        task = await asyncio.to_thread(
            self.store.claim_task,
            worker_id=self.worker_id,
            lease_seconds=self.lease_seconds,
        )
        if task is None:
            return False
        logger.info(
            "[领到任务] worker=%s|request_id=%s|租约到期=%s",
            self.worker_id,
            task.get("request_id"),
            task.get("lease_expires_at"),
        )
        await self._execute(task)
        return True

    async def _execute(self, task: dict[str, Any]) -> None:
        request = AgentRunRequest.model_validate_json(task["request_json"])
        request_id = str(request.request_id)
        attempt = int(task.get("attempts") or 1)
        logger.info(
            "[开始执行] request_id=%s|执行次数=%s|worker=%s",
            request_id, attempt, self.worker_id,
        )
        existing = await asyncio.to_thread(
            self.store.get_result_by_request_id,
            user_id=request.user_id,
            request_id=request_id,
        )
        if existing is not None:
            logger.info("[已有结果，直接结束] request_id=%s|状态=%s", request_id, existing.get("status"))
            await asyncio.to_thread(
                self.store.append_event,
                request_id,
                "done",
                {"result": existing},
            )
            await asyncio.to_thread(
                self.store.finish_task,
                request_id=request_id,
                worker_id=self.worker_id,
                status=(
                    "succeeded"
                    if existing.get("status") != RunStatus.FAILED.value
                    else "failed"
                ),
                error=existing.get("error"),
            )
            return

        if attempt > 1:
            logger.warning("[checkpointer断点恢复] request_id=%s|执行次数=%s", request_id, attempt)
            await asyncio.to_thread(
                self.store.append_event,
                request_id,
                "reset",
                {"attempt": attempt},
            )

        event_queue: asyncio.Queue[
            tuple[str, dict[str, Any]] | None
        ] = asyncio.Queue()

        async def flush_events() -> None:
            while True:
                item = await event_queue.get()
                if item is None:
                    return
                event_type, payload = item
                await asyncio.to_thread(
                    self.store.append_event,
                    request_id,
                    event_type,
                    payload,
                )

        def on_event(event: dict[str, Any]) -> None:
            text = stream_text(event)
            if text:
                event_queue.put_nowait(("delta", {"text": text}))

        flusher = asyncio.create_task(flush_events())
        heartbeat_stop = threading.Event()
        # 独立线程续租
        heartbeat = threading.Thread(
            target=self._heartbeat_loop,
            args=(request_id, heartbeat_stop),
            daemon=True,
            name=f"agent-heartbeat-{request_id[:8]}",
        )
        heartbeat.start()
        result = None
        failure: Exception | None = None
        try:
            logger.info("[调用agent] request_id=%s|执行次数=%s|是否属于恢复=%s", request_id, attempt, attempt > 1)
            result = await self.service.run(
                request,
                on_event=on_event,
                recover=attempt > 1,
            )
            logger.info(
                "[agent已返回] request_id=%s|状态=%s",
                request_id,
                result.status,
            )
        except Exception as error:
            failure = error
            logger.exception(
                "[agent异常] request_id=%s|执行次数=%s|worker=%s",
                request_id,
                attempt,
                self.worker_id,
            )
        finally:
            heartbeat_stop.set()
            heartbeat.join(timeout=2.0)
            event_queue.put_nowait(None)
            await flusher

        if failure is not None:
            if isinstance(failure, UnsupportedAttachmentError):
                await asyncio.to_thread(
                    self.store.append_event,
                    request_id,
                    "error",
                    {"error": str(failure)},
                )
                await asyncio.to_thread(
                    self.store.fail_task,
                    request_id=request_id,
                    worker_id=self.worker_id,
                    error=str(failure),
                    retry_delay_seconds=self.retry_delay_seconds,
                    retry=False,
                )
                return
            requeued = await asyncio.to_thread(
                self.store.fail_task,
                request_id=request_id,
                worker_id=self.worker_id,
                error=str(failure),
                retry_delay_seconds=self.retry_delay_seconds,
                retry=True,
            )
            if not requeued:
                await asyncio.to_thread(
                    self.store.append_event,
                    request_id,
                    "error",
                    {"error": str(failure)},
                )
            return

        assert result is not None
        if result.trace is not None:
            await asyncio.to_thread(self.store.record, request, result)
        logger.info("[落库完成] request_id=%s", request_id)
        await asyncio.to_thread(
            self.store.append_event,
            request_id,
            "done",
            {},
        )
        await asyncio.to_thread(
            self.store.finish_task,
            request_id=request_id,
            worker_id=self.worker_id,
            status=(
                "succeeded"
                if result.status != RunStatus.FAILED
                else "failed"
            ),
            error=result.error,
        )

    def _heartbeat_loop(
        self,
        request_id: str,
        stop_event: threading.Event,
    ) -> None:
        interval = max(1.0, self.lease_seconds / 3)
        while not stop_event.wait(interval):
            alive = self.store.heartbeat_task(
                request_id=request_id,
                worker_id=self.worker_id,
                lease_seconds=self.lease_seconds,
            )
            if not alive:
                logger.warning(
                    "[租约丢失，停止续租] 请求=%s worker=%s（可能已被别的worker接管）",
                    request_id, self.worker_id,
                )
                return


async def _run(args: argparse.Namespace) -> None:
    worker = AgentWorker(
        worker_id=args.worker_id,
        poll_interval=args.poll_interval,
        lease_seconds=args.lease_seconds,
    )
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    worker_task: asyncio.Task | None = None
    stop_requested = threading.Event()

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal worker_task
        if stop_requested.is_set():
            os._exit(128 + signum)
        stop_requested.set()
        logger.info("Stop requested; cancelling current worker task.")
        loop.call_soon_threadsafe(stop_event.set)
        if worker_task is not None:
            loop.call_soon_threadsafe(worker_task.cancel)
        # A blocked non-cooperative thread must not keep the console stuck.
        force_exit = threading.Timer(5.0, lambda: os._exit(128 + signum))
        force_exit.daemon = True
        force_exit.start()

    for signal_name in ("SIGINT", "SIGTERM"):
        signal_value = getattr(signal, signal_name, None)
        if signal_value is None:
            continue
        try:
            signal.signal(signal_value, request_stop)
        except (ValueError, OSError):
            pass

    if args.once:
        worker_task = asyncio.create_task(worker.run_once())
    else:
        worker_task = asyncio.create_task(worker.run_forever(stop_event))
    try:
        await worker_task
    except asyncio.CancelledError:
        logger.info("Worker task cancelled.")
    finally:
        try:
            await close_default_checkpointer()
        except Exception:
            logger.exception("Failed to close checkpoint store cleanly.")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.getLogger("src").setLevel(logging.DEBUG)
    parser = argparse.ArgumentParser(description="Run the TravelOps task worker.")
    parser.add_argument("--worker-id", default=None)
    parser.add_argument("--poll-interval", type=float, default=None)
    parser.add_argument("--lease-seconds", type=int, default=None)
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process at most one currently available task, then exit.",
    )
    args = parser.parse_args()
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
