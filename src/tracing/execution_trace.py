"""Collect structured spans from a LangChain agent turn."""

from __future__ import annotations

import json
import inspect
from collections.abc import Callable
from datetime import datetime, timezone
from time import perf_counter
from typing import Any
from uuid import uuid4

from src.domain import ExecutionTrace, TraceSpan, TraceSpanKind, TraceSpanStatus


def content_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            block if isinstance(block, str) else str(block.get("text", ""))
            for block in value
            if isinstance(block, (str, dict))
        )
    return str(value or "")


def stream_text(event: dict[str, Any]) -> str:
    """Extract user-visible text from one LangChain streaming event."""
    if event.get("event") != "on_chat_model_stream":
        return ""
    chunk = event.get("data", {}).get("chunk")
    if chunk is None:
        return ""
    if (
        getattr(chunk, "tool_calls", None)
        or getattr(chunk, "tool_call_chunks", None)
        or (getattr(chunk, "additional_kwargs", None) or {}).get("tool_calls")
    ):
        return ""
    return content_text(getattr(chunk, "content", ""))


def _json_value(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _span_kind(event_name: str) -> TraceSpanKind | None:
    if event_name.startswith("on_tool_"):
        return TraceSpanKind.TOOL
    if event_name.startswith("on_chat_model_") or event_name.startswith("on_llm_"):
        return TraceSpanKind.MODEL
    if event_name.startswith("on_retriever_"):
        return TraceSpanKind.RETRIEVER
    if event_name.startswith("on_chain_"):
        return TraceSpanKind.CHAIN
    return None


def _span_output(kind: TraceSpanKind, value: Any) -> Any:
    if kind in {TraceSpanKind.TOOL, TraceSpanKind.MODEL}:
        output = getattr(value, "content", value)
        return content_text(output)[:4000
                                    ]
    return None


async def run_with_evidence(
    agent,
    messages: list[dict[str, str]] | None,
    config: dict[str, Any],
    on_event: Callable[[dict[str, Any]], None] | None = None,
    on_span: Callable[[TraceSpan], Any] | None = None,
) -> tuple[str, ExecutionTrace]:
    """Run an agent and return its final answer plus a serializable span trace."""
    trace_started_at = datetime.now(timezone.utc)
    final_response = ""
    spans: list[TraceSpan] = []
    spans_by_run_id: dict[str, TraceSpan] = {}
    clocks_by_run_id: dict[str, float] = {}

    payload = None if messages is None else {"messages": messages}
    async for event in agent.astream_events(payload, config=config, version="v2"):
        if on_event:
            on_event(event)
        event_name = str(event.get("event", ""))
        kind = _span_kind(event_name)
        if kind is None:
            continue

        event_data = event.get("data", {})
        run_id = str(event.get("run_id") or uuid4())
        now = datetime.now(timezone.utc)

        if event_name.endswith("_start"):
            span = TraceSpan(
                span_id=run_id,
                parent_span_ids=[str(item) for item in event.get("parent_ids", [])],
                name=str(event.get("name", "unknown")),
                kind=kind,
                status=TraceSpanStatus.RUNNING,
                input=(
                    _json_value(event_data.get("input", {}))
                    if kind == TraceSpanKind.TOOL
                    else None
                ),
                started_at=now,
            )
            spans.append(span)
            spans_by_run_id[run_id] = span
            clocks_by_run_id[run_id] = perf_counter()
            if on_span is not None:
                callback_result = on_span(span)
                if inspect.isawaitable(callback_result):
                    await callback_result
            continue

        span = spans_by_run_id.get(run_id)
        if span is None:
            span = TraceSpan(
                span_id=run_id,
                parent_span_ids=[str(item) for item in event.get("parent_ids", [])],
                name=str(event.get("name", "unknown")),
                kind=kind,
                status=TraceSpanStatus.RUNNING,
                started_at=now,
            )
            spans.append(span)
            spans_by_run_id[run_id] = span

        elapsed = clocks_by_run_id.get(run_id)
        duration_ms = round((perf_counter() - elapsed) * 1000) if elapsed else 0
        if event_name.endswith("_error"):
            error = content_text(event_data.get("error", "Unknown execution error"))
            updated = span.model_copy(
                update={
                    "status": TraceSpanStatus.FAILED,
                    "error": error[:4000],
                    "completed_at": now,
                    "duration_ms": duration_ms,
                }
            )
        elif event_name.endswith("_end"):
            output = _span_output(kind, event_data.get("output"))
            updated = span.model_copy(
                update={
                    "status": TraceSpanStatus.SUCCEEDED,
                    "output": output,
                    "completed_at": now,
                    "duration_ms": duration_ms,
                }
            )
            if kind == TraceSpanKind.MODEL and output and str(output).strip():
                final_response = str(output)
        else:
            continue

        spans[spans.index(span)] = updated
        spans_by_run_id[run_id] = updated
        if on_span is not None:
            callback_result = on_span(updated)
            if inspect.isawaitable(callback_result):
                await callback_result

    return final_response, ExecutionTrace(
        spans=spans,
        started_at=trace_started_at,
        completed_at=datetime.now(timezone.utc),
    )
