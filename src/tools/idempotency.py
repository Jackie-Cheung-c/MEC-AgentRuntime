"""为有副作用的工具提供稳定的幂等键"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from langchain.tools import ToolRuntime

# 返回request_id 或 tool_call_id，确保在同一请求中调用的工具具有相同的作用域
def request_scope(runtime: ToolRuntime) -> str:
    """Return the durable run scope injected by the application service."""
    config = getattr(runtime, "config", None) or {}
    configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
    raw_request_id = (configurable or {}).get("request_id")
    request_id = str(raw_request_id).strip() if raw_request_id else ""
    if request_id:
        return request_id

    tool_call_id = str(getattr(runtime, "tool_call_id", "") or "").strip()
    if tool_call_id:
        return f"tool-call:{tool_call_id}"
    raise ValueError("Durable request_id is missing from the tool runtime.")

# 生成幂等键
def build_idempotency_key(
    *,
    runtime: ToolRuntime,
    operation: str,
    user_id: str,
    payload: Any | None = None,
) -> str:
    """Build a key that is stable across retries of the same business action."""
    scope = request_scope(runtime)
    digest = f":{payload_fingerprint(payload)}" if payload is not None else ""
    return f"{operation}:{user_id}:{scope}{digest}"


def payload_fingerprint(payload: Any) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:32]
