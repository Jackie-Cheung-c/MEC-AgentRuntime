"""LLM summary compression for LangGraph checkpoint history."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import tiktoken
from config import settings
from langchain_core.messages import RemoveMessage, SystemMessage
from src.paths import DATA_DIR
from src.prompts.prompt_manager import prompt_manager


COMPRESSION_LOG = DATA_DIR / "context_compression.jsonl"
COMPRESSION_SYSTEM_PROMPT = prompt_manager.get("compression_system_prompt")
TOKEN_ENCODER = tiktoken.get_encoding("o200k_base")


def _tool_call_ids(message) -> set[str]:
    """Return tool-call IDs requested by an AI message."""
    return {
        str(call["id"])
        for call in getattr(message, "tool_calls", [])
        if call.get("id")
    }


def estimate_message_tokens(messages: list) -> int:
    """Count checkpoint message tokens, including tool calls and results."""
    serialized_messages = [
        {
            "type": getattr(message, "type", "message"),
            "content": getattr(message, "content", ""),
            "tool_calls": getattr(message, "tool_calls", []),
            "tool_call_id": getattr(message, "tool_call_id", None),
            "additional_kwargs": getattr(message, "additional_kwargs", {}),
        }
        for message in messages
    ]
    payload = json.dumps(serialized_messages, ensure_ascii=False, default=str)
    return len(TOKEN_ENCODER.encode(payload))


async def _aget_state(agent, config):
    aget_state = getattr(agent, "aget_state", None)
    if aget_state is not None:
        return await aget_state(config)
    return agent.get_state(config)


async def _aupdate_state(agent, config, update):
    aupdate_state = getattr(agent, "aupdate_state", None)
    if aupdate_state is not None:
        return await aupdate_state(config, update)
    return agent.update_state(config, update)


async def _remove_orphan_tool_messages(agent, config, messages: list) -> list:
    """Discard persisted tool outputs whose matching tool call is no longer present."""
    known_call_ids: set[str] = set()
    orphan_messages = []
    valid_messages = []
    for message in messages:
        known_call_ids.update(_tool_call_ids(message))
        if getattr(message, "type", "") == "tool":
            tool_call_id = str(getattr(message, "tool_call_id", ""))
            if tool_call_id not in known_call_ids:
                orphan_messages.append(message)
                continue
        valid_messages.append(message)

    removals = [RemoveMessage(id=message.id) for message in orphan_messages if message.id]
    if removals:
        await _aupdate_state(agent, config, {"messages": removals})
        print(f"[debug] removed {len(removals)} orphaned tool result(s) from context")
    return valid_messages


def _recent_complete_messages(messages: list, keep_recent: int) -> list:
    """Keep recent messages without splitting an AI tool call from its result."""
    start = max(0, len(messages) - keep_recent)
    while start > 0:
        kept = messages[start:]
        requested_ids = set().union(*(_tool_call_ids(message) for message in kept))
        result_ids = {
            str(getattr(message, "tool_call_id", ""))
            for message in kept
            if getattr(message, "type", "") == "tool"
        }
        missing_call_ids = result_ids - requested_ids
        if not missing_call_ids:
            break
        start -= 1
    return messages[start:]


async def compress_checkpoint(
    agent,
    model,
    config,
    keep_recent: int = 2,
    threshold_tokens: int = settings.CONTEXT_COMPRESSION_THRESHOLD_TOKENS,
    log_path: Path = COMPRESSION_LOG,
) -> None:
    state = await _aget_state(agent, config)
    messages = await _remove_orphan_tool_messages(
        agent, config, list(state.values.get("messages", []))
    )
    before_tokens = estimate_message_tokens(messages)
    if before_tokens < threshold_tokens:
        return

    recent_messages = _recent_complete_messages(messages, keep_recent)
    old_messages = messages[: len(messages) - len(recent_messages)]
    if not old_messages:
        return
    source = "\n".join(
        f"{getattr(message, 'type', 'message')}: {str(getattr(message, 'content', ''))[:1200]}"
        for message in old_messages
    )
    response = await model.ainvoke(
        COMPRESSION_SYSTEM_PROMPT + "\n\n以下是需要压缩的历史对话：\n\n" + source
    )
    summary = SystemMessage(content="历史对话摘要（旧消息已压缩）：\n" + str(response.content)[:3000])
    removals = [RemoveMessage(id=message.id) for message in old_messages if getattr(message, "id", None)]
    await _aupdate_state(agent, config, {"messages": removals + [summary]})
    event = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "thread_id": config.get("configurable", {}).get("thread_id"),
        "before_messages": len(messages),
        "before_tokens": before_tokens,
        "compression_threshold_tokens": threshold_tokens,
        "removed_messages": len(old_messages),
        "kept_recent_messages": len(recent_messages),
        "after_messages": len(recent_messages) + 1,
        "summary": str(response.content)[:3000],
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(
        f"[debug] context 压缩完毕: {event['before_tokens']} tokens, "
        f"{event['before_messages']} -> {event['after_messages']} messages; "
        f"removed={event['removed_messages']}"
    )
