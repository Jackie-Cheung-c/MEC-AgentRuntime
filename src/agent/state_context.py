"""Inject per-turn state context without persisting it in checkpoint messages."""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage
from langgraph.config import get_config


class StateContextMiddleware(AgentMiddleware):
    """Append transient state context to the current model call only."""

    @staticmethod
    def _state_messages() -> tuple[str, ...]:
        configurable = get_config().get("configurable", {})
        raw = configurable.get("state_context_messages", ())

        if isinstance(raw, str):
            values = (raw,)
        elif isinstance(raw, (list, tuple, set)):
            values = raw
        else:
            values = ()

        return tuple(
            str(value).strip()
            for value in values
            if str(value).strip()
        )

    @staticmethod
    def _system_content(
        current: Any,
        state_context: str,
    ) -> str | list[Any]:
        if isinstance(current, str) and current.strip():
            return f"{current.strip()}\n\n{state_context}"

        if isinstance(current, list):
            return [
                *current,
                {"type": "text", "text": state_context},
            ]

        return state_context

    def _apply(self, request):
        state_messages = self._state_messages()
        if not state_messages:
            return request

        state_context = (
            "Current turn state context. This is transient data, not a "
            "persistent conversation instruction:\n\n"
            + "\n\n".join(state_messages)
        )
        current_content = (
            request.system_message.content
            if request.system_message is not None
            else None
        )
        merged = SystemMessage(
            content=self._system_content(
                current_content,
                state_context,
            )
        )
        return request.override(system_message=merged)

    def wrap_model_call(self, request, handler):
        return handler(self._apply(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._apply(request))