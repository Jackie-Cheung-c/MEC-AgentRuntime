"""Per-turn tool visibility for one shared Agent instance."""

from __future__ import annotations

import logging
from fnmatch import fnmatchcase
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langgraph.config import get_config

logger = logging.getLogger(__name__)


class DynamicToolSelectionMiddleware(AgentMiddleware):
    """Filter request.tools without rebuilding the Agent."""

    @staticmethod
    def _config_patterns() -> tuple[str, ...]:
        configurable = get_config().get("configurable", {})
        raw = configurable.get("tool_patterns", ())

        if isinstance(raw, str):
            values = raw.replace(";", ",").replace("\n", ",").split(",")
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
    def _tool_name(tool: Any) -> str:
        if isinstance(tool, dict):
            return str(tool.get("name", "")).strip()
        return str(
            getattr(tool, "name", "")
            or getattr(tool, "__name__", "")
        ).strip()

    def _apply(self, request):
        patterns = self._config_patterns()
        if not patterns:
            return request

        selected = [
            tool
            for tool in request.tools
            if any(
                fnmatchcase(self._tool_name(tool), pattern)
                for pattern in patterns
            )
        ]

        if not selected:
            logger.warning(
                "No tools matched tool_patterns=%s; keeping all tools.",
                patterns,
            )
            return request

        logger.debug(
            "Selected %d/%d tools for patterns=%s",
            len(selected),
            len(request.tools),
            patterns,
        )
        return request.override(tools=selected)

    def wrap_model_call(self, request, handler):
        return handler(self._apply(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._apply(request))