"""Shared tool catalog. Implementations stay in src.tools."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import config.settings  # noqa: F401  # Load project .env before env checks.

from src.tools.expenses_tool import get_expense_summary, record_expenses
from src.tools.hotel_tool import search_nearby_hotels
from src.tools.registry import CapabilityRegistry
from src.tools.search_tool import recommend_attractions
from src.tools.traffic_recommand_tool import plan_route
from src.tools.weather_tool import get_weather
from src.tools.mcp import MCPManager, TRAVEL_MCP_SERVERS

_mcp_manager = MCPManager(TRAVEL_MCP_SERVERS)

async def _load_travel_mcp_tools() -> list[Any]:
    from src.tools.mcp_client import get_travel_mcp_tools

    return await get_travel_mcp_tools()


def _build_registry() -> CapabilityRegistry:
    registry = CapabilityRegistry()

    registry.register_tool(
        get_weather,
        capability_id="weather.current",
        tags=("weather", "location", "travel"),
        requires_env=("GAODE_API_KEY",),
    )
    registry.register_tool(
        search_nearby_hotels,
        capability_id="places.hotel.search",
        tags=("places", "lodging", "travel"),
        requires_env=("GAODE_API_KEY",),
    )
    registry.register_tool(
        recommend_attractions,
        capability_id="places.attraction.recommend",
        tags=("places", "discovery", "travel"),
        requires_env=("GAODE_API_KEY",),
    )
    registry.register_tool(
        plan_route,
        capability_id="maps.route.plan",
        tags=("maps", "route", "travel"),
        requires_env=("GAODE_API_KEY",),
    )
    registry.register_tool(
        record_expenses,
        capability_id="finance.expense.record",
        tags=("finance", "expense", "ledger"),
    )
    registry.register_tool(
        get_expense_summary,
        capability_id="finance.expense.summary",
        tags=("finance", "expense", "ledger"),
    )

    registry.register_provider(
        capability_id="mcp.travel",
        loader=_mcp_manager.load_tools,
        tags=("mcp", "travel", "rail", "flight"),
        kind="mcp",
        description="Optional travel rail and flight MCP tools.",
    )
    return registry


tool_registry = _build_registry()


async def load_enabled_tools(
    *,
    capability_ids: Sequence[str] | None = None,
    tags: Iterable[str] = (),
    any_tags: Iterable[str] = (),
    strict: bool = False,
) -> list[Any]:
    return await tool_registry.load(
        capability_ids=capability_ids,
        tags=tags,
        any_tags=any_tags,
        strict=strict,
    )


def capability_manifest() -> list[dict[str, Any]]:
    return tool_registry.describe()