"""Backward-compatible MCP entrypoint."""

from typing import Any

from src.tools.mcp import default_mcp_manager


async def get_travel_mcp_tools() -> list[Any]:
    """Return all currently available travel MCP tools.

    Missing credentials and unavailable MCP servers are skipped by MCPManager.
    """
    return await default_mcp_manager().load_tools()