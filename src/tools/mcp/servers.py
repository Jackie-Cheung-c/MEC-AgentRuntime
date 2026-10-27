"""MCP server definitions shared by the application."""

from __future__ import annotations

from src.tools.mcp.manager import MCPServerSpec


TRAVEL_MCP_SERVERS = (
    MCPServerSpec(
        name="12306-mcp",
        connection={
            "transport": "stdio",
            "command": "npx.cmd",
            "args": ["-y", "12306-mcp"],
        },
        tags=frozenset({"travel", "rail"}),
        description="Railway ticket query tools.",
    ),
    MCPServerSpec(
        name="variflight",
        connection={
            "transport": "stdio",
            "command": "npx.cmd",
            "args": ["-y", "@variflight-ai/variflight-mcp"],
        },
        tags=frozenset({"travel", "flight"}),
        required_env=("VARIFLIGHT_API_KEY",),
        description="Flight status and flight ticket query tools.",
    ),
)