"""MCP connections used by TravelTrace Agent."""

import os

from langchain_mcp_adapters.client import MultiServerMCPClient


async def get_travel_mcp_tools():
    """Discover train and flight tools exposed by local MCP servers."""
    variflight_key = os.getenv("VARIFLIGHT_API_KEY", "").strip()
    if not variflight_key:
        raise RuntimeError(
            "VARIFLIGHT_API_KEY is not configured. Set it as an environment variable "
            "before starting TravelTrace Agent."
        )
    client = MultiServerMCPClient(
        {
            "12306-mcp": {
                "transport": "stdio",
                "command": "npx.cmd",
                "args": ["-y", "12306-mcp"],
            },
            "variflight": {
                "transport": "stdio",
                "command": "npx.cmd",
                "args": ["-y", "@variflight-ai/variflight-mcp"],
                "env": {**os.environ, "VARIFLIGHT_API_KEY": variflight_key},
            },
        }
    )
    return await client.get_tools()
