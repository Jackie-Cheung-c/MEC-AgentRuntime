from src.tools.mcp.manager import (
    MCPLoadError,
    MCPManager,
    MCPServerSpec,
    MCPServerStatus,
    MCPUnavailableError,
)
from src.tools.mcp.servers import TRAVEL_MCP_SERVERS


def default_mcp_manager() -> MCPManager:
    return MCPManager(TRAVEL_MCP_SERVERS)


__all__ = [
    "MCPLoadError",
    "MCPManager",
    "MCPServerSpec",
    "MCPServerStatus",
    "MCPUnavailableError",
    "TRAVEL_MCP_SERVERS",
    "default_mcp_manager",
]