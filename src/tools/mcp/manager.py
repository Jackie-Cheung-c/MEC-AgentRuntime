"""Generic MCP server lifecycle and tool discovery manager."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MCPServerSpec:
    name: str
    connection: Mapping[str, Any]
    tags: frozenset[str] = field(default_factory=frozenset)
    required_env: tuple[str, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not name:
            raise ValueError("MCP server name cannot be empty")

        tags = frozenset(
            str(tag).strip().casefold()
            for tag in self.tags
            if str(tag).strip()
        )
        env_names = tuple(
            dict.fromkeys(
                str(name).strip()
                for name in self.required_env
                if str(name).strip()
            )
        )

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "tags", tags)
        object.__setattr__(self, "required_env", env_names)
        object.__setattr__(self, "description", self.description.strip())


@dataclass(frozen=True, slots=True)
class MCPServerStatus:
    name: str
    available: bool
    loaded: bool = False
    tool_count: int = 0
    missing_env: tuple[str, ...] = ()
    error: str | None = None


class MCPUnavailableError(RuntimeError):
    pass


class MCPLoadError(RuntimeError):
    pass


class MCPManager:
    """Load MCP tools one server at a time.

    A missing credential or a failed server is isolated to that server.
    The Agent can continue with the remaining MCP servers and ordinary tools.
    """

    def __init__(
        self,
        servers: Iterable[MCPServerSpec],
        *,
        tool_name_prefix: bool = True,
    ) -> None:
        self._servers: dict[str, MCPServerSpec] = {}
        for server in servers:
            if server.name in self._servers:
                raise ValueError(f"Duplicate MCP server: {server.name}")
            self._servers[server.name] = server

        self.tool_name_prefix = tool_name_prefix
        self._statuses: dict[str, MCPServerStatus] = {}

    @property
    def server_names(self) -> tuple[str, ...]:
        return tuple(self._servers)

    def statuses(self) -> list[MCPServerStatus]:
        return list(self._statuses.values())

    def _select(
        self,
        *,
        server_names: Sequence[str] | None = None,
        tags: Iterable[str] = (),
    ) -> list[MCPServerSpec]:
        if server_names is None:
            candidates = list(self._servers.values())
        else:
            candidates = [self._servers[name] for name in server_names]

        required_tags = {
            str(tag).strip().casefold()
            for tag in tags
            if str(tag).strip()
        }
        return [
            server
            for server in candidates
            if required_tags.issubset(server.tags)
        ]

    @staticmethod
    def _missing_env(server: MCPServerSpec) -> tuple[str, ...]:
        return tuple(
            name
            for name in server.required_env
            if not os.getenv(name, "").strip()
        )

    @staticmethod
    def _connection(server: MCPServerSpec) -> dict[str, Any]:
        connection = deepcopy(dict(server.connection))
        configured_env = connection.get("env")

        if server.required_env or configured_env:
            child_env = dict(os.environ)
            child_env.update(dict(configured_env or {}))
            connection["env"] = child_env

        return connection

    async def load_tools(
        self,
        *,
        server_names: Sequence[str] | None = None,
        tags: Iterable[str] = (),
        strict: bool = False,
    ) -> list[Any]:
        tools: list[Any] = []
        seen_names: set[str] = set()

        for server in self._select(server_names=server_names, tags=tags):
            missing_env = self._missing_env(server)
            if missing_env:
                self._statuses[server.name] = MCPServerStatus(
                    name=server.name,
                    available=False,
                    missing_env=missing_env,
                )
                message = (
                    f"MCP server {server.name!r} skipped; "
                    f"missing environment: {', '.join(missing_env)}"
                )
                if strict:
                    raise MCPUnavailableError(message)
                logger.info(message)
                continue

            try:
                client = MultiServerMCPClient(
                    {server.name: self._connection(server)},
                    tool_name_prefix=self.tool_name_prefix,
                    handle_tool_errors=True,
                )
                loaded = await client.get_tools(server_name=server.name)
            except Exception as error:
                self._statuses[server.name] = MCPServerStatus(
                    name=server.name,
                    available=True,
                    error=str(error) or error.__class__.__name__,
                )
                message = f"Failed to load MCP server {server.name!r}: {error}"
                if strict:
                    raise MCPLoadError(message) from error
                logger.exception(message)
                continue

            added = 0
            for tool in loaded:
                name = str(getattr(tool, "name", "") or id(tool))
                if name in seen_names:
                    continue
                seen_names.add(name)
                tools.append(tool)
                added += 1

            self._statuses[server.name] = MCPServerStatus(
                name=server.name,
                available=True,
                loaded=True,
                tool_count=added,
            )

        return tools