"""Global capability registry for shared tools and MCP providers."""

from __future__ import annotations

import inspect
import logging
import os
import re
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

CapabilityLoader = Callable[[], Any | Awaitable[Any]]
CapabilityKind = Literal["tool", "mcp"]

_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]*$")

# 懒加载工具
@dataclass(frozen=True, slots=True)
class Capability:
    capability_id: str
    loader: CapabilityLoader = field(repr=False, compare=False)
    tags: frozenset[str] = field(default_factory=frozenset)
    kind: CapabilityKind = "tool"
    description: str = ""
    requires_env: tuple[str, ...] = ()
    enabled: bool = True

    def __post_init__(self) -> None:
        capability_id = self.capability_id.strip().casefold()
        if not _ID_PATTERN.fullmatch(capability_id):
            raise ValueError(f"Invalid capability_id: {self.capability_id!r}")
        if self.kind not in {"tool", "mcp"}:
            raise ValueError(f"Invalid capability kind: {self.kind!r}")

        tags = frozenset(
            str(tag).strip().casefold()
            for tag in self.tags
            if str(tag).strip()
        )
        env_names = tuple(
            dict.fromkeys(
                str(name).strip()
                for name in self.requires_env
                if str(name).strip()
            )
        )

        object.__setattr__(self, "capability_id", capability_id)
        object.__setattr__(self, "tags", tags)
        object.__setattr__(self, "requires_env", env_names)
        object.__setattr__(self, "description", self.description.strip())


class CapabilityRegistry:
    """索引工具，但不拥有工具实现"""

    def __init__(self) -> None:
        self._capabilities: dict[str, Capability] = {}

    def register(self, capability: Capability) -> Capability:
        if capability.capability_id in self._capabilities:
            raise ValueError(
                f"Capability already registered: {capability.capability_id}"
            )
        self._capabilities[capability.capability_id] = capability
        return capability

    def register_tool(
        self,
        tool: Any,
        *,
        capability_id: str,
        tags: Iterable[str] = (),
        description: str = "",
        requires_env: Iterable[str] = (),
        enabled: bool = True,
    ) -> Capability:
        return self.register(
            Capability(
                capability_id=capability_id,
                loader=lambda: tool,
                tags=frozenset(tags),
                kind="tool",
                description=description
                or str(getattr(tool, "description", "") or ""),
                requires_env=tuple(requires_env),
                enabled=enabled,
            )
        )

    def register_provider(
        self,
        *,
        capability_id: str,
        loader: CapabilityLoader,
        tags: Iterable[str] = (),
        kind: CapabilityKind = "tool",
        description: str = "",
        requires_env: Iterable[str] = (),
        enabled: bool = True,
    ) -> Capability:
        return self.register(
            Capability(
                capability_id=capability_id,
                loader=loader,
                tags=frozenset(tags),
                kind=kind,
                description=description,
                requires_env=tuple(requires_env),
                enabled=enabled,
            )
        )

    def get(self, capability_id: str) -> Capability:
        return self._capabilities[capability_id.strip().casefold()]

    def unavailable_reason(self, capability: Capability) -> str | None:
        missing = [
            name for name in capability.requires_env
            if not os.getenv(name, "").strip()
        ]
        return f"missing environment: {', '.join(missing)}" if missing else None

    def select(
        self,
        *,
        capability_ids: Sequence[str] | None = None,
        tags: Iterable[str] = (),
        any_tags: Iterable[str] = (),
        kind: CapabilityKind | None = None,
    ) -> list[Capability]:
        required_tags = {str(tag).strip().casefold() for tag in tags}
        optional_tags = {str(tag).strip().casefold() for tag in any_tags}

        if capability_ids is None:
            candidates = list(self._capabilities.values())
        else:
            candidates = [self.get(item) for item in capability_ids]

        selected = []
        for capability in candidates:
            if not capability.enabled:
                continue
            if kind is not None and capability.kind != kind:
                continue
            if not required_tags.issubset(capability.tags):
                continue
            if optional_tags and not optional_tags.intersection(capability.tags):
                continue
            selected.append(capability)
        return selected

    def describe(self) -> list[dict[str, Any]]:
        result = []
        for capability in self._capabilities.values():
            reason = self.unavailable_reason(capability)
            result.append(
                {
                    "capability_id": capability.capability_id,
                    "kind": capability.kind,
                    "tags": sorted(capability.tags),
                    "description": capability.description,
                    "requires_env": list(capability.requires_env),
                    "available": reason is None,
                    "unavailable_reason": reason,
                }
            )
        return result

    async def load(
        self,
        *,
        capability_ids: Sequence[str] | None = None,
        tags: Iterable[str] = (),
        any_tags: Iterable[str] = (),
        kind: CapabilityKind | None = None,
        strict: bool = False,
    ) -> list[Any]:
        tools: list[Any] = []
        seen: set[str] = set()

        for capability in self.select(
            capability_ids=capability_ids,
            tags=tags,
            any_tags=any_tags,
            kind=kind,
        ):
            reason = self.unavailable_reason(capability)
            if reason:
                if strict:
                    raise RuntimeError(
                        f"{capability.capability_id} unavailable: {reason}"
                    )
                logger.info("Skipping %s: %s", capability.capability_id, reason)
                continue

            try:
                loader_kwargs: dict[str, Any] = {}
                try:
                    loader_signature = inspect.signature(
                        capability.loader
                    )
                except (TypeError, ValueError):
                    loader_signature = None
                if (
                    strict
                    and loader_signature is not None
                    and (
                        "strict" in loader_signature.parameters
                        or any(
                            parameter.kind
                            == inspect.Parameter.VAR_KEYWORD
                            for parameter in loader_signature.parameters.values()
                        )
                    )
                ):
                    loader_kwargs["strict"] = True
                loaded = capability.loader(**loader_kwargs)
                if inspect.isawaitable(loaded):
                    loaded = await loaded
            except Exception:
                if strict:
                    raise
                logger.exception("Failed to load %s", capability.capability_id)
                continue

            values = loaded if isinstance(loaded, (list, tuple, set)) else [loaded]
            for tool in values:
                if tool is None:
                    continue
                name = str(getattr(tool, "name", "") or id(tool))
                if name in seen:
                    continue
                seen.add(name)
                tools.append(tool)

        return tools
