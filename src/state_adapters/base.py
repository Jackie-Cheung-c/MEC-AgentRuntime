"""Generic state context and outcome adapters used by the Agent service."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from src.domain import AgentRunRequest, ConfirmationRequest, ExecutionTrace
from src.domain import (
    AgentArtifact,
    AgentEvent,
    AgentRunRequest,
    ConfirmationRequest,
    ExecutionTrace,
    StateEnvelope,
)

@dataclass(slots=True)
class StateContext:
    """Persisted state and context injected into one Agent turn."""

    adapter: str
    state: Any | None = None
    messages: list[str] = field(default_factory=list)


@dataclass(slots=True)
class StateOutcome:
    """State changes discovered after the Agent has finished."""

    state: StateEnvelope | None = None
    events: list[AgentEvent] = field(default_factory=list)
    artifacts: list[AgentArtifact] = field(default_factory=list)
    pending_confirmations: list[ConfirmationRequest] = field(
        default_factory=list
    )
    legacy_state: Any | None = None

    def merge(self, other: "StateOutcome") -> None:
        if self.state is None and other.state is not None:
            self.state = other.state
        if self.legacy_state is None and other.legacy_state is not None:
            self.legacy_state = other.legacy_state
        self.events.extend(other.events)
        self.artifacts.extend(other.artifacts)
        self.pending_confirmations.extend(other.pending_confirmations)

class StateAdapter(Protocol):
    """Read state before a run and collect changes after a run.

    This interface must not control tool order or call tools directly.
    """

    name: str

    async def load_context(
        self,
        request: AgentRunRequest,
    ) -> StateContext:
        """Load persisted context for the current request."""
        ...

    async def collect_outcome(
        self,
        request: AgentRunRequest,
        trace: ExecutionTrace,
        context: StateContext | None,
    ) -> StateOutcome:
        """Extract state changes from the autonomous Agent trace."""
        ...