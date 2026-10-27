"""Travel state adapter for the autonomous Agent."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from src.domain import (
    AgentEvent,
    AgentRunRequest,
    ConfirmationRequest,
    ExecutionTrace,
    StateEnvelope,
    TraceSpanKind,
    TraceSpanStatus,
    TravelEvent,
)
from src.travel.planning import ItineraryPlanningService
from src.state_adapters.base import StateContext, StateOutcome

import logging
from pydantic import ValidationError

from src.domain import StateMutation
from src.travel.contracts import TravelStateOperation
logger = logging.getLogger(__name__)

class TravelStateAdapter:
    name = "travel_planning"

    def __init__(self, planner: ItineraryPlanningService) -> None:
        self.planner = planner

    async def load_context(
        self,
        request: AgentRunRequest,
    ) -> StateContext:
        if request.state_adapter not in {None, self.name}:
            return StateContext(adapter=self.name)

        state_id = request.state_id or request.trip_id
        if state_id is None:
            return StateContext(adapter=self.name)

        current_trip = await asyncio.to_thread(
            self.planner.repository.get,
            request.user_id,
            state_id,
        )
        return StateContext(
            adapter=self.name,
            state=current_trip,
            messages=[
                (
                    "Current persisted TripState. Treat it as the base version "
                    "for all replanning and never mutate it without explicit "
                    "confirmation:\n"
                    + current_trip.model_dump_json()
                )
            ],
        )

    async def collect_outcome(
        self,
        request: AgentRunRequest,
        trace: ExecutionTrace,
        context: StateContext | None,
    ) -> StateOutcome:
        trip_state = context.state if context is not None else None
        events: list[AgentEvent] = []
        confirmations: list[ConfirmationRequest] = []

        for span in trace.spans:
            if (
                span.kind != TraceSpanKind.TOOL
                or span.status != TraceSpanStatus.SUCCEEDED
            ):
                continue

            mutation = self._parse_mutation(span.output)
            if mutation is None:
                continue

            state_id = mutation.state_id
            if state_id is None:
                raise RuntimeError(
                    f"Travel mutation {mutation.operation!r} has no state_id"
                )

            if mutation.operation == TravelStateOperation.STATE_CREATED.value:
                trip_state = await asyncio.to_thread(
                    self.planner.repository.get,
                    request.user_id,
                    state_id,
                )

            elif mutation.operation == TravelStateOperation.PROPOSAL_CREATED.value:
                if mutation.reference_id is None:
                    raise RuntimeError(
                        "proposal_created mutation requires reference_id"
                    )

                proposal = await asyncio.to_thread(
                    self.planner.repository.get_proposal,
                    request.user_id,
                    state_id,
                    mutation.reference_id,
                )
                trip_state = await asyncio.to_thread(
                    self.planner.repository.get,
                    request.user_id,
                    state_id,
                )
                events.append(self._agent_event(proposal.event))
                confirmations.append(
                    ConfirmationRequest(
                        action="confirm_trip_replan",
                        prompt="是否应用该重规划方案？",
                        payload=proposal.model_dump(mode="json"),
                        evidence_ids=proposal.event.evidence_ids,
                    )
                )

            elif mutation.operation == TravelStateOperation.STATE_CONFIRMED.value:
                trip_state = await asyncio.to_thread(
                    self.planner.repository.get,
                    request.user_id,
                    state_id,
                )

        return StateOutcome(
            state=self._state_envelope(trip_state),
            legacy_state=trip_state,
            events=events,
            pending_confirmations=confirmations,
        )
    @staticmethod
    def _state_envelope(trip_state) -> StateEnvelope | None:
        if trip_state is None:
            return None

        return StateEnvelope(
            adapter=TravelStateAdapter.name,
            state_id=trip_state.trip_id,
            version=trip_state.version,
            payload=trip_state.model_dump(mode="json"),
        )


    @staticmethod
    def _agent_event(event: TravelEvent) -> AgentEvent:
        affected_item_ids = list(event.affected_item_ids)

        return AgentEvent(
            event_id=event.event_id,
            event_type=event.event_type.value,
            severity=event.severity.value,
            status=event.status.value,
            summary=event.summary,
            occurred_at=event.occurred_at,
            detected_at=event.detected_at,
            affected_refs=[
                str(item_id)
                for item_id in affected_item_ids
            ],
            affected_item_ids=affected_item_ids,
            evidence_ids=list(event.evidence_ids),
            details=dict(event.details),
        )
    @staticmethod
    def _parse_mutation(output: Any) -> StateMutation | None:
        if isinstance(output, StateMutation):
            return output

        if isinstance(output, dict):
            payload = output
        elif isinstance(output, str):
            text = output.strip()
            if not text.startswith("{"):
                return None
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                logger.warning(
                    "Skipping tool output that is not valid JSON: %s",
                    text[:200],
                )
                return None
        else:
            return None

        if not isinstance(payload, dict):
            return None

        if payload.get("adapter") != TravelStateAdapter.name:
            return None

        try:
            return StateMutation.model_validate(payload)
        except ValidationError as error:
            raise RuntimeError(
                f"Invalid Travel StateMutation: {error}"
            ) from error