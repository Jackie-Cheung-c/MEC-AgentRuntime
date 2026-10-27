"""Agent tools for dialogue-driven planning and evidence-driven replanning."""

import json
from typing import Any
from uuid import UUID

from langchain.tools import ToolRuntime, tool

from src.domain import Location, Money, ReplanRequest, TravelEvent, TravelItem
from src.travel.planning import ItineraryPlanningService

from src.domain import StateMutation
from src.travel.contracts import TravelStateOperation
from src.tools.idempotency import build_idempotency_key


def _user_id(runtime: ToolRuntime) -> str:
    user_id = str(runtime.config.get("configurable", {}).get("user_id", "")).strip()
    if not user_id:
        raise ValueError("Authenticated user_id is missing from the tool runtime.")
    return user_id


def build_planning_tools(planner: ItineraryPlanningService) -> list[Any]:
    @tool
    def create_trip_plan(
        title: str,
        timezone: str,
        traveler_count: int,
        destinations: list[dict[str, Any]],
        budget: dict[str, Any] | None,
        preferences: dict[str, Any],
        items: list[dict[str, Any]],
        runtime: ToolRuntime,
    ) -> str:
        """Create and persist the first trip plan after all required user details and external facts have been collected. 
        Never call this while destination, dates, party size, budget, or essential preferences are still missing."""
        user_id = _user_id(runtime)
        idempotency_key = build_idempotency_key(
            runtime=runtime,
            operation="create_trip_plan",
            user_id=user_id,
        )
        result = planner.create_plan(
            user_id=user_id,
            title=title,
            timezone=timezone,
            traveler_count=traveler_count,
            destinations=[Location.model_validate(item) for item in destinations],
            budget=Money.model_validate(budget) if budget else None,
            preferences=preferences,
            items=[TravelItem.model_validate(item) for item in items],
            idempotency_key=idempotency_key,
        )
        return StateMutation(
            adapter="travel-planning",
            operation=TravelStateOperation.STATE_CREATED.value,
            state_id=result.trip_state.trip_id,
            version=result.trip_state.version,
            data={
                "trip_id":str(result.trip_state.trip_id),
                "conflicts":[item.model_dump(mode="json") for item in result.conflicts]
            },
        ).model_dump_json

    @tool
    def propose_trip_replan(
        trip_id: str,
        expected_version: int,
        event: dict[str, Any],
        runtime: ToolRuntime,
        changes: dict[str, Any] | None = None,
        trip_changes: dict[str, Any] | None = None,
        add_items: list[dict[str, Any]] | None = None,
        remove_item_ids: list[str] | None = None,
    ) -> str:
        """Create a pending replan proposal without modifying the current trip.

        For an existing item update/cancellation, put its item ID in
        event.affected_item_ids and its field patch in changes. For trip-wide
        title, budget, traveler count, destinations, or preferences, use
        trip_changes and affected_item_ids may be empty. To replace attractions
        or other itinerary entries, use remove_item_ids and add_items. Explicit
        user changes do not require attachment evidence.
        """
        user_id = _user_id(runtime)
        idempotency_key = build_idempotency_key(
            runtime=runtime,
            operation="propose_trip_replan",
            user_id=user_id,
        )
        proposal = planner.propose_replan(
            UUID(trip_id),
            ReplanRequest(
                user_id=user_id,
                expected_version=expected_version,
                event=TravelEvent.model_validate(event),
                changes=changes or {},
                trip_changes=trip_changes or {},
                add_items=[
                    TravelItem.model_validate(item) for item in (add_items or [])
                ],
                remove_item_ids=[UUID(item_id) for item_id in (remove_item_ids or [])],
            ),
            idempotency_key=idempotency_key,
        )
        return StateMutation(
            adapter="travel_planning",
            operation=TravelStateOperation.PROPOSAL_CREATED.value,
            state_id=proposal.trip_id,
            version=proposal.base_version,
            reference_id=proposal.proposal_id,
            data={
                "trip_id":str(proposal.trip_id),
                "proposal_id":str(proposal.proposal_id),
                "base_version": proposal.base_version,
                "proposed_version": proposal.proposed_state.version,
                "item_changes":[item.model_dump(mode="josn") for item in proposal.changes],
                "trip_changes":[item.model_dump(mode="josn") for item in proposal.trip_changes],
                "conflicts":[item.model_dump(mode="josn") for item in proposal.conflicts],  
                }

        ).model_dump_json

    @tool
    def confirm_trip_replan(
        trip_id: str,
        proposal_id: str,
        runtime: ToolRuntime,
    ) -> str:
        """Confirm a pending replan only after the user explicitly accepts that exact proposal. Never infer consent from silence or from the original disruption report."""
        user_id = _user_id(runtime)
        idempotency_key = build_idempotency_key(
            runtime=runtime,
            operation="confirm_trip_replan",
            user_id=user_id,
            payload={"trip_id": trip_id, "proposal_id": proposal_id},
        )
        state = planner.confirm_replan(
            user_id,
            UUID(trip_id),
            UUID(proposal_id),
            idempotency_key=idempotency_key,
        )
        return StateMutation(
            adapter="travel_planning",
            operation=TravelStateOperation.STATE_CONFIRMED.value,
            state_id=state.trip_id,
            version=state.version,
            reference_id=UUID(proposal_id),
            data={
                "trip_id": str(state.trip_id),
                "proposal_id": proposal_id,
                "version": state.version,
            },
        ).model_dump_json()

    return [create_trip_plan, propose_trip_replan, confirm_trip_replan]
