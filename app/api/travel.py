"""Travel-domain HTTP routes.

The generic Agent entrypoint intentionally stays at ``/api/agent``.
Only concrete travel business operations live under this router.
"""

from __future__ import annotations

import asyncio
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.api.auth import get_current_user, resolve_user_id
from src.auth import Principal
from src.domain import ReplanProposal, ReplanRequest, TripState
from src.tools.expenses_tool import (
    get_expense_summary_for_user,
    record_expenses_for_user,
)
from src.tools.hotel_tool import search_nearby_hotels
from src.tools.idempotency import payload_fingerprint
from src.tools.search_tool import recommend_attractions
from src.tools.traffic_recommand_tool import plan_route
from src.tools.weather_tool import get_weather
from src.travel import (
    ItineraryPlanningService,
    PlanningError,
    ProposalConflictError,
    TripNotFoundError,
    TripVersionConflictError,
)


router = APIRouter(prefix="/api/travel", tags=["travel"])
CurrentUser = Annotated[Principal, Depends(get_current_user)]


class DestinationRequest(BaseModel):
    destination: str = Field(min_length=1, max_length=100)


class RouteRequest(BaseModel):
    origin: str = Field(min_length=1, max_length=100)
    destination: str = Field(min_length=1, max_length=100)


class ExpenseRequest(BaseModel):
    expense_text: str = Field(min_length=1, max_length=500)


class TripOwnerRequest(BaseModel):
    user_id: str = Field(default="local_user", min_length=1, max_length=128)


def get_planning_service(request: Request) -> ItineraryPlanningService:
    return request.app.state.planning_service


async def _call_tool(tool, payload: dict) -> dict[str, str]:
    try:
        result = await asyncio.to_thread(tool.invoke, payload)
        return {"result": str(result)}
    except Exception as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@router.post("/weather")
async def weather(
    request: DestinationRequest,
    principal: CurrentUser,
) -> dict[str, str]:
    return await _call_tool(get_weather, {"city": request.destination})


@router.post("/attractions")
async def attractions(
    request: DestinationRequest,
    principal: CurrentUser,
) -> dict[str, str]:
    return await _call_tool(
        recommend_attractions,
        {"destination": request.destination},
    )


@router.post("/hotels")
async def hotels(
    request: DestinationRequest,
    principal: CurrentUser,
) -> dict[str, str]:
    return await _call_tool(
        search_nearby_hotels,
        {"attraction": request.destination},
    )


@router.post("/route")
async def route(
    request: RouteRequest,
    principal: CurrentUser,
) -> dict[str, str]:
    return await _call_tool(plan_route, request.model_dump())


@router.post("/expenses")
async def expenses(
    request: ExpenseRequest,
    principal: CurrentUser,
    idempotency_key: Annotated[
        str | None,
        Header(alias="Idempotency-Key", min_length=1, max_length=255),
    ] = None,
) -> dict[str, str]:
    effective_user_id = resolve_user_id(principal)
    scoped_key = (
        f"record_expenses:{effective_user_id}:{idempotency_key}:"
        + payload_fingerprint({"expense_text": request.expense_text.strip()})
        if idempotency_key
        else None
    )
    result = await asyncio.to_thread(
        record_expenses_for_user,
        request.expense_text,
        effective_user_id,
        idempotency_key=scoped_key,
    )
    return {"result": str(result)}


@router.get("/expenses")
async def expense_summary(principal: CurrentUser) -> dict[str, str]:
    result = await asyncio.to_thread(
        get_expense_summary_for_user,
        resolve_user_id(principal),
    )
    return {"result": str(result)}


@router.get("/trips/{trip_id}", response_model=TripState)
async def get_trip(
    trip_id: UUID,
    principal: CurrentUser,
    planner: Annotated[
        ItineraryPlanningService,
        Depends(get_planning_service),
    ],
    user_id: Annotated[
        str | None,
        Query(min_length=1, max_length=128),
    ] = None,
) -> TripState:
    effective_user_id = resolve_user_id(principal, user_id)
    try:
        return await asyncio.to_thread(
            planner.repository.get,
            effective_user_id,
            trip_id,
        )
    except TripNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.post("/trips/{trip_id}/replan", response_model=ReplanProposal)
async def propose_trip_replan(
    trip_id: UUID,
    request: ReplanRequest,
    principal: CurrentUser,
    planner: Annotated[
        ItineraryPlanningService,
        Depends(get_planning_service),
    ],
    idempotency_key: Annotated[
        str | None,
        Header(alias="Idempotency-Key", min_length=1, max_length=255),
    ] = None,
) -> ReplanProposal:
    secured_request = request.model_copy(
        update={"user_id": resolve_user_id(principal, request.user_id)}
    )
    scoped_key = (
        f"propose_trip_replan:{secured_request.user_id}:{idempotency_key}"
        if idempotency_key
        else None
    )
    try:
        return await asyncio.to_thread(
            planner.propose_replan,
            trip_id,
            secured_request,
            idempotency_key=scoped_key,
        )
    except TripNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except TripVersionConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except PlanningError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.post(
    "/trips/{trip_id}/replan/{proposal_id}/confirm",
    response_model=TripState,
)
async def confirm_trip_replan(
    trip_id: UUID,
    proposal_id: UUID,
    request: TripOwnerRequest,
    principal: CurrentUser,
    planner: Annotated[
        ItineraryPlanningService,
        Depends(get_planning_service),
    ],
    idempotency_key: Annotated[
        str | None,
        Header(alias="Idempotency-Key", min_length=1, max_length=255),
    ] = None,
) -> TripState:
    effective_user_id = resolve_user_id(principal, request.user_id)
    scoped_key = (
        f"confirm_trip_replan:{effective_user_id}:{idempotency_key}:"
        + payload_fingerprint(
            {"trip_id": str(trip_id), "proposal_id": str(proposal_id)}
        )
        if idempotency_key
        else None
    )
    try:
        return await asyncio.to_thread(
            planner.confirm_replan,
            effective_user_id,
            trip_id,
            proposal_id,
            idempotency_key=scoped_key,
        )
    except TripNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (TripVersionConflictError, ProposalConflictError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
