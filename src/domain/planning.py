"""Contracts for deterministic itinerary planning and confirmed replanning."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import Field, JsonValue, model_validator

from src.domain.base import DomainModel
from src.domain.travel import TravelEvent, TravelItem, TripState


class PlanningConflictKind(StrEnum):
    TIME_OVERLAP = "time_overlap"
    INVALID_CHANGE = "invalid_change"
    MISSING_SCHEDULE = "missing_schedule"


class PlanningConflict(DomainModel):
    kind: PlanningConflictKind
    summary: str = Field(min_length=1, max_length=1000)
    item_ids: list[UUID] = Field(default_factory=list)
    blocking: bool = True


class ReplanRequest(DomainModel):
    user_id: str = Field(default="local_user", min_length=1, max_length=128)
    expected_version: int = Field(ge=1)
    event: TravelEvent
    changes: dict[str, JsonValue] = Field(default_factory=dict)
    trip_changes: dict[str, JsonValue] = Field(default_factory=dict)
    add_items: list[TravelItem] = Field(default_factory=list)
    remove_item_ids: list[UUID] = Field(default_factory=list)


class ReplanChange(DomainModel):
    item_id: UUID
    action: str = Field(pattern=r"^(update|cancel|add|remove)$")
    before: TravelItem | None = None
    after: TravelItem | None = None
    reason: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def validate_item_versions(self) -> "ReplanChange":
        if self.action == "add" and (self.before is not None or self.after is None):
            raise ValueError("add changes require only an after item")
        if self.action == "remove" and (self.before is None or self.after is not None):
            raise ValueError("remove changes require only a before item")
        if self.action in {"update", "cancel"} and (
            self.before is None or self.after is None
        ):
            raise ValueError("update and cancel changes require before and after items")
        return self


class TripFieldChange(DomainModel):
    field: str = Field(
        pattern=r"^(title|timezone|traveler_count|destinations|budget|preferences)$"
    )
    before: JsonValue = None
    after: JsonValue = None
    reason: str = Field(min_length=1, max_length=1000)


class ReplanProposal(DomainModel):
    proposal_id: UUID = Field(default_factory=uuid4)
    trip_id: UUID
    user_id: str = Field(min_length=1, max_length=128)
    base_version: int = Field(ge=1)
    proposed_state: TripState
    event: TravelEvent
    changes: list[ReplanChange] = Field(default_factory=list)
    trip_changes: list[TripFieldChange] = Field(default_factory=list)
    conflicts: list[PlanningConflict] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    confirmed_at: datetime | None = None

    @model_validator(mode="after")
    def validate_versions(self) -> "ReplanProposal":
        if self.proposed_state.trip_id != self.trip_id:
            raise ValueError("proposed state must belong to the same trip")
        if self.proposed_state.version != self.base_version + 1:
            raise ValueError("proposed state version must follow the base version")
        return self


class PlanningResult(DomainModel):
    trip_state: TripState
    conflicts: list[PlanningConflict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
