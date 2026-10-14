"""Evidence-backed itinerary planning and two-phase replanning."""

from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, time, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any
from uuid import UUID
from zoneinfo import ZoneInfo
from zoneinfo import ZoneInfoNotFoundError

from src.domain import (
    DocumentType,
    EventStatus,
    ExtractedDocument,
    ExtractedField,
    ExtractionStatus,
    Location,
    Money,
    PlanningConflict,
    PlanningConflictKind,
    PlanningResult,
    ReplanChange,
    ReplanProposal,
    ReplanRequest,
    TravelEventType,
    TravelItem,
    TravelItemKind,
    TravelItemStatus,
    TripState,
    TripFieldChange,
)


class PlanningError(RuntimeError):
    pass


class TripNotFoundError(LookupError):
    pass


class TripVersionConflictError(RuntimeError):
    pass


class ProposalConflictError(RuntimeError):
    pass


_KIND_BY_DOCUMENT = {
    DocumentType.FLIGHT_TICKET: TravelItemKind.FLIGHT,
    DocumentType.TRAIN_TICKET: TravelItemKind.TRAIN,
    DocumentType.HOTEL_BOOKING: TravelItemKind.HOTEL,
    DocumentType.ATTRACTION_BOOKING: TravelItemKind.ATTRACTION,
}

_START_SUFFIXES = (
    "departure_time", "depart_at", "start_at", "start_time", "check_in", "checkin"
)
_END_SUFFIXES = (
    "arrival_time", "arrive_at", "end_at", "end_time", "check_out", "checkout"
)
_ORIGIN_SUFFIXES = ("origin", "departure_city", "departure_station", "departure_airport")
_DESTINATION_SUFFIXES = ("destination", "arrival_city", "arrival_station", "arrival_airport")
_REFERENCE_SUFFIXES = ("booking_reference", "booking_number", "order_number", "number")
_TITLE_SUFFIXES = ("name", "hotel_name", "attraction_name")
_COST_SUFFIXES = ("total", "amount", "price", "cost")


class ItineraryPlanningService:
    """Create itinerary state from reviewed extraction data and propose changes."""

    def __init__(self, repository: Any) -> None:
        self.repository = repository

    def create_plan(
        self,
        *,
        user_id: str,
        items: list[TravelItem],
        title: str | None = None,
        timezone: str = "Asia/Shanghai",
        traveler_count: int = 1,
        destinations: list[Location] | None = None,
        budget: Money | None = None,
        preferences: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> PlanningResult:
        """Persist a dialogue-produced itinerary after deterministic validation."""
        if idempotency_key:
            get_cached = getattr(self.repository, "get_idempotent_trip", None)
            if get_cached is not None:
                existing = get_cached(idempotency_key)
                if existing is not None:
                    return PlanningResult(
                        trip_state=existing,
                        conflicts=self.detect_conflicts(existing.items),
                    )
        try:
            ZoneInfo(timezone)
        except ZoneInfoNotFoundError as error:
            raise PlanningError(f"Unknown trip timezone: {timezone}.") from error
        if not items:
            raise PlanningError("A trip requires at least one itinerary item.")
        normalized_items = [
            item.model_copy(update={"status": TravelItemStatus.PLANNED})
            for item in items
        ]
        ordered = sorted(
            normalized_items,
            key=lambda item: item.start_at
            or datetime.max.replace(tzinfo=ZoneInfo(timezone)),
        )
        state = TripState(
            user_id=user_id,
            title=title,
            timezone=timezone,
            traveler_count=traveler_count,
            destinations=destinations or [],
            budget=budget,
            preferences=preferences or {},
            items=ordered,
        )
        conflicts = self.detect_conflicts(ordered)
        if any(conflict.blocking for conflict in conflicts):
            raise PlanningError(
                "The initial itinerary contains blocking schedule conflicts."
            )
        stored = self.repository.create(state, idempotency_key=idempotency_key)
        same_state = stored.trip_id == state.trip_id
        return PlanningResult(
            trip_state=stored,
            conflicts=(
                conflicts if same_state else self.detect_conflicts(stored.items)
            ),
        )

    def create_plan_from_documents(
        self,
        *,
        user_id: str,
        documents: list[ExtractedDocument],
        title: str | None = None,
        timezone: str = "Asia/Shanghai",
        idempotency_key: str | None = None,
    ) -> PlanningResult:
        if idempotency_key:
            get_cached = getattr(self.repository, "get_idempotent_trip", None)
            if get_cached is not None:
                existing = get_cached(idempotency_key)
                if existing is not None:
                    return PlanningResult(
                        trip_state=existing,
                        conflicts=self.detect_conflicts(existing.items),
                    )
        try:
            zone = ZoneInfo(timezone)
        except ZoneInfoNotFoundError as error:
            raise PlanningError(f"Unknown trip timezone: {timezone}.") from error
        items: list[TravelItem] = []
        warnings: list[str] = []
        for document in documents:
            item = self._item_from_document(document, zone, warnings)
            if item is not None:
                items.append(item)
        if not items:
            raise PlanningError(
                "No supported, usable travel documents were available to build a trip."
            )
        items.sort(key=lambda item: item.start_at or datetime.max.replace(tzinfo=zone))
        state = TripState(
            user_id=user_id,
            title=title,
            timezone=timezone,
            items=items,
        )
        conflicts = self.detect_conflicts(items)
        stored = self.repository.create(state, idempotency_key=idempotency_key)
        same_state = stored.trip_id == state.trip_id
        return PlanningResult(
            trip_state=stored,
            conflicts=(
                conflicts if same_state else self.detect_conflicts(stored.items)
            ),
            warnings=warnings,
        )

    def propose_replan(
        self,
        trip_id: UUID,
        request: ReplanRequest,
        *,
        idempotency_key: str | None = None,
    ) -> ReplanProposal:
        if idempotency_key:
            get_cached = getattr(self.repository, "get_idempotent_proposal", None)
            if get_cached is not None:
                existing = get_cached(idempotency_key)
                if existing is not None:
                    return existing
        current = self.repository.get(request.user_id, trip_id)
        if current.version != request.expected_version:
            raise TripVersionConflictError(
                f"Expected trip version {request.expected_version}, found {current.version}."
            )
        event = request.event.model_copy(deep=True)
        state = current.model_copy(deep=True)
        changes: list[ReplanChange] = []
        trip_changes: list[TripFieldChange] = []
        affected = set(event.affected_item_ids)
        remove_ids = set(request.remove_item_ids)
        item_ids = {item.item_id for item in state.items}
        unknown_ids = (affected | remove_ids) - item_ids
        if unknown_ids:
            raise PlanningError(
                "Replan references item IDs that do not belong to this trip: "
                + ", ".join(sorted(str(item_id) for item_id in unknown_ids))
            )
        if affected & remove_ids:
            raise PlanningError(
                "An item cannot be updated and removed in the same replan request."
            )

        for index, item in enumerate(state.items):
            if item.item_id not in affected:
                continue
            before = item.model_copy(deep=True)
            after = self._apply_event(item.model_copy(deep=True), event.event_type, request.changes)
            state.items[index] = after
            changes.append(
                ReplanChange(
                    item_id=item.item_id,
                    action="cancel" if after.status == TravelItemStatus.CANCELLED else "update",
                    before=before,
                    after=after,
                    reason=event.summary,
                )
            )

        if remove_ids:
            retained_items: list[TravelItem] = []
            for item in state.items:
                if item.item_id in remove_ids:
                    changes.append(
                        ReplanChange(
                            item_id=item.item_id,
                            action="remove",
                            before=item.model_copy(deep=True),
                            reason=event.summary,
                        )
                    )
                else:
                    retained_items.append(item)
            state.items = retained_items

        for raw_item in request.add_items:
            item = raw_item.model_copy(update={"status": TravelItemStatus.PLANNED})
            state.items.append(item)
            changes.append(
                ReplanChange(
                    item_id=item.item_id,
                    action="add",
                    after=item.model_copy(deep=True),
                    reason=event.summary,
                )
            )

        if request.trip_changes:
            state, trip_changes = self._apply_trip_changes(
                state, request.trip_changes, event.summary
            )

        if not changes and not trip_changes:
            raise PlanningError(
                "The replan request contains no supported trip or itinerary changes."
            )

        state = TripState.model_validate(state.model_dump(mode="python"))
        state.items.sort(
            key=lambda item: item.start_at
            or datetime.max.replace(tzinfo=ZoneInfo(state.timezone))
        )
        event.status = EventStatus.ACKNOWLEDGED
        state.version = current.version + 1
        state.updated_at = datetime.now(current.updated_at.tzinfo)
        if event.event_id not in state.active_event_ids:
            state.active_event_ids.append(event.event_id)
        proposal = ReplanProposal(
            trip_id=current.trip_id,
            user_id=request.user_id,
            base_version=current.version,
            proposed_state=state,
            event=event,
            changes=changes,
            trip_changes=trip_changes,
            conflicts=self.detect_conflicts(state.items),
        )
        return self.repository.save_proposal(
            proposal,
            idempotency_key=idempotency_key,
        )

    @staticmethod
    def _apply_trip_changes(
        state: TripState,
        requested: dict[str, Any],
        reason: str,
    ) -> tuple[TripState, list[TripFieldChange]]:
        allowed = {
            "title",
            "timezone",
            "traveler_count",
            "destinations",
            "budget",
            "preferences",
        }
        unsupported = set(requested) - allowed
        if unsupported:
            raise PlanningError(
                "Unsupported trip-level changes: " + ", ".join(sorted(unsupported))
            )

        payload = state.model_dump(mode="python")
        field_changes: list[TripFieldChange] = []
        for field, value in requested.items():
            before = state.model_dump(mode="json").get(field)
            if field == "preferences":
                if not isinstance(value, dict):
                    raise PlanningError("preferences must be an object")
                value = {**state.preferences, **value}
            payload[field] = deepcopy(value)

            candidate = TripState.model_validate(payload)
            after = candidate.model_dump(mode="json").get(field)
            if before != after:
                field_changes.append(
                    TripFieldChange(
                        field=field,
                        before=before,
                        after=after,
                        reason=reason,
                    )
                )
            state = candidate
            payload = state.model_dump(mode="python")

        return state, field_changes

    def confirm_replan(
        self,
        user_id: str,
        trip_id: UUID,
        proposal_id: UUID,
        *,
        idempotency_key: str | None = None,
    ) -> TripState:
        if idempotency_key:
            get_cached = getattr(self.repository, "get_idempotent_trip", None)
            if get_cached is not None:
                existing = get_cached(idempotency_key)
                if existing is not None:
                    return existing
        proposal = self.repository.get_proposal(user_id, trip_id, proposal_id)
        if any(conflict.blocking for conflict in proposal.conflicts):
            raise ProposalConflictError(
                "Replan proposal contains blocking conflicts and cannot be confirmed."
            )
        return self.repository.confirm_proposal(
            user_id,
            trip_id,
            proposal_id,
            idempotency_key=idempotency_key,
        )

    #时间重叠检测
    @staticmethod
    def detect_conflicts(items: list[TravelItem]) -> list[PlanningConflict]:
        scheduled = sorted(
            (
                item for item in items
                if item.start_at and item.end_at and item.status != TravelItemStatus.CANCELLED
                and item.kind != TravelItemKind.HOTEL
            ),
            key=lambda item: item.start_at,
        )
        conflicts: list[PlanningConflict] = []
        for previous, current in zip(scheduled, scheduled[1:]):
            if current.start_at < previous.end_at:
                conflicts.append(
                    PlanningConflict(
                        kind=PlanningConflictKind.TIME_OVERLAP,
                        summary=f"{previous.title} overlaps {current.title}.",
                        item_ids=[previous.item_id, current.item_id],
                    )
                )
        return conflicts

    def _item_from_document(
        self, document: ExtractedDocument, zone: ZoneInfo, warnings: list[str]
    ) -> TravelItem | None:
        kind = _KIND_BY_DOCUMENT.get(document.document_type)
        if kind is None or document.status == ExtractionStatus.FAILED:
            return None
        fields = {
            field.path.lower(): field
            for field in document.fields
            if not field.requires_confirmation
        }
        skipped = [field.path for field in document.fields if field.requires_confirmation]
        if skipped:
            warnings.append(
                f"Document {document.document_id} skipped unconfirmed fields: {', '.join(skipped)}"
            )
        start_field = self._find(fields, _START_SUFFIXES)
        end_field = self._find(fields, _END_SUFFIXES)
        start_at = self._datetime(start_field.value, zone) if start_field else None
        end_at = self._datetime(end_field.value, zone, end_of_day=True) if end_field else None
        if (start_at is None) != (end_at is None):
            warnings.append(f"Document {document.document_id} has an incomplete schedule.")
            start_at = end_at = None
        title_field = self._find(fields, _TITLE_SUFFIXES)
        reference_field = self._find(fields, _REFERENCE_SUFFIXES)
        origin_field = self._find(fields, _ORIGIN_SUFFIXES)
        destination_field = self._find(fields, _DESTINATION_SUFFIXES)
        cost_field = self._find(fields, _COST_SUFFIXES)
        reference = str(reference_field.value) if reference_field else None
        title = str(title_field.value) if title_field else self._default_title(kind, reference)
        used_fields = [
            field for field in (
                start_field, end_field, title_field, reference_field,
                origin_field, destination_field, cost_field,
            ) if field is not None
        ]
        evidence_ids = list(dict.fromkeys(
            evidence_id for field in used_fields for evidence_id in field.evidence_ids
        ))
        return TravelItem(
            kind=kind,
            title=title,
            status=TravelItemStatus.CONFIRMED,
            start_at=start_at,
            end_at=end_at,
            origin=Location(name=str(origin_field.value)) if origin_field else None,
            destination=Location(name=str(destination_field.value)) if destination_field else None,
            booking_reference=reference,
            cost=self._money(cost_field.value) if cost_field else None,
            evidence_ids=evidence_ids,
            details={"document_id": str(document.document_id), "attachment_id": str(document.attachment_id)},
        )

    @staticmethod
    def _find(fields: dict[str, ExtractedField], suffixes: tuple[str, ...]) -> ExtractedField | None:
        return next(
            (field for path, field in fields.items() if any(path.endswith(suffix) for suffix in suffixes)),
            None,
        )

    @staticmethod
    def _datetime(value: Any, zone: ZoneInfo, *, end_of_day: bool = False) -> datetime | None:
        if not isinstance(value, str):
            return None
        text = value.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            try:
                parsed_date = date.fromisoformat(text)
            except ValueError:
                return None
            parsed = datetime.combine(parsed_date, time.min, zone)
            if end_of_day:
                parsed += timedelta(days=1)
        return parsed.replace(tzinfo=zone) if parsed.tzinfo is None else parsed

    @staticmethod
    def _money(value: Any) -> Money | None:
        try:
            amount = Decimal(str(value).replace("¥", "").replace(",", "").strip())
            return Money(amount=amount)
        except (InvalidOperation, ValueError):
            return None

    @staticmethod
    def _default_title(kind: TravelItemKind, reference: str | None) -> str:
        labels = {TravelItemKind.FLIGHT: "航班", TravelItemKind.TRAIN: "列车", TravelItemKind.HOTEL: "酒店", TravelItemKind.ATTRACTION: "景点"}
        return f"{labels[kind]} {reference}" if reference else labels[kind]

    @staticmethod
    def _apply_event(item: TravelItem, event_type: TravelEventType, changes: dict[str, Any]) -> TravelItem:
        if event_type == TravelEventType.CANCELLATION:
            item.status = TravelItemStatus.CANCELLED
            return item
        allowed = {"title", "start_at", "end_at", "origin", "destination", "booking_reference"}
        updates = {key: deepcopy(value) for key, value in changes.items() if key in allowed}
        if not updates:
            raise PlanningError("This event requires at least one supported item change.")
        payload = item.model_dump(mode="python")
        payload.update(updates)
        payload["status"] = TravelItemStatus.CHANGED
        return TravelItem.model_validate(payload)
