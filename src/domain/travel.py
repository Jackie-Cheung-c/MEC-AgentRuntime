"""旅行状态及异常事件的数据契约规范"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, JsonValue, model_validator

from src.domain.base import DomainModel


class Money(DomainModel):
    amount: Decimal = Field(ge=0, max_digits=14, decimal_places=2)
    currency: str = Field(default="CNY", pattern=r"^[A-Z]{3}$")


class Location(DomainModel):
    name: str = Field(min_length=1, max_length=255)
    address: str | None = Field(default=None, max_length=500)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    provider_place_id: str | None = Field(default=None, max_length=255)

    @model_validator(mode="after")
    def validate_coordinates(self) -> "Location":
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("latitude and longitude must be provided together")
        return self


class TravelItemKind(StrEnum):
    FLIGHT = "flight"
    TRAIN = "train"
    HOTEL = "hotel"
    ATTRACTION = "attraction"
    TRANSFER = "transfer"
    ACTIVITY = "activity"
    OTHER = "other"

# 用户端状态
class TravelItemStatus(StrEnum):
    PLANNED = "planned"
    CONFIRMED = "confirmed"
    CHANGED = "changed"
    CANCELLED = "cancelled"
    COMPLETED = "completed"


class TravelItem(DomainModel):
    item_id: UUID = Field(default_factory=uuid4)
    kind: TravelItemKind
    title: str = Field(min_length=1, max_length=255)
    status: TravelItemStatus = TravelItemStatus.PLANNED
    start_at: datetime | None = None
    end_at: datetime | None = None
    origin: Location | None = None
    destination: Location | None = None
    booking_reference: str | None = Field(default=None, max_length=255)
    cost: Money | None = None
    evidence_ids: list[UUID] = Field(default_factory=list)
    details: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_schedule(self) -> "TravelItem":
        if (self.start_at is None) != (self.end_at is None):
            raise ValueError("start_at and end_at must be provided together")
        if self.start_at is not None and self.end_at is not None:
            if self.start_at.tzinfo is None or self.end_at.tzinfo is None:
                raise ValueError("travel item timestamps must include a timezone")
            if self.end_at <= self.start_at:
                raise ValueError("end_at must be after start_at")
        return self

# 旅行状态
class TripState(DomainModel):
    trip_id: UUID = Field(default_factory=uuid4)
    user_id: str = Field(min_length=1, max_length=128)
    version: int = Field(default=1, ge=1) # 乐观锁，防止并发修改
    title: str | None = Field(default=None, max_length=255)
    timezone: str = "Asia/Shanghai"
    traveler_count: int = Field(default=1, ge=1, le=100)
    destinations: list[Location] = Field(default_factory=list)
    preferences: dict[str, JsonValue] = Field(default_factory=dict)
    items: list[TravelItem] = Field(default_factory=list)
    budget: Money | None = None
    spent: Money | None = None
    active_event_ids: list[UUID] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def validate_state(self) -> "TripState":
        try:
            ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as error:
            raise ValueError("timezone must be a valid IANA timezone") from error
        item_ids = [item.item_id for item in self.items]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("trip item IDs must be unique")
        if self.budget and self.spent and self.budget.currency != self.spent.currency:
            raise ValueError("budget and spent must use the same currency")
        return self


class TravelEventType(StrEnum):
    CANCELLATION = "cancellation"
    DELAY = "delay"
    SCHEDULE_CHANGE = "schedule_change"
    CLOSURE = "closure"
    SEVERE_WEATHER = "severe_weather"
    MISSING_DOCUMENT = "missing_document"
    USER_CHANGE = "user_change"
    TOOL_FAILURE = "tool_failure"
    OTHER = "other"


class EventSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class EventStatus(StrEnum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"

# 旅行端可能事件
class TravelEvent(DomainModel):
    event_id: UUID = Field(default_factory=uuid4)
    event_type: TravelEventType
    severity: EventSeverity = EventSeverity.WARNING
    status: EventStatus = EventStatus.OPEN
    summary: str = Field(min_length=1, max_length=1000)
    occurred_at: datetime | None = None
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    affected_item_ids: list[UUID] = Field(default_factory=list)
    evidence_ids: list[UUID] = Field(default_factory=list)
    details: dict[str, JsonValue] = Field(default_factory=dict)
