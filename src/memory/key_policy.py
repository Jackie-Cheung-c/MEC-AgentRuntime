"""长期记忆逻辑key的基数规则"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from src.memory.models import MemoryCategory


class MemoryValueCardinality(StrEnum):
    SINGLE = "single"
    MULTI = "multi"


SINGLE_VALUE_KEYS = {
    "name",
    "home_city",
    "preferred_departure_city",
    "preferred_transport",
    "preferred_hotel_level",
    "seat_preference",
    "travel_pace",
}

MULTI_VALUE_KEYS = {
    "food_like",
    "travel_interest",
    "dietary_restriction",
    "accessibility_requirement",
    "visited_destination",
    "completed_trip",
}


def memory_value_cardinality(
    category: MemoryCategory,
    memory_key: str,
) -> MemoryValueCardinality:
    if memory_key in SINGLE_VALUE_KEYS:
        return MemoryValueCardinality.SINGLE
    if memory_key in MULTI_VALUE_KEYS:
        return MemoryValueCardinality.MULTI
    if category == MemoryCategory.CONSTRAINT:
        return MemoryValueCardinality.MULTI
    return MemoryValueCardinality.SINGLE


def merge_memory_values(current: Any, incoming: Any) -> list[Any]:
    """Merge multi-value content without losing earlier values."""
    values: list[Any] = []
    for source in (current, incoming):
        if source is None:
            continue
        items = source if isinstance(source, list) else [source]
        for item in items:
            if item not in values:
                values.append(item)
    return values
