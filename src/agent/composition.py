"""Production dependency composition for the universal Agent."""

from __future__ import annotations

from typing import Any


def build_default_dependencies():
    """Build default dependencies, including the optional Travel adapter.

    This module is the composition boundary. Core orchestration code must not
    import Travel repositories, Travel state contracts, or Travel tools.
    """
    from src.agent.service import ServiceDependencies
    from src.attachments.factory import get_default_image_vlm_processor
    from src.domain import (
        Attachment,
        Evidence,
        ExtractedDocument,
        ExtractionPolicy,
    )
    from src.travel import ItineraryPlanningService, build_planning_tools
    from src.state_adapters.travel import TravelStateAdapter
    from src.storage.trip_store import SQLiteTripRepository
    from src.tools.catalog import load_enabled_tools

    planner = ItineraryPlanningService(SQLiteTripRepository())

    async def process_attachments(
        attachments: list[Attachment],
        *,
        user_id: str,
        extraction_policy: ExtractionPolicy,
    ) -> tuple[list[ExtractedDocument], list[Evidence]]:
        processor = get_default_image_vlm_processor()
        return await processor(
            attachments,
            user_id=user_id,
            policy=extraction_policy,
        )

    async def default_tool_provider() -> list[Any]:
        return [
            *await load_enabled_tools(),
            *build_planning_tools(planner),
        ]

    return ServiceDependencies(
        attachment_processor=process_attachments,
        tool_provider=default_tool_provider,
        state_adapters=(TravelStateAdapter(planner),),
    )
