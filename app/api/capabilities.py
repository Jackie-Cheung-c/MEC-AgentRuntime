"""Metadata-driven capability catalog for the generic workspace UI."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from src.domain import ExtractionDocumentType, general_extraction_policy
from src.skills.capabilities import load_skill_capabilities
from src.skills.registry import (
    discover_skills,
    resolve_skill_extraction_policy,
)
from src.tools.catalog import capability_manifest


router = APIRouter(prefix="/api", tags=["capabilities"])


def _document_type_payload(
    item: ExtractionDocumentType,
    *,
    skill: str | None = None,
    domain: str | None = None,
) -> dict[str, Any]:
    return {
        "id": item.id,
        "label": item.label or item.id,
        "description": item.description,
        "skill": skill,
        "domain": domain,
    }


def _deduplicate(
    items: list[dict[str, Any]],
    *,
    key: str,
) -> list[dict[str, Any]]:
    seen: set[Any] = set()
    result: list[dict[str, Any]] = []
    for item in items:
        value = item.get(key)
        if value in seen:
            continue
        seen.add(value)
        result.append(item)
    return result


@router.get("/capabilities")
def get_capabilities() -> dict[str, Any]:
    """Return Skill metadata, extraction types, and registered tools."""
    tools = capability_manifest()
    tools_by_id = {
        item["capability_id"]: item
        for item in tools
    }
    defaults = general_extraction_policy()
    document_types = [
        _document_type_payload(item, domain="general")
        for item in defaults.document_types
    ]
    quick_prompts: list[dict[str, Any]] = []
    tool_entries: list[dict[str, Any]] = []
    skills: list[dict[str, Any]] = []

    for skill in discover_skills():
        name = str(skill.get("name") or "")
        domain = str(skill.get("domain") or "general")
        policy = None
        if str(skill.get("extraction_policy") or "").strip():
            policy = resolve_skill_extraction_policy(skill)
        ui_metadata = load_skill_capabilities(skill)

        skill_document_types = (
            [
                _document_type_payload(
                    item,
                    skill=name,
                    domain=domain,
                )
                for item in policy.document_types
            ]
            if policy is not None
            else []
        )
        skill_quick_prompts = [
            {
                **item.model_dump(),
                "skill": name,
                "domain": domain,
            }
            for item in ui_metadata.quick_prompts
        ]
        skill_tool_entries = []
        for item in ui_metadata.tool_entries:
            manifest = tools_by_id.get(item.capability_id)
            skill_tool_entries.append(
                {
                    **item.model_dump(),
                    "skill": name,
                    "domain": domain,
                    "available": (
                        bool(manifest.get("available"))
                        if manifest is not None
                        else True
                    ),
                    "unavailable_reason": (
                        manifest.get("unavailable_reason")
                        if manifest is not None
                        else None
                    ),
                }
            )

        document_types.extend(skill_document_types)
        quick_prompts.extend(skill_quick_prompts)
        tool_entries.extend(skill_tool_entries)
        skills.append(
            {
                "name": name,
                "description": skill.get("description", ""),
                "domain": domain,
                "state_adapter": skill.get("state_adapter"),
                "capabilities": list(skill.get("capabilities", [])),
                "tools": list(skill.get("tools", [])),
                "priority": int(skill.get("priority", 0)),
                "document_types": skill_document_types,
                "quick_prompts": skill_quick_prompts,
                "tool_entries": skill_tool_entries,
            }
        )

    return {
        "schema_version": 1,
        "skills": skills,
        "document_types": _deduplicate(document_types, key="id"),
        "quick_prompts": _deduplicate(quick_prompts, key="prompt"),
        "tool_entries": _deduplicate(
            tool_entries,
            key="capability_id",
        ),
        "tools": tools,
    }
