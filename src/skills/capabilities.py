"""Skill-owned UI capability metadata."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class QuickPrompt(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str = Field(min_length=1, max_length=80)
    prompt: str = Field(min_length=1, max_length=4000)
    description: str = Field(default="", max_length=200)


class ToolEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    capability_id: str = Field(
        pattern=r"^[a-z][a-z0-9_.-]*$",
        max_length=100,
    )
    title: str = Field(min_length=1, max_length=80)
    prompt: str = Field(min_length=1, max_length=4000)
    description: str = Field(default="", max_length=200)


class SkillCapabilitiesMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quick_prompts: list[QuickPrompt] = Field(
        default_factory=list,
        max_length=12,
    )
    tool_entries: list[ToolEntry] = Field(
        default_factory=list,
        max_length=24,
    )


def load_skill_capabilities(
    skill: Mapping[str, Any],
) -> SkillCapabilitiesMetadata:
    """Load optional UI metadata owned by one Skill."""
    reference = str(skill.get("capabilities_metadata") or "").strip()
    if not reference:
        return SkillCapabilitiesMetadata()

    skill_dir = Path(str(skill["path"])).parent.resolve()
    metadata_path = (skill_dir / reference).resolve()
    if metadata_path.parent != skill_dir:
        raise ValueError(
            "Skill capabilities metadata must live in its Skill directory."
        )
    return SkillCapabilitiesMetadata.model_validate_json(
        metadata_path.read_text(encoding="utf-8")
    )
