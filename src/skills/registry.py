"""Metadata-first discovery and on-demand loading for application skills."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.domain import ExtractionPolicy, general_extraction_policy


SKILLS_DIR = Path(__file__).resolve().parent

FRONTMATTER_RE = re.compile(
    r"^---\s*\n(.*?)\n---\s*(?:\n|$)",
    re.DOTALL,
)
LIST_FIELDS = {
    "tools",
    "capabilities",
}
INT_FIELDS = {
    "priority",
}

def _strip_quotes(value: str) -> str:
    return value.strip().strip("\"'")

def _parse_list(value: str) -> list[str]:
    text = value.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]

    if text.casefold() in {"", "none", "null", "[]"}:
        return []

    return [
        _strip_quotes(item)
        for item in re.split(r"[,;，]", text)
        if _strip_quotes(item)
    ]

def _parse_frontmatter(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    metadata: dict[str, Any] = {
        "name": path.parent.name,
        "description": "",
        "domain": "general",
        "state_adapter": None,
        "extraction_policy": None,
        "capabilities_metadata": None,
        "tools": [],
        "capabilities": [],
        "priority": 0,
    }
    match = FRONTMATTER_RE.match(text)
    if not match:
        return metadata
    for line in match.group(1).splitlines():
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        key = key.strip()
        value = raw_value.strip()

        if key in LIST_FIELDS:
            metadata[key] = _parse_list(value)
            continue
        if key in INT_FIELDS:
            try:
                metadata[key] = int(value)
            except ValueError:
                metadata[key] = 0
            continue

        if key == "state_adapter":
            normalized = _strip_quotes(value)
            metadata[key] = (
                None
                if normalized.casefold() in {"", "none", "null"}
                else normalized
            )
            continue

        metadata[key] = _strip_quotes(value)
    if not str(metadata.get("domain", "")).strip():
        metadata["domain"] = "general"

    return metadata


def discover_skills(
    *,
    domain: str | None = None,
    state_adapter: str | None = None,
) -> list[dict[str, Any]]:
    """Discover all global skills and filter by optional metadata."""
    skills: list[dict[str, Any]] = []
    for path in sorted(SKILLS_DIR.glob("*/SKILL.md")):
        metadata = _parse_frontmatter(path)
        metadata["path"] = str(path)

        if domain is not None:
            if str(metadata.get("domain", "general")) != domain:
                continue

        if state_adapter is not None:
            skill_adapter = metadata.get("state_adapter")
            if skill_adapter not in {None, "", state_adapter}:
                continue
        skills.append(metadata)

        skills.sort(
            key=lambda item: (
                -int(item.get("priority", 0)),
                str(item.get("name", "")),
            )
        )
    return skills

def _format_skill_catalog(skills: list[dict[str, Any]]) -> str:
    lines = []

    for skill in skills:
        capabilities = ", ".join(skill.get("capabilities", []))
        state_adapter = skill.get("state_adapter") or "none"

        lines.append(
            "- "
            f"{skill['name']} | "
            f"domain={skill.get('domain', 'general')} | "
            f"state_adapter={state_adapter} | "
            f"capabilities={capabilities or 'none'} | "
            f"priority={skill.get('priority', 0)} | "
            f"description={skill.get('description', '')}"
        )

    return "\n".join(lines)


async def select_skill(
    model, 
    user_input: str,
    *,
    domain: str | None = None,
    state_adapter: str | None = None,
) -> dict[str, Any] | None:
    """Choose one global Skill by metadata, or return None."""
    skills = discover_skills(
        domain=domain,
        state_adapter=state_adapter,
    )
    if not skills:
        return None

    response = await model.ainvoke(
        [
            SystemMessage(
                content=(
                    "你是 Skill 路由器。只能输出下方一个 Skill 的 name，"
                    "或者输出 NONE。根据用户任务、Skill 描述、domain、"
                    "capabilities 和 priority 选择最匹配的 Skill。"
                    "priority 较高表示更具体的专用 Skill。"
                    "不得输出解释、标点或 Markdown。\n\n"
                    + _format_skill_catalog(skills)
                )
            ),
            HumanMessage(content=user_input),
        ]
    )
    text = str(response.content).strip()
    selected_name = text.splitlines()[0].strip().strip("`\"'")
    if selected_name.casefold() == "none":
        return None
    return next(
        (
            skill for skill in skills
            if str(skill.get("name", "")).casefold() == selected_name.casefold()
        ),
        None,
    )


def get_skill(name: str) -> dict[str, Any] | None:
    """Return one discovered Skill by name using a case-insensitive lookup."""
    normalized = name.strip().casefold()
    return next(
        (
            skill
            for skill in discover_skills()
            if str(skill.get("name", "")).casefold() == normalized
        ),
        None,
    )


def get_skill_by_state_adapter(adapter: str) -> dict[str, Any] | None:
    """Return the Skill that declares one state-adapter workflow."""
    normalized = adapter.strip().casefold()
    return next(
        (
            skill
            for skill in discover_skills()
            if str(skill.get("state_adapter") or "").casefold()
            == normalized
        ),
        None,
    )


def resolve_skill_extraction_policy(
    skill: Mapping[str, Any] | None,
) -> ExtractionPolicy:
    """Load a Skill-owned extraction policy, or use the general fallback."""
    if skill is None:
        return general_extraction_policy()

    reference = str(skill.get("extraction_policy") or "").strip()
    if not reference:
        return general_extraction_policy()

    policy_path = (Path(str(skill["path"])).parent / reference).resolve()
    if not policy_path.is_relative_to(SKILLS_DIR.resolve()):
        raise ValueError("Extraction policy must live under the skills directory.")
    return ExtractionPolicy.model_validate_json(
        policy_path.read_text(encoding="utf-8")
    )


def load_skill(skill: Mapping[str, Any]) -> str:
    """仅当元数据匹配时才加载技能的完整指令集"""
    path = Path(str(skill["path"]))
    text = path.read_text(encoding="utf-8")
    match = FRONTMATTER_RE.match(text)

    if match is None:
        return text
    return text[match.end():].lstrip()
