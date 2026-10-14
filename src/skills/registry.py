"""Metadata-first discovery and on-demand loading for application skills."""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage


SKILLS_DIR = Path(__file__).resolve().parent


def _parse_frontmatter(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    metadata = {"name": path.parent.name, "description": ""}
    if not match:
        return metadata
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip("\"'")
    return metadata


def discover_skills() -> list[dict[str, str]]:
    """Read only frontmatter metadata, never full skill instructions."""
    skills = []
    for path in sorted(SKILLS_DIR.glob("*/SKILL.md")):
        metadata = _parse_frontmatter(path)
        metadata["path"] = str(path)
        skills.append(metadata)
    return skills


async def select_skill(model, user_input: str) -> dict[str, str] | None:
    """Choose one skill from frontmatter metadata, or choose NONE."""
    skills = discover_skills()
    if not skills:
        return None
    catalog = "\n".join(
        f"- {skill['name']}: {skill.get('description', '')}" for skill in skills
    )
    response = await model.ainvoke(
        [
            SystemMessage(
                content=(
                    "你是 Skill 路由器。只能输出下方一个 Skill 名称或 NONE。"
                    "根据描述选择最匹配的 Skill；单项查询不选择完整规划 Skill；不要解释。\n"
                    + catalog
                )
            ),
            HumanMessage(content=user_input),
        ]
    )
    selected_name = str(response.content).strip().strip("`")
    return next((skill for skill in skills if skill["name"] == selected_name), None)


def load_skill(skill: dict[str, str]) -> str:
    """仅当元数据匹配时才加载技能的完整指令集"""
    return Path(skill["path"]).read_text(encoding="utf-8")
