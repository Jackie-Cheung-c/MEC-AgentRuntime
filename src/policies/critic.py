"""Second-pass quality review for TravelTrace answers."""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from src.prompts.prompt_manager import prompt_manager
from langchain_core.messages import HumanMessage, SystemMessage


@dataclass(frozen=True)
class CriticResult:
    approved: bool
    response: str
    reason: str = ""


class CriticRunResult(str):
    """String-compatible answer carrying retry metadata for evaluation."""

    retries_used: int

    def __new__(cls, value: str, retries_used: int) -> "CriticRunResult":
        instance = super().__new__(cls, value)
        instance.retries_used = retries_used
        return instance


CRITIC_PROMPT = prompt_manager.get("critic_prompt")
MAX_AGENT_RETRIES = 3

def _content_text(value: object) -> str:
    """从 LangChain 的响应内容中提取文本"""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            block if isinstance(block, str) else str(block.get("text", ""))
            for block in value
            if isinstance(block, (str, dict))
        )
    return str(value or "")


def _parse_verdict(content: object) -> dict[str, Any]:
    """Parse plain or fenced JSON returned by an OpenAI-compatible model."""
    text = _content_text(content).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    return json.loads(text)


async def _review_once(
    model,
    user_input: str,
    candidate_answer: str,
    execution_evidence: dict | None = None,
) -> CriticResult:
    """Approve an answer or return one critic-authored revision.

    A malformed critic response falls back to the original answer so the review
    step never turns a successful primary answer into a failed request.
    """
    if not candidate_answer.strip():
        return CriticResult(approved=True, response=candidate_answer)

    try:
        history_evidence = collect_tool_evidence(messages)
        response = await model.ainvoke(
            [
                SystemMessage(content=CRITIC_PROMPT),
                HumanMessage(
                    content=(
                        f"用户问题：\n{user_input}\n\n"
                        f"候选回答：\n{candidate_answer}\n\n"
                        "主 Agent 本轮执行证据（Skill 选择和工具调用记录）：\n"
                        + json.dumps(execution_evidence or {}, ensure_ascii=False, default=str)
                        + "\n\n=== 历史轮次已获取证据（与本轮同等有效，可用于核验） ===\n"
                        + json.dumps(history_evidence, ensure_ascii=False, default=str)
                    )
                ),
            ]
        )
        verdict = _parse_verdict(response.content)
        approved = bool(verdict["approved"])
        reason = str(verdict.get("reason", ""))
        revised_answer = str(verdict.get("revised_answer", "")).strip()
        if not approved and revised_answer:
            return CriticResult(False, revised_answer, reason)
        return CriticResult(True, candidate_answer, reason)
    except Exception as error:
        reason = f"Critic 服务不可用 ({type(error).__name__}: {str(error)[:300]})"
        print(f"[critic] {reason}")
        return CriticResult(True, candidate_answer, reason)


def _evidence_payload(execution_evidence: Any) -> dict[str, Any]:
    if isinstance(execution_evidence, Mapping):
        return dict(execution_evidence)
    model_dump = getattr(execution_evidence, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    return {"value": str(execution_evidence)}


async def generate_with_critic_retry(
    invoke_agent: Callable[[list[dict[str, str]]], Awaitable[tuple[str, Any]]],
    critic_model,
    user_input: str,
    messages: list[dict[str, str]],
    selected_skill: dict[str, str] | None = None,
    attachment_evidence: dict[str, Any] | None = None,
    max_retries: int = MAX_AGENT_RETRIES,
    on_critic_result: Callable[[int, CriticResult, dict[str, Any]], None] | None = None,
) -> CriticRunResult:
    """Run the complete generation, review, and retry workflow."""
    retry_messages = list(messages)
    for attempt in range(max_retries + 1):
        message_agent_answer, evidence = await invoke_agent(retry_messages)
        print(f"[debug] agent ouput:{message_agent_answer}")
        review_evidence = _evidence_payload(evidence)
        review_evidence["selected_skill"] = selected_skill or None
        if attachment_evidence:
            review_evidence.update(attachment_evidence)
        critic_result = await _review_once(
            critic_model, user_input, message_agent_answer, review_evidence
        )
        if on_critic_result:
            print(f"[debug] critic: 审查结果, reason:{critic_result.reason}")
            on_critic_result(attempt, critic_result, review_evidence)

        if critic_result.approved:
            # print(f"[debug] critic: 处理完毕, reason:{critic_result.reason}")
            return CriticRunResult(message_agent_answer, retries_used=attempt)
        if attempt == max_retries:
            print(f"[debug] critic: 已超过最大重试次数")
            return CriticRunResult(critic_result.response, retries_used=attempt)

        # The first attempt is checkpointed, so retries only add review feedback.
        retry_messages = [
            {
                "role": "system",
                "content": (
                    "上一版回答未通过质量审查。请根据以下审查原因重新生成完整答案；"
                    "需要时可以重新调用工具，但不得编造未验证信息。\n"
                    f"审查原因：{critic_result.reason}"
                ),
            }
        ]

    return CriticRunResult("", retries_used=max_retries)
