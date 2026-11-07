"""Security scanning and grounding for untrusted evidence."""

from __future__ import annotations

import ipaddress
import re
import unicodedata
from dataclasses import dataclass
from urllib.parse import urlparse

from src.domain.evidence import (
    Evidence,
    EvidenceSafety,
    GroundingStatus,
)


_PROMPT_INJECTION_PATTERNS = (
    re.compile(
        r"(?is)\b(?:ignore|disregard|forget)\s+"
        r"(?:all\s+)?(?:previous|prior|above)\s+instructions?\b"
    ),
    re.compile(
        r"(?is)(?:忽略|无视|忘记|覆盖).{0,20}"
        r"(?:之前|以上|前面|系统).{0,20}(?:指令|提示|规则)"
    ),
    re.compile(r"(?is)\b(?:system|developer)\s+(?:prompt|message)\b"),
)

_SCRIPT_PATTERN = re.compile(
    r"(?is)<\s*(?:script|iframe|object|embed|svg)\b.*?>.*?"
    r"<\s*/\s*(?:script|iframe|object|embed|svg)\s*>"
    r"|<\s*(?:script|iframe|object|embed|svg)\b[^>]*?/?>"
)

_URL_PATTERN = re.compile(
    r"(?i)\b(?:https?|ftp|file|javascript|data|vbscript):"
    r"[^\s<>'\"]+"
)

_SECRET_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{12,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(
        r"(?i)\b(?:password|passwd|secret|api[_-]?key|access[_-]?token)"
        r"\s*[:=]\s*\S+"
    ),
)

_PII_PATTERNS = (
    re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
    re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    re.compile(r"\b\d{17}[\dXx]\b"),
)

_HIGH_RISK_TYPES = {
    "prompt_injection",
    "script",
    "unsafe_url",
    "secret",
}


@dataclass(frozen=True, slots=True)
class EvidenceScanResult:
    status: EvidenceSafety
    risk_types: tuple[str, ...]
    redacted_content: str


def _normalized(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "")


def _is_unsafe_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return True

    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        return True

    host = (parsed.hostname or "").lower()
    if not host or host in {"localhost", "localhost.localdomain"}:
        return True
    if host.endswith(".local"):
        return True

    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False

    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
    )


def _replace_patterns(
    text: str,
    patterns: tuple[re.Pattern[str], ...],
    replacement: str,
) -> str:
    for pattern in patterns:
        text = pattern.sub(replacement, text)
    return text


def scan_text(text: str) -> EvidenceScanResult:
    """Heuristic scan. This is a guard, not a security boundary."""
    original = text or ""
    normalized = _normalized(original)
    risks: set[str] = set()
    redacted = original

    for pattern in _PROMPT_INJECTION_PATTERNS:
        if pattern.search(normalized):
            risks.add("prompt_injection")
        redacted = pattern.sub("[REDACTED_INSTRUCTION]", redacted)

    if _SCRIPT_PATTERN.search(normalized):
        risks.add("script")
        redacted = _SCRIPT_PATTERN.sub("[REDACTED_SCRIPT]", redacted)

    for match in list(_URL_PATTERN.finditer(redacted)):
        if _is_unsafe_url(match.group(0)):
            risks.add("unsafe_url")
            redacted = redacted.replace(match.group(0), "[REDACTED_URL]")

    for pattern in _SECRET_PATTERNS:
        if pattern.search(normalized):
            risks.add("secret")
        redacted = pattern.sub("[REDACTED_SECRET]", redacted)

    for pattern in _PII_PATTERNS:
        if pattern.search(normalized):
            risks.add("pii")
        redacted = pattern.sub("[REDACTED_PII]", redacted)

    if risks & _HIGH_RISK_TYPES:
        status = EvidenceSafety.QUARANTINED
    elif risks:
        status = EvidenceSafety.SUSPICIOUS
    else:
        status = EvidenceSafety.CLEAN

    return EvidenceScanResult(
        status=status,
        risk_types=tuple(sorted(risks)),
        redacted_content=redacted,
    )


def combine_scan_results(
    *results: EvidenceScanResult,
) -> EvidenceScanResult:
    risk_types = sorted(
        {
            risk
            for result in results
            for risk in result.risk_types
        }
    )
    if any(
        result.status == EvidenceSafety.QUARANTINED
        for result in results
    ):
        status = EvidenceSafety.QUARANTINED
    elif any(
        result.status == EvidenceSafety.SUSPICIOUS
        for result in results
    ):
        status = EvidenceSafety.SUSPICIOUS
    else:
        status = EvidenceSafety.CLEAN

    return EvidenceScanResult(
        status=status,
        risk_types=tuple(risk_types),
        redacted_content=" ".join(
            result.redacted_content
            for result in results
            if result.redacted_content
        ),
    )


def normalize_for_match(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"\s+", "", text)


def grounding_status_for(
    evidence_text: str,
    raw_text: str,
    field_value: object = None,
) -> GroundingStatus:
    """Check the VLM field against its raw transcription.

    MODEL_TRANSCRIPT_MATCH means it matches the VLM transcript.
    It does not mean it has been independently verified by OCR.
    """
    quote = normalize_for_match(evidence_text)
    source = normalize_for_match(raw_text)
    value = normalize_for_match(field_value)

    if quote and quote in source:
        return GroundingStatus.MODEL_TRANSCRIPT_MATCH

    if quote and value and value in quote and value in source:
        return GroundingStatus.MODEL_TRANSCRIPT_MATCH

    return GroundingStatus.UNGROUNDED


def safe_evidence_content(evidence: Evidence) -> str:
    return evidence.redacted_content or evidence.content


def is_model_safe(evidence: Evidence) -> bool:
    return (
        evidence.safety_status == EvidenceSafety.CLEAN
        and evidence.grounding_status
        in {
            GroundingStatus.NOT_APPLICABLE,
            GroundingStatus.MODEL_TRANSCRIPT_MATCH,
        }
    )