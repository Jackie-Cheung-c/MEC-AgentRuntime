"""SQLite FTS5和长期记忆索引共享的Text tokenization"""

from __future__ import annotations

import re
import unicodedata

try:
    import jieba
except ImportError:  # pragma: no cover - exercised without optional jieba
    jieba = None


_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_ENGLISH_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_ENGLISH_PART_RE = re.compile(r"[._-]+|(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])")

_TRAVEL_TERMS = (
    "高铁",
    "航班",
    "机票",
    "火车票",
    "酒店",
    "民宿",
    "行程",
    "预算",
    "退改签",
    "中转",
    "自由行",
    "跟团游",
    "入住",
    "退房",
    "旅行规划",
)

if jieba is not None:
    for term in _TRAVEL_TERMS:
        jieba.add_word(term)


def _english_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for raw_token in _ENGLISH_TOKEN_RE.findall(text):
        token = raw_token.casefold()
        tokens.append(token)
        camel_split = _CAMEL_BOUNDARY_RE.sub(" ", raw_token).casefold()
        parts: list[str] = []
        for chunk in camel_split.split():
            parts.extend(
                part for part in _ENGLISH_PART_RE.split(chunk) if part
            )
        if len(parts) > 1:
            tokens.extend(parts)
            joined = "".join(parts)
            if joined != token:
                tokens.append(joined)
    return tokens


def _cjk_tokens(text: str) -> list[str]:
    if jieba is not None:
        return [token for token in jieba.cut_for_search(text) if token.strip()]
    return list(_CJK_RE.findall(text))


def tokenize(text: str) -> list[str]:
    """Return stable search terms for English and Chinese text."""
    normalized = unicodedata.normalize("NFKC", text)
    tokens: list[str] = []
    cursor = 0
    for match in _CJK_RE.finditer(normalized):
        tokens.extend(_english_tokens(normalized[cursor : match.start()]))
        tokens.extend(_cjk_tokens(match.group()))
        cursor = match.end()
    tokens.extend(_english_tokens(normalized[cursor:]))
    return list(dict.fromkeys(token for token in tokens if token.strip()))


def fts_match_query(text: str) -> str:
    """Build an OR query from already-tokenized search terms."""
    terms = []
    for term in tokenize(text):
        escaped = term.replace('"', '""')
        terms.append(f'"{escaped}"')
    return " OR ".join(terms)
