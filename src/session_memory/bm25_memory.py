"""BM25 retrieval over persisted Chroma documents with a context budget."""

from __future__ import annotations

import math
import logging
import re
import unicodedata
from collections import Counter

from src.session_memory.memory import memory_store

try:
    import jieba
except ImportError:  # pragma: no cover - exercised when jieba is not installed
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
    "行程单",
    "景点",
    "门票",
    "预算",
    "退改签",
    "中转",
    "自由行",
    "跟团游",
    "入住",
    "退房",
    "接机",
    "送机",
    "旅行规划",
    "tripstate",
    "mcp",
)

if jieba is not None:
    jieba.setLogLevel(logging.WARNING)
    for term in _TRAVEL_TERMS:
        jieba.add_word(term)

logger = logging.getLogger(__name__)


def _english_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for raw_token in _ENGLISH_TOKEN_RE.findall(text):
        token = raw_token.casefold()
        tokens.append(token)

        parts: list[str] = []
        camel_split = _CAMEL_BOUNDARY_RE.sub(" ", raw_token).casefold()
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
    # Keep the previous behavior available when the optional dependency is absent.
    return re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", text)


def _tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text)

    tokens: list[str] = []
    cursor = 0
    for match in _CJK_RE.finditer(normalized):
        tokens.extend(_english_tokens(normalized[cursor : match.start()]))
        tokens.extend(_cjk_tokens(match.group()))
        cursor = match.end()
    tokens.extend(_english_tokens(normalized[cursor:]))
    return tokens


def _rank_bm25_documents(
    documents: list[str],
    query: str,
    limit: int,
) -> list[int]:
    """Return document indexes ordered by BM25, excluding zero-score matches."""
    query_terms = list(dict.fromkeys(_tokens(query)))
    if not documents or not query_terms:
        return []

    tokenized = [Counter(_tokens(document)) for document in documents]
    document_count = len(documents)

    df = {
        term: sum(term in frequencies for frequencies in tokenized)
        for term in set(query_terms)
    }

    average_length = (
        sum(sum(frequencies.values()) for frequencies in tokenized)
        / document_count
    ) or 1

    ranked: list[tuple[float, int]] = []
    for index, frequencies in enumerate(tokenized):
        score = 0.0
        document_length = sum(frequencies.values())

        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue

            idf = math.log(
                1
                + (document_count - df[term] + 0.5)
                / (df[term] + 0.5)
            )
            norm = frequency + 1.2 * (
                1 - 0.75 + 0.75 * document_length / average_length
            )
            score += idf * frequency * 2.2 / norm

        if score > 0:
            ranked.append((score, index))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [index for _, index in ranked[:limit]]


# def retrieve_bm25_memories(
#     user_id: str,
#     thread_id: str,
#     query: str,
#     limit: int = 2,
#     char_budget: int = 1200,
# ) -> str:
#     records = memory_store.get(
#         where={"user_id": user_id},
#         include=["documents", "metadatas"],
#     )

#     documents = [
#         document
#         for document, metadata in zip(
#             records.get("documents", []),
#             records.get("metadatas", []),
#         )
#         if document
#         and metadata
#         and metadata.get("thread_id") == thread_id
#     ]

#     ranked_indexes = _rank_bm25_documents(documents, query, limit)
#     if not ranked_indexes:
#         return ""

#     remaining = char_budget
#     fragments: list[str] = []

#     for index, document_index in enumerate(ranked_indexes, start=1):
#         document = documents[document_index]
#         content = document[:remaining]
#         if not content:
#             break

#         fragments.append(f"当前会话历史记录 {index}:\n{content}")
#         remaining -= len(content)

#     if not fragments:
#         return ""

#     return (
#         "以下内容只来自当前 thread_id，可用于补充当前会话。"
#         "如果它与当前用户输入、TripState 或已审核证据冲突，必须忽略它。\n\n"
#         + "\n\n".join(fragments)
#     )


def _reciprocal_rank_fusion(
    rankings: list[list[str]],
    *,
    rank_constant: int = 60,
) -> tuple[dict[str, float], dict[str, int]]:
    """Fuse ranked ID lists without comparing their incomparable raw scores."""
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    position = 0

    for ranking in rankings:
        for rank, document_id in enumerate(ranking, start=1):
            scores[document_id] = scores.get(document_id, 0.0) + (
                1.0 / (rank_constant + rank)
            )
            if document_id not in first_seen:
                first_seen[document_id] = position
                position += 1

    return scores, first_seen


def retrieve_hybrid_memories(
    user_id: str,
    thread_id: str,
    query: str,
    limit: int = 2,
    char_budget: int = 1200,
    candidate_limit: int = 10,
) -> str:
    """Retrieve current-thread history with vector search and BM25 RRF fusion."""
    if not query.strip() or limit <= 0 or char_budget <= 0:
        return ""

    thread_filter = {
        "$and": [
            {"user_id": user_id},
            {"thread_id": thread_id},
        ]
    }
    records = memory_store.get(
        where=thread_filter,
        include=["documents", "metadatas"],
    )

    documents_by_id: dict[str, str] = {}
    for document_id, document, metadata in zip(
        records.get("ids", []),
        records.get("documents", []),
        records.get("metadatas", []),
        strict=False,
    ):
        if (
            document_id
            and document
            and metadata
            and metadata.get("user_id") == user_id
            and metadata.get("thread_id") == thread_id
        ):
            documents_by_id[str(document_id)] = document

    thread_documents = list(documents_by_id.values())
    bm25_indexes = _rank_bm25_documents(
        thread_documents,
        query,
        candidate_limit,
    )
    thread_ids = list(documents_by_id)
    bm25_ids = [thread_ids[index] for index in bm25_indexes]

    vector_ids: list[str] = []
    try:
        vector_results = memory_store.similarity_search_with_score(
            query,
            k=candidate_limit,
            filter=thread_filter,
        )
    except Exception as error:
        logger.warning("Vector memory retrieval failed; using BM25 only: %s", error)
        vector_results = []

    for document, _distance in vector_results:
        metadata = document.metadata or {}
        document_id = document.id
        if (
            not document_id
            or metadata.get("user_id") != user_id
            or metadata.get("thread_id") != thread_id
        ):
            continue
        document_id = str(document_id)
        documents_by_id.setdefault(document_id, document.page_content)
        if document_id not in vector_ids:
            vector_ids.append(document_id)

    scores, first_seen = _reciprocal_rank_fusion([vector_ids, bm25_ids])
    if not scores:
        return ""

    ordered_ids = sorted(
        scores,
        key=lambda document_id: (-scores[document_id], first_seen[document_id]),
    )

    remaining = char_budget
    fragments: list[str] = []
    for index, document_id in enumerate(ordered_ids[:limit], start=1):
        content = documents_by_id[document_id][:remaining]
        if not content:
            break

        fragments.append(f"当前会话历史记录 {index}:\n{content}")
        remaining -= len(content)

    if not fragments:
        return ""

    return (
        "以下内容只来自当前 user_id 和 thread_id，"
        "由向量检索与 BM25 关键词检索融合得到。"
        "如果它与当前用户输入、TripState 或已审核证据冲突，必须忽略它。\n\n"
        + "\n\n".join(fragments)
    )
