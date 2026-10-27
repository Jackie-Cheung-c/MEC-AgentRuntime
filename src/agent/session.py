"""Stable checkpoint identities for isolated Agent conversations."""
# ----------------把“用户ID”和“会话ID”(可能有冲突)，压缩成一个固定长度、绝对安全、绝不重复的“缓存key”-------
from __future__ import annotations

import hashlib

LEGACY_CHECKPOINT_NAMESPACE = "travelops"


# 创建session key，基于用户ID和线程ID生成唯一的哈希值
def build_session_key(user_id: str, thread_id: str) -> str:
    """Return an opaque checkpoint key scoped by both user and public thread ID."""
    identity = f"{len(user_id)}:{user_id}{len(thread_id)}:{thread_id}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()  # 压缩成 64 位固定的十六进制串
    # Persisted checkpoints depend on this stable legacy namespace.
    return f"{LEGACY_CHECKPOINT_NAMESPACE}:{digest}"
