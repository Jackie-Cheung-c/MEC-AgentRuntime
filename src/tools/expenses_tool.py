"""Persistent personal expense recording and summary utilities."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date

from langchain.tools import ToolRuntime, tool
from src.paths import DATA_DIR
from src.tools.idempotency import build_idempotency_key


DATABASE_PATH = DATA_DIR / "expenses.db"
DEFAULT_USER_ID = "local_user"
EXPENSE_PATTERN = re.compile(
    r"(?P<category>[\u4e00-\u9fffA-Za-z0-9]{1,20}?)"
    r"(?:花了|花费了|花费|消费了|消费|支出|用了|支付了|付了)\s*"
    r"(?:人民币|RMB|CNY|¥)?\s*(?P<amount>\d+(?:\.\d{1,2})?)\s*(?:元|块)?",
    re.IGNORECASE,
)


@contextmanager
def _connection() -> Iterator[sqlite3.Connection]:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH, timeout=30)
    try:
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS expenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                category TEXT NOT NULL,
                amount REAL NOT NULL CHECK(amount > 0),
                expense_date TEXT NOT NULL,
                source_text TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_expenses_user_date "
            "ON expenses(user_id, expense_date)"
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS expense_operations (
                idempotency_key TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                result_text TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _clean_category(value: str) -> str:
    category = value.strip(" ，,。；;：:\n\t")
    for prefix in ("今天", "昨日", "昨天", "前天", "我", "的"):
        if category.startswith(prefix) and len(category) > len(prefix):
            category = category[len(prefix) :]
    return category or "未分类"


def _parse_expenses(expense_text: str) -> list[tuple[str, float]]:
    entries: list[tuple[str, float]] = []
    for match in EXPENSE_PATTERN.finditer(expense_text):
        amount = float(match.group("amount"))
        if amount > 0:
            entries.append((_clean_category(match.group("category")), amount))
    return entries


def _runtime_user_id(runtime: ToolRuntime) -> str:
    user_id = str(
        runtime.config.get("configurable", {}).get("user_id", "")
    ).strip()
    if not user_id:
        raise ValueError("Authenticated user_id is missing from the tool runtime.")
    return user_id


def record_expenses_for_user(
    expense_text: str,
    user_id: str,
    *,
    idempotency_key: str | None = None,
) -> str:
    expense_text = expense_text.strip()
    user_id = user_id.strip() or DEFAULT_USER_ID
    if not expense_text:
        return "请提供花费信息，例如“今天吃饭花了100，交通费花了50”。"

    entries = _parse_expenses(expense_text)
    if not entries:
        return "未识别到有效花费。请使用“类别花了金额”的格式，例如“餐饮花了100元”。"

    today = date.today().isoformat()
    with _connection() as connection:
        if idempotency_key:
            if not connection.in_transaction:
                connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT result_text FROM expense_operations
                WHERE idempotency_key = ?
                """,
                (idempotency_key,),
            ).fetchone()
            if existing is not None:
                return str(existing[0])
        connection.executemany(
            """
            INSERT INTO expenses (user_id, category, amount, expense_date, source_text)
            VALUES (?, ?, ?, ?, ?)
            """,
            [(user_id, category, amount, today, expense_text) for category, amount in entries],
        )
        total = connection.execute(
            "SELECT COALESCE(SUM(amount), 0) FROM expenses WHERE user_id = ?",
            (user_id,),
        ).fetchone()[0]

        current_total = sum(amount for _, amount in entries)
        lines = [f"已记录 {len(entries)} 项花费（{today}）："]
        lines.extend(f"- {category}: ¥{amount:.2f}" for category, amount in entries)
        lines.append(f"本次合计: ¥{current_total:.2f}")
        lines.append(f"累计总花费: ¥{float(total):.2f}")
        result_text = "\n".join(lines)
        if idempotency_key:
            connection.execute(
                """
                INSERT INTO expense_operations (
                    idempotency_key, user_id, result_text
                ) VALUES (?, ?, ?)
                """,
                (idempotency_key, user_id, result_text),
            )
    return result_text


@tool
def record_expenses(expense_text: str, runtime: ToolRuntime) -> str:
    """记录一句话中的一项或多项花费，并计算该用户的累计花费。

    示例：今天吃饭花了100，交通费花了50，买零食花了200。
    """
    user_id = _runtime_user_id(runtime)
    normalized_text = expense_text.strip()
    idempotency_key = build_idempotency_key(
        runtime=runtime,
        operation="record_expenses",
        user_id=user_id,
        payload={"expense_text": normalized_text},
    )
    return record_expenses_for_user(
        normalized_text,
        user_id,
        idempotency_key=idempotency_key,
    )


def get_expense_summary_for_user(user_id: str) -> str:
    user_id = user_id.strip() or DEFAULT_USER_ID
    with _connection() as connection:
        rows = connection.execute(
            """
            SELECT category, SUM(amount) AS total
            FROM expenses
            WHERE user_id = ?
            GROUP BY category
            ORDER BY total DESC, category ASC
            """,
            (user_id,),
        ).fetchall()

    if not rows:
        return "暂无费用记录。"
    total = sum(float(row[1]) for row in rows)
    lines = ["费用分类汇总："]
    lines.extend(f"- {category}: ¥{float(amount):.2f}" for category, amount in rows)
    lines.append(f"累计总花费: ¥{total:.2f}")
    return "\n".join(lines)


@tool
def get_expense_summary(runtime: ToolRuntime) -> str:
    """查看用户已记录花费的分类汇总和累计总额。"""
    return get_expense_summary_for_user(_runtime_user_id(runtime))
