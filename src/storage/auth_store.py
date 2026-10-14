"""SQLite persistence for application users and password hashes."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from src.auth.models import AuthUser
from src.paths import DATA_DIR


class EmailAlreadyExistsError(ValueError):
    """该邮箱已被注册时raise"""

class DisplayNameAlreadyExistsError(ValueError):
    """该用户名已被注册时raise"""

class SQLiteAuthRepository:
    def __init__(self, path: str | Path = DATA_DIR / "auth.db") -> None:
        self.path = Path(path)
        self._initialize()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS auth_users (
                    user_id TEXT PRIMARY KEY,
                    email TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    display_name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    disabled INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS ux_auth_users_email
                ON auth_users(email COLLATE NOCASE)
                """
            )
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS ux_users_display_name
                ON auth_users(display_name COLLATE NOCASE);
                """
            )

    def create_user(
        self,
        *,
        email: str,
        display_name: str,
        password_hash: str,
    ) -> AuthUser:
        user = AuthUser(
            user_id=uuid4(),
            email=email,
            display_name=display_name,
        )
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO auth_users (
                        user_id, email, display_name, password_hash,
                        disabled, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 0, ?, ?)
                    """,
                    (
                        str(user.user_id),
                        user.email,
                        user.display_name,
                        password_hash,
                        user.created_at.isoformat(),
                        now,
                    ),
                )
        except sqlite3.IntegrityError as error:
            msg = str(error).lower()
            if "display_name" in msg:
                raise DisplayNameAlreadyExistsError(display_name) from error
            if "email" in msg:
                raise EmailAlreadyExistsError(email) from error
            raise
        return user

    def get_by_email(self, email: str) -> tuple[AuthUser, str] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM auth_users
                WHERE email = ? COLLATE NOCASE
                """,
                (email.strip().casefold(),),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_user(row), str(row["password_hash"])

    def get_by_id(self, user_id: UUID | str) -> AuthUser | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM auth_users WHERE user_id = ?",
                (str(user_id),),
            ).fetchone()
        return self._row_to_user(row) if row else None

    def update_password_hash(
        self,
        user_id: UUID | str,
        password_hash: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE auth_users
                SET password_hash = ?, updated_at = ?
                WHERE user_id = ?
                """,
                (
                    password_hash,
                    datetime.now(timezone.utc).isoformat(),
                    str(user_id),
                ),
            )

    @staticmethod
    def _row_to_user(row: sqlite3.Row) -> AuthUser:
        return AuthUser(
            user_id=row["user_id"],
            email=row["email"],
            display_name=row["display_name"],
            disabled=bool(row["disabled"]),
            created_at=row["created_at"],
        )
