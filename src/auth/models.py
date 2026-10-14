"""认证领域的公共契约层"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from uuid import UUID, uuid4

from pydantic import Field, field_validator

from src.domain.base import DomainModel


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# 持久化用户实体
class AuthUser(DomainModel):
    user_id: UUID = Field(default_factory=uuid4)
    email: str = Field(min_length=3, max_length=254)
    display_name: str = Field(min_length=1, max_length=80)
    disabled: bool = False
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        normalized = value.strip().casefold()
        if not _EMAIL_RE.fullmatch(normalized):
            raise ValueError("invalid email address")
        return normalized

# 客户端身份
class Principal(DomainModel):
    """Authenticated identity derived by the server, never from request data."""

    user_id: str = Field(default="", max_length=128)
    email: str = Field(default="", max_length=254)
    display_name: str = Field(default="", max_length=80)

    @property
    def authenticated(self) -> bool:
        return bool(self.user_id)

# 注册DTO
class RegisterRequest(DomainModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=3, max_length=128)
    display_name: str = Field(min_length=1, max_length=80)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        normalized = value.strip().casefold()
        if not _EMAIL_RE.fullmatch(normalized):
            raise ValueError("invalid email address")
        return normalized

# 登录DTO
class LoginRequest(DomainModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return value.strip().casefold()

# TOKEN响应DTO
class TokenResponse(DomainModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(ge=1)
    user: AuthUser
