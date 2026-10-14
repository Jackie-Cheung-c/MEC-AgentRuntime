"""Account registration, login, and HMAC-signed bearer tokens."""

from __future__ import annotations

import os
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from config import settings
from src.auth.models import AuthUser, Principal, TokenResponse
from src.paths import DATA_DIR
from src.storage.auth_store import EmailAlreadyExistsError, SQLiteAuthRepository,DisplayNameAlreadyExistsError


class AuthenticationError(RuntimeError):
    pass

def _load_or_create_secret() -> str:
    configured = settings.AUTH_SECRET_KEY.strip()
    if configured:
        return configured
    path: Path = DATA_DIR / ".auth_secret"
    if path.exists():
        secret = path.read_text(encoding="utf-8").strip()
        if secret:
            return secret
    secret = secrets.token_urlsafe(48)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(secret, encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return secret


class AuthService:
    def __init__(
        self,
        repository: SQLiteAuthRepository,
        *,
        secret_key: str | None = None,
        token_ttl_seconds: int | None = None,
        issuer: str | None = None,
    ) -> None:
        self.repository = repository
        self.secret_key = secret_key or _load_or_create_secret()
        self.token_ttl_seconds = (
            token_ttl_seconds
            if token_ttl_seconds is not None
            else settings.AUTH_TOKEN_TTL_SECONDS
        )
        self.issuer = issuer or settings.AUTH_ISSUER
        self.passwords = PasswordHasher()

    @classmethod
    def with_defaults(cls) -> "AuthService":
        return cls(SQLiteAuthRepository())

    def register(
        self,
        *,
        email: str,
        password: str,
        display_name: str,
    ) -> TokenResponse:
        password_hash = self.passwords.hash(password)
        try:
            user = self.repository.create_user(
                email=email,
                display_name=display_name,
                password_hash=password_hash,
            )
        except EmailAlreadyExistsError as error:
            raise AuthenticationError("邮箱已注册") from error
        except DisplayNameAlreadyExistsError as error:
            raise AuthenticationError("用户名已注册") from error
        return self._token_response(user)

    def login(self, *, email: str, password: str) -> TokenResponse:
        record = self.repository.get_by_email(email)
        if record is None:
            raise AuthenticationError("Invalid email or password.")
        user, password_hash = record
        if user.disabled:
            raise AuthenticationError("This account is disabled.")
        try:
            self.passwords.verify(password_hash, password)
        except (VerifyMismatchError, VerificationError, InvalidHashError) as error:
            raise AuthenticationError("Invalid email or password.") from error
        if self.passwords.check_needs_rehash(password_hash):
            self.repository.update_password_hash(
                user.user_id,
                self.passwords.hash(password),
            )
        return self._token_response(user)

    def authenticate_token(self, token: str) -> Principal:
        try:
            payload: dict[str, Any] = jwt.decode(
                token,
                self.secret_key,
                algorithms=["HS256"],
                issuer=self.issuer,
                options={"require": ["exp", "iat", "sub"]},
            )
        except jwt.PyJWTError as error:
            raise AuthenticationError("Invalid or expired access token.") from error
        user = self.repository.get_by_id(str(payload["sub"]))
        if user is None or user.disabled:
            raise AuthenticationError("The account is unavailable.")
        return Principal(
            user_id=str(user.user_id),
            email=user.email,
            display_name=user.display_name,
        )

    def _token_response(self, user: AuthUser) -> TokenResponse:
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=self.token_ttl_seconds)
        token = jwt.encode(
            {
                "sub": str(user.user_id),
                "email": user.email,
                "iat": now,
                "exp": expires_at,
                "iss": self.issuer,
            },
            self.secret_key,
            algorithm="HS256",
        )
        return TokenResponse(
            access_token=token,
            expires_in=self.token_ttl_seconds,
            user=user,
        )
