"""Authentication routes and bearer-token dependencies."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import settings
from src.auth import (
    AuthService,
    AuthenticationError,
    LoginRequest,
    Principal,
    RegisterRequest,
    TokenResponse,
)


router = APIRouter(prefix="/api/auth", tags=["auth"])
_bearer = HTTPBearer(auto_error=False)


def get_auth_service(request: Request) -> AuthService:
    return request.app.state.auth_service


def get_current_user(
    request: Request,
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(_bearer),
    ],
    service: Annotated[AuthService, Depends(get_auth_service)],
) -> Principal:
    """Resolve identity from a bearer token, never from request user_id fields."""
    if credentials is None or credentials.scheme.casefold() != "bearer":
        if settings.AUTH_REQUIRED:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return Principal()
    try:
        return service.authenticate_token(credentials.credentials)
    except AuthenticationError as error:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(error),
            headers={"WWW-Authenticate": "Bearer"},
        ) from error


def resolve_user_id(
    principal: Principal,
    claimed_user_id: str | None = None,
) -> str:
    """Return the authenticated user ID.

    In explicit development mode, callers may still supply the legacy
    user_id field. Production must keep AUTH_REQUIRED=true.
    """
    if principal.authenticated:
        return principal.user_id
    if settings.AUTH_REQUIRED:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
        )
    return (claimed_user_id or "local_user").strip() or "local_user"


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register(
    payload: RegisterRequest,
    service: Annotated[AuthService, Depends(get_auth_service)],
) -> TokenResponse:
    try:
        return service.register(
            email=payload.email,
            password=payload.password,
            display_name=payload.display_name,
        )
    except RuntimeError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error


@router.post("/login", response_model=TokenResponse)
async def login(
    payload: LoginRequest,
    service: Annotated[AuthService, Depends(get_auth_service)],
) -> TokenResponse:
    try:
        return service.login(email=payload.email, password=payload.password)
    except AuthenticationError as error:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(error),
        ) from error


@router.get("/me", response_model=Principal)
async def me(
    principal: Annotated[Principal, Depends(get_current_user)],
) -> Principal:
    if not principal.authenticated:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
        )
    return principal
