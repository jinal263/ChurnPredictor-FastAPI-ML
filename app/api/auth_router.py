"""
app/api/auth_router.py
POST /api/auth/signup  — create a new viewer account
GET  /api/auth/me      — return current logged-in user info

Wire into main.py with these two lines (already in your main.py):
    from app.api import auth_router
    app.include_router(auth_router.router, prefix="/api", tags=["auth"])
"""
from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.auth import require_auth, require_admin, get_user_by_username, create_user

router = APIRouter()


# ── Request / Response schemas ────────────────────────────────────

class SignUpRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=30)
    password: str = Field(..., min_length=6)


class SignUpResponse(BaseModel):
    message:  str
    username: str
    role:     str


class MeResponse(BaseModel):
    username: str
    role:     str


# ── Endpoints ─────────────────────────────────────────────────────

@router.post(
    "/auth/signup",
    response_model=SignUpResponse,
    status_code=201,
    summary="Create a new viewer account (no login required)",
)
async def signup(
    body: SignUpRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Anyone can call this to register a viewer account.
    Viewers can run predictions and see history, but cannot train models.

    To create an admin account, an existing admin must call
    POST /api/auth/signup/admin while logged in.
    """
    existing = await get_user_by_username(body.username, db)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Username '{body.username}' is already taken.",
        )

    user = await create_user(
        username=body.username,
        plain_password=body.password,
        role="viewer",
        db=db,
    )

    return SignUpResponse(
        message=f"Account created. You can now sign in as '{user.username}'.",
        username=user.username,
        role=user.role,
    )


@router.post(
    "/auth/signup/admin",
    response_model=SignUpResponse,
    status_code=201,
    summary="Create an admin account — existing admin only",
)
async def signup_admin(
    body: SignUpRequest,
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(require_admin),
):
    """
    Only an existing admin can create another admin account.
    Prevents anyone from self-registering as admin.
    """
    existing = await get_user_by_username(body.username, db)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Username '{body.username}' is already taken.",
        )

    user = await create_user(
        username=body.username,
        plain_password=body.password,
        role="admin",
        db=db,
    )

    return SignUpResponse(
        message=f"Admin account '{user.username}' created by {current_user}.",
        username=user.username,
        role=user.role,
    )


@router.get(
    "/auth/me",
    response_model=MeResponse,
    summary="Return currently logged-in user info",
)
async def me(
    db: AsyncSession = Depends(get_db),
    current_user: str = Depends(require_auth),
):
    """Returns the username and role of whoever is currently logged in."""
    user = await get_user_by_username(current_user, db)
    return MeResponse(username=user.username, role=user.role)