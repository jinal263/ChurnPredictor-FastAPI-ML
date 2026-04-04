"""
app/auth.py — Authentication and authorisation for CC Underwriting.

CHANGE FROM PREVIOUS VERSION:
  Removed passlib entirely — it is incompatible with bcrypt 4.x and Python 3.14.
  Now uses bcrypt directly instead. Everything else is identical.

  pip install bcrypt   (you already have it — passlib installed it as a dependency)
"""
from __future__ import annotations
import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models import User

security = HTTPBasic()


# ── Password helpers ──────────────────────────────────────────────

def hash_password(plain: str) -> str:
    """Hash a plain-text password with bcrypt. Returns a string for DB storage."""
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Check a plain password against a stored bcrypt hash."""
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


# ── Database helpers (used by auth_router and database.py) ────────

async def get_user_by_username(username: str, db: AsyncSession) -> User | None:
    result = await db.execute(select(User).where(User.username == username))
    return result.scalar_one_or_none()


async def create_user(
    username: str,
    plain_password: str,
    role: str,
    db: AsyncSession,
) -> User:
    user = User(
        username=username,
        hashed_password=hash_password(plain_password),
        role=role,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


# ── Internal credential checker ───────────────────────────────────

async def _get_verified_user(
    credentials: HTTPBasicCredentials,
    db: AsyncSession,
) -> User:
    """
    Look up user and verify password.
    Always runs verify_password even for unknown users to prevent
    timing attacks that could reveal valid usernames.
    """
    DUMMY_HASH = "$2b$12$KIXm/SNR/TDvSlKTIOAMuOtHKREkpd5E3HGuXrAbXh8IwF9/l9bXq"

    user = await get_user_by_username(credentials.username, db)

    password_ok = verify_password(
        credentials.password,
        user.hashed_password if user else DUMMY_HASH,
    )

    if not user or not password_ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
            headers={"WWW-Authenticate": "Basic"},
        )

    return user


# ── Public dependencies ───────────────────────────────────────────

async def require_auth(
    credentials: HTTPBasicCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> str:
    """Any logged-in user (admin OR viewer) can pass. Returns username."""
    user = await _get_verified_user(credentials, db)
    return user.username


async def require_admin(
    credentials: HTTPBasicCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> str:
    """Only admin users can pass. Viewers get 403 Forbidden."""
    user = await _get_verified_user(credentials, db)

    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You need admin privileges to access this resource.",
        )

    return user.username