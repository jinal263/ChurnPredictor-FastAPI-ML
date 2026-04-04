"""
database.py — SQLAlchemy 2.x async engine.
Swap DATABASE_URL env var to use Postgres, MySQL, etc.
SQLite is the default for zero-config local development.

CHANGES FROM YOUR ORIGINAL — only 2 things added:
  1. seed_default_admin() — new function (see below)
  2. init_db() — one extra line at the end calling seed_default_admin()
  3. User added to the import inside init_db() so its table gets created

Everything else is exactly your original code, word for word.
"""
import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase

# SQLite async needs aiosqlite driver.
# Postgres: postgresql+asyncpg://user:pass@host/db
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite+aiosqlite:///./cc_underwriting.db"
)

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    # SQLite-only: allow multiple threads
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


class Base(DeclarativeBase):
    pass


# ── NEW: seed function ────────────────────────────────────────────

async def seed_default_admin() -> None:
    """
    Create the default admin account if it does not already exist.

    Runs on every startup — safe to call repeatedly because it checks
    first whether the admin row already exists before inserting anything.

    WHY this is needed:
      The users table starts empty on a fresh database. Without this
      there would be no way to log in at all on first run.
    """
    import logging
    from sqlalchemy import select
    from app.models import User
    from app.auth import hash_password

    DEFAULT_USERNAME = "admin"
    DEFAULT_PASSWORD = "churn2024"

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(User).where(User.username == DEFAULT_USERNAME)
        )
        if result.scalar_one_or_none() is None:
            db.add(User(
                username=DEFAULT_USERNAME,
                hashed_password=hash_password(DEFAULT_PASSWORD),
                role="admin",
            ))
            await db.commit()
            logging.getLogger(__name__).info(
                f"Default admin account '{DEFAULT_USERNAME}' created."
            )


# ── YOUR ORIGINAL — init_db with 2 small additions ───────────────

async def init_db() -> None:
    """Create all tables on startup."""
    async with engine.begin() as conn:
        from app.models import Application, TrainingRun, User  # ← added User
        await conn.run_sync(Base.metadata.create_all)

    await seed_default_admin()                                  # ← NEW


async def get_db() -> AsyncSession:
    """FastAPI dependency — yields an async DB session."""
    async with AsyncSessionLocal() as session:
        yield session