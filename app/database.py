"""
database.py — SQLAlchemy 2.x async engine.
Swap DATABASE_URL env var to use Postgres, MySQL, etc.
SQLite is the default for zero-config local development.
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


async def init_db() -> None:
    """Create all tables on startup."""
    async with engine.begin() as conn:
        from app.models import Application, TrainingRun   # noqa: F401 — registers tables
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncSession:
    """FastAPI dependency — yields an async DB session."""
    async with AsyncSessionLocal() as session:
        yield session
