"""
models.py — SQLAlchemy ORM table definitions.
Kept separate from database.py so imports stay clean.
"""
from datetime import datetime
from sqlalchemy import (
    Integer, Float, String, Boolean, DateTime, Text, func
)
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base


class Application(Base):
    """One row per prediction request."""
    __tablename__ = "applications"

    id: Mapped[int]          = mapped_column(Integer, primary_key=True, index=True)
    customer_id: Mapped[str | None] = mapped_column(String(50), index=True, nullable=True)

    # Raw inputs
    age: Mapped[int]                    = mapped_column(Integer)
    income: Mapped[float]               = mapped_column(Float)
    credit_score: Mapped[int]           = mapped_column(Integer)
    tenure_months: Mapped[int]          = mapped_column(Integer)
    monthly_charges: Mapped[float]      = mapped_column(Float)
    num_products: Mapped[int]           = mapped_column(Integer)
    support_calls: Mapped[int]          = mapped_column(Integer)
    complaints_last_6m: Mapped[int]     = mapped_column(Integer)
    avg_monthly_usage_gb: Mapped[float] = mapped_column(Float)
    payment_delay_days: Mapped[int]     = mapped_column(Integer)
    gender: Mapped[str]                 = mapped_column(String(20))
    education: Mapped[str]              = mapped_column(String(30))
    marital_status: Mapped[str]         = mapped_column(String(20))
    contract: Mapped[str]               = mapped_column(String(30))
    signup_month: Mapped[int]           = mapped_column(Integer)
    signup_year: Mapped[int]            = mapped_column(Integer)

    # Outputs
    churn_prediction: Mapped[bool]      = mapped_column(Boolean)
    churn_probability: Mapped[float]    = mapped_column(Float)
    risk_label: Mapped[str]             = mapped_column(String(10))
    model_version: Mapped[str]          = mapped_column(String(60))
    created_at: Mapped[datetime]        = mapped_column(DateTime, default=datetime.utcnow)


class TrainingRun(Base):
    """One row per completed training run."""
    __tablename__ = "training_runs"

    id: Mapped[int]             = mapped_column(Integer, primary_key=True, index=True)
    model_version: Mapped[str]  = mapped_column(String(60), unique=True, index=True)
    status: Mapped[str]         = mapped_column(String(20), default="running")  # running | done | failed
    n_samples: Mapped[int]      = mapped_column(Integer, nullable=True)
    n_features: Mapped[int]     = mapped_column(Integer, nullable=True)
    accuracy: Mapped[float | None]   = mapped_column(Float, nullable=True)
    roc_auc: Mapped[float | None]    = mapped_column(Float, nullable=True)
    f1_score: Mapped[float | None]   = mapped_column(Float, nullable=True)
    oob_score: Mapped[float | None]  = mapped_column(Float, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool]     = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
