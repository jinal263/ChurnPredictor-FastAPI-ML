"""
app/models.py
SQLAlchemy ORM table definitions.

IMPORTANT — Base is imported from app.database, NOT defined here.
Your database.py owns Base. All models must inherit from that same
Base so SQLAlchemy knows they belong to the same engine and can
create their tables with Base.metadata.create_all().

If each file defined its own Base, the tables would be invisible
to each other and init_db() would create nothing.

CHANGES FROM YOUR ORIGINAL:
  - Added: from app.database import Base  (instead of defining Base here)
  - Added: the User class at the bottom
  - Application and TrainingRun are completely unchanged
"""
from __future__ import annotations
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean,
    DateTime, Text, func
)
from app.database import Base                                   # ← import, not define


# ── Your existing tables — UNCHANGED ─────────────────────────────

class Application(Base):
    __tablename__ = "applications"

    id                   = Column(Integer, primary_key=True, index=True)
    customer_id          = Column(String,  index=True)
    age                  = Column(Integer)
    income               = Column(Float)
    credit_score         = Column(Integer)
    tenure_months        = Column(Integer)
    monthly_charges      = Column(Float)
    num_products         = Column(Integer)
    support_calls        = Column(Integer)
    complaints_last_6m   = Column(Integer)
    avg_monthly_usage_gb = Column(Float)
    payment_delay_days   = Column(Integer)
    gender               = Column(String)
    education            = Column(String)
    marital_status       = Column(String)
    contract             = Column(String)
    signup_month         = Column(Integer)
    signup_year          = Column(Integer)
    churn_prediction     = Column(Boolean)
    churn_probability    = Column(Float)
    risk_label           = Column(String)
    model_version        = Column(String)
    created_at           = Column(DateTime, default=datetime.utcnow)


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id            = Column(Integer, primary_key=True, index=True)
    model_version = Column(String,  unique=True, index=True)
    status        = Column(String,  default="running")
    n_samples     = Column(Integer, nullable=True)
    n_features    = Column(Integer, nullable=True)
    accuracy      = Column(Float,   nullable=True)
    roc_auc       = Column(Float,   nullable=True)
    f1_score      = Column(Float,   nullable=True)
    oob_score     = Column(Float,   nullable=True)
    error_message = Column(Text,    nullable=True)
    is_active     = Column(Boolean, default=False)
    created_at    = Column(DateTime, default=datetime.utcnow)
    finished_at   = Column(DateTime, nullable=True)


# ── NEW: User table ───────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id              = Column(Integer, primary_key=True, index=True)
    username        = Column(String,  unique=True, index=True, nullable=False)
    hashed_password = Column(String,  nullable=False)
    role            = Column(String,  nullable=False, default="viewer")
    created_at      = Column(DateTime, server_default=func.now())