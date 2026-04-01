"""
schemas.py — Pydantic v2 request / response contracts.
These are the shapes of every API request and response body.
"""
from __future__ import annotations
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


# ── Input contract ───────────────────────────────────────────────

class ApplicantIn(BaseModel):
    """Fields sent by the client when requesting a churn prediction."""
    customer_id:          Optional[str]  = Field(None,  examples=["CUST_00001"])
    age:                  int            = Field(..., ge=18, le=100)
    income:               float          = Field(..., ge=0)
    credit_score:         int            = Field(..., ge=300, le=850)
    tenure_months:        int            = Field(..., ge=1, le=240)
    monthly_charges:      float          = Field(..., ge=0, le=1000)
    num_products:         int            = Field(..., ge=1, le=20)
    support_calls:        int            = Field(..., ge=0, le=50)
    complaints_last_6m:   int            = Field(..., ge=0, le=30)
    avg_monthly_usage_gb: float          = Field(..., ge=0, le=500)
    payment_delay_days:   int            = Field(..., ge=0, le=365)
    gender:               str            = Field(..., examples=["Male"])
    education:            str            = Field(..., examples=["Bachelor"])
    marital_status:       str            = Field(..., examples=["Married"])
    contract:             str            = Field(..., examples=["Month-to-month"])
    signup_month:         int            = Field(..., ge=1, le=12)
    signup_year:          int            = Field(..., ge=2000, le=2030)

    @field_validator("gender")
    @classmethod
    def v_gender(cls, v):
        allowed = {"Male", "Female", "Other"}
        if v not in allowed:
            raise ValueError(f"gender must be one of {allowed}")
        return v

    @field_validator("education")
    @classmethod
    def v_education(cls, v):
        allowed = {"High School", "Bachelor", "Master", "PhD"}
        if v not in allowed:
            raise ValueError(f"education must be one of {allowed}")
        return v

    @field_validator("marital_status")
    @classmethod
    def v_marital(cls, v):
        allowed = {"Single", "Married", "Divorced"}
        if v not in allowed:
            raise ValueError(f"marital_status must be one of {allowed}")
        return v

    @field_validator("contract")
    @classmethod
    def v_contract(cls, v):
        allowed = {"Month-to-month", "One year", "Two year"}
        if v not in allowed:
            raise ValueError(f"contract must be one of {allowed}")
        return v


class BatchPredictIn(BaseModel):
    customers: List[ApplicantIn] = Field(..., min_length=1, max_length=500)


# ── Output contract ──────────────────────────────────────────────

class PredictionOut(BaseModel):
    record_id:         int
    customer_id:       Optional[str]
    churn_prediction:  bool
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    risk_label:        str   # Low | Medium | High
    model_version:     str


class BatchPredictOut(BaseModel):
    results:       List[PredictionOut]
    total:         int
    model_version: str


# ── Training ─────────────────────────────────────────────────────

class TrainIn(BaseModel):
    n_estimators:      int = Field(200, ge=10, le=2000)
    max_depth:         int = Field(12,  ge=1,  le=50)
    min_samples_split: int = Field(10,  ge=2,  le=100)
    min_samples_leaf:  int = Field(5,   ge=1,  le=50)
    n_samples:         int = Field(5000, ge=500, le=50_000,
                                   description="Synthetic rows to generate")


class TrainStartOut(BaseModel):
    model_version: str
    status:        str
    message:       str


class TrainStatusOut(BaseModel):
    id:            int
    model_version: str
    status:        str          # running | done | failed
    accuracy:      Optional[float]
    roc_auc:       Optional[float]
    f1_score:      Optional[float]
    oob_score:     Optional[float]
    error_message: Optional[str]
    is_active:     bool
    created_at:    datetime
    finished_at:   Optional[datetime]

    model_config = {"from_attributes": True}


# ── Application history ───────────────────────────────────────────

# ── Application history ───────────────────────────────────────────

class ApplicationOut(BaseModel):
    id:                int
    customer_id:       Optional[str]
    churn_prediction:  bool
    churn_probability: float
    risk_label:        str
    model_version:     str
    created_at:        datetime

    model_config = {"from_attributes": True}


class PaginatedApplications(BaseModel):
    items:       List[ApplicationOut]
    total:       int
    page:        int
    page_size:   int
    total_pages: int
    has_next:    bool
    has_prev:    bool


# ── Error ─────────────────────────────────────────────────────────

class ErrorOut(BaseModel):
    error_code: str
    message:    str
    detail:     str
