"""
app/ml/feature_engineering.py
Exact feature engineering pipeline from the notebook.
Two modes:
  fit_transform(df)  — used during training (learns medians for imputation)
  transform(df)      — used during inference (applies saved imputer stats)
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd

EDU_ORDER = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}

# Feature columns in the exact order the model was trained on
FEATURE_COLUMNS = [
    # Raw numeric
    "age", "income", "credit_score", "tenure_months", "monthly_charges",
    "num_products", "support_calls", "complaints_last_6m",
    "avg_monthly_usage_gb", "payment_delay_days",
    # Engineered
    "charges_per_tenure", "usage_per_dollar", "complaint_rate",
    "support_per_product", "financial_stress", "engagement_score",
    "education_encoded", "credit_score_missing_flag",
    "is_q4_signup", "signup_month_sin", "signup_month_cos",
    # One-hot
    "gender_Male", "gender_Other",
    "contract_One year", "contract_Two year",
    "marital_status_Married", "marital_status_Single",
]


def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Create all engineered columns from raw inputs."""
    df = df.copy()
    df["charges_per_tenure"]    = df["monthly_charges"] / (df["tenure_months"] + 1)
    df["usage_per_dollar"]      = df["avg_monthly_usage_gb"] / (df["monthly_charges"] + 1)
    df["complaint_rate"]        = df["complaints_last_6m"] / (df["tenure_months"] + 1)
    df["support_per_product"]   = df["support_calls"] / (df["num_products"] + 1)
    df["financial_stress"]      = (df["payment_delay_days"] * df["monthly_charges"]) / (df["income"] + 1)
    df["engagement_score"]      = (df["num_products"] * df["avg_monthly_usage_gb"]) / (df["monthly_charges"] + 1)
    df["education_encoded"]         = df["education"].map(EDU_ORDER).fillna(0).astype(int)
    df["credit_score_missing_flag"] = df["credit_score"].isna().astype(int)
    df["is_q4_signup"]      = df["signup_month"].isin([10, 11, 12]).astype(int)
    df["signup_month_sin"]  = np.sin(2 * math.pi * df["signup_month"] / 12)
    df["signup_month_cos"]  = np.cos(2 * math.pi * df["signup_month"] / 12)
    df["gender_Male"]            = (df["gender"] == "Male").astype(int)
    df["gender_Other"]           = (df["gender"] == "Other").astype(int)
    df["contract_One year"]      = (df["contract"] == "One year").astype(int)
    df["contract_Two year"]      = (df["contract"] == "Two year").astype(int)
    df["marital_status_Married"] = (df["marital_status"] == "Married").astype(int)
    df["marital_status_Single"]  = (df["marital_status"] == "Single").astype(int)
    return df


def fit_transform(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Used during training.
    Returns:
      X           — feature DataFrame ready for the model
      imputer_stats — dict of median values to save as imputer_stats.pkl
    """
    df = _add_derived(df)
    X  = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    X  = X.replace([np.inf, -np.inf], np.nan)

    # Learn medians from training data
    imputer_stats: dict = X.median(numeric_only=True).to_dict()
    X = X.fillna(imputer_stats)

    return X, imputer_stats


def transform(df: pd.DataFrame, imputer_stats: dict) -> pd.DataFrame:
    """
    Used during inference.
    Applies the saved imputer_stats instead of recalculating medians.
    """
    df = _add_derived(df)
    X  = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    X  = X.replace([np.inf, -np.inf], np.nan)
    X  = X.fillna(imputer_stats)
    return X
