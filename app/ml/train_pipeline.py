"""
app/ml/train_pipeline.py
Full training pipeline:
  1. Generate synthetic data (mirrors the notebook)
  2. Feature engineering (fit_transform)
  3. Scale features (StandardScaler)
  4. RandomizedSearchCV to tune Random Forest
  5. Save all artefacts into a versioned models_store folder
"""
from __future__ import annotations
import json
import logging
import pickle
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from app.ml.feature_engineering import fit_transform, FEATURE_COLUMNS

logger = logging.getLogger(__name__)

MODELS_STORE = Path(os.getenv("MODELS_STORE", "./models_store"))


# ── Synthetic data (mirrors notebook exactly) ────────────────────

def _generate_data(n: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    age            = rng.normal(42, 14, n).clip(18, 85).astype(int)
    gender         = rng.choice(["Male", "Female", "Other"], n, p=[0.48, 0.48, 0.04])
    education      = rng.choice(["High School", "Bachelor", "Master", "PhD"], n, p=[0.35, 0.35, 0.20, 0.10])
    marital_status = rng.choice(["Single", "Married", "Divorced"], n, p=[0.35, 0.50, 0.15])
    edu_bonus      = np.where(education == "High School", 0,
                     np.where(education == "Bachelor", 15000,
                     np.where(education == "Master", 30000, 50000)))
    income              = (30000 + edu_bonus + age * 500 + rng.normal(0, 12000, n)).clip(15000, 250000).round(2)
    tenure_months       = rng.exponential(24, n).clip(1, 120).astype(int)
    monthly_charges     = rng.uniform(20, 150, n).round(2)
    num_products        = rng.poisson(2, n).clip(1, 8)
    support_calls       = rng.poisson(1.5, n).clip(0, 15)
    complaints_last_6m  = rng.poisson(0.8, n).clip(0, 10)
    avg_monthly_usage_gb = (num_products * 5 + rng.normal(0, 10, n)).clip(0, 200).round(1)
    payment_delay_days  = rng.exponential(5, n).clip(0, 90).round(0).astype(int)
    credit_score        = (300 + (income - 15000) / 235000 * 550 + rng.normal(0, 40, n)).clip(300, 850).astype(int)
    contract            = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.50, 0.30, 0.20])
    signup_month        = rng.choice(range(1, 13), n,
                              p=[0.12, 0.10, 0.09, 0.07, 0.06, 0.06, 0.06, 0.07, 0.08, 0.09, 0.10, 0.10])
    signup_year         = rng.choice([2020, 2021, 2022, 2023, 2024, 2025], n,
                              p=[0.05, 0.10, 0.15, 0.25, 0.30, 0.15])

    churn_logit = (
        -2.0 + 0.03 * support_calls + 0.15 * complaints_last_6m
        + 0.01 * monthly_charges - 0.02 * tenure_months
        + 0.5 * (contract == "Month-to-month") - 0.3 * (contract == "Two year")
        + 0.005 * payment_delay_days - 0.005 * age
        - 0.00001 * income + 0.3 * (num_products == 1)
        + rng.normal(0, 0.3, n)
    )
    churn = (rng.random(n) < 1 / (1 + np.exp(-churn_logit))).astype(int)

    return pd.DataFrame({
        "age": age, "gender": gender, "education": education,
        "marital_status": marital_status, "income": income,
        "credit_score": credit_score, "tenure_months": tenure_months,
        "contract": contract, "monthly_charges": monthly_charges,
        "num_products": num_products, "support_calls": support_calls,
        "complaints_last_6m": complaints_last_6m,
        "avg_monthly_usage_gb": avg_monthly_usage_gb,
        "payment_delay_days": payment_delay_days,
        "signup_month": signup_month, "signup_year": signup_year,
        "churn": churn,
    })


# ── Label encoders (saved separately per tutor spec) ─────────────

def _fit_label_encoders(df: pd.DataFrame) -> dict[str, LabelEncoder]:
    cat_cols = ["gender", "education", "marital_status", "contract"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        encoders[col] = le
    return encoders


# ── Main training entry point ────────────────────────────────────

def run_training(
    n_samples: int = 5000,
    n_estimators: int = 200,
    max_depth: int = 12,
    min_samples_split: int = 10,
    min_samples_leaf: int = 5,
) -> dict:
    """
    Full pipeline. Returns a metadata dict with version + metrics.
    All artefacts saved to models_store/<version>/
    """
    version = datetime.utcnow().strftime("v%Y%m%d_%H%M%S")
    store   = MODELS_STORE / version
    store.mkdir(parents=True, exist_ok=True)
    logger.info(f"Training {version} | n={n_samples}")

    # 1. Data
    df = _generate_data(n_samples)

    # 2. Label encoders (saved but not used in feature pipeline — kept for reference)
    label_encoders = _fit_label_encoders(df)

    # 3. Feature engineering
    X, imputer_stats = fit_transform(df)
    y = df["churn"]

    # 4. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 5. Scale
    scaler   = StandardScaler()
    Xtr_s    = scaler.fit_transform(X_train)
    Xte_s    = scaler.transform(X_test)

    # 6. RandomizedSearchCV over RF hyperparameters
    param_dist = {
        "n_estimators":     [100, 200, n_estimators],
        "max_depth":        [8, max_depth, 16],
        "min_samples_split":[5, min_samples_split, 20],
        "min_samples_leaf": [3, min_samples_leaf, 10],
        "max_features":     ["sqrt", "log2"],
        "class_weight":     ["balanced"],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        RandomForestClassifier(oob_score=True, bootstrap=True, random_state=42, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=10,
        scoring="roc_auc",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        refit=True,
    )
    search.fit(Xtr_s, y_train)
    model = search.best_estimator_

    # 7. Evaluate
    y_pred  = model.predict(Xte_s)
    y_proba = model.predict_proba(Xte_s)[:, 1]
    metrics = {
        "accuracy":  round(float(accuracy_score(y_test, y_pred)), 4),
        "roc_auc":   round(float(roc_auc_score(y_test, y_proba)), 4),
        "f1_score":  round(float(f1_score(y_test, y_pred)), 4),
        "oob_score": round(float(model.oob_score_), 4),
    }

    # 8. Save all artefacts into versioned folder
    _save(store / "model.pkl",          model)
    _save(store / "scaler.pkl",         scaler)
    _save(store / "imputer_stats.pkl",  imputer_stats)
    _save(store / "label_encoders.pkl", label_encoders)
    _save(store / "final_features.pkl", FEATURE_COLUMNS)

    manifest = {
        "version":    version,
        "n_samples":  n_samples,
        "n_features": len(FEATURE_COLUMNS),
        "best_params":search.best_params_,
        **metrics,
        "created_at": datetime.utcnow().isoformat(),
    }
    _save(store / "manifest.pkl", manifest)

    # Also write a human-readable JSON copy
    with open(store / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Saved artefacts to {store} | ROC-AUC={metrics['roc_auc']}")
    return manifest


def _save(path: Path, obj) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def get_latest_version() -> str | None:
    """Return the most recently created version folder name, or None."""
    if not MODELS_STORE.exists():
        return None
    versions = sorted(
        [d.name for d in MODELS_STORE.iterdir() if d.is_dir() and d.name.startswith("v")],
        reverse=True,
    )
    return versions[0] if versions else None
