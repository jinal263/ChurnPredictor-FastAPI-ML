"""
app/ml/predict.py
Load pickles from the active versioned folder and score one applicant.
"""
from __future__ import annotations
import pickle
import logging
from pathlib import Path
import pandas as pd

from app.ml.feature_engineering import transform
from app.ml.train_pipeline import MODELS_STORE, get_latest_version

logger = logging.getLogger(__name__)

# Module-level cache so we only load from disk once per process
_cache: dict = {}


def _load_artefacts(version: str) -> dict:
    global _cache
    if _cache.get("version") == version:
        return _cache

    store = MODELS_STORE / version
    if not store.exists():
        raise FileNotFoundError(f"Model store not found: {store}")

    logger.info(f"Loading artefacts from {store}")
    _cache = {
        "version":       version,
        "model":         _load(store / "model.pkl"),
        "scaler":        _load(store / "scaler.pkl"),
        "imputer_stats": _load(store / "imputer_stats.pkl"),
        "manifest":      _load(store / "manifest.pkl"),
    }
    return _cache


def _load(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def score_applicant(applicant_dict: dict) -> dict:
    """
    Score a single applicant dict.
    Returns churn_prediction, churn_probability, risk_label, model_version.
    Raises RuntimeError if no trained model exists.
    """
    version = get_latest_version()
    if version is None:
        raise RuntimeError("No trained model found. Run training first.")

    art = _load_artefacts(version)

    df  = pd.DataFrame([applicant_dict])
    X   = transform(df, art["imputer_stats"])
    Xs  = art["scaler"].transform(X)

    proba = float(art["model"].predict_proba(Xs)[0, 1])
    pred  = proba >= 0.5
    risk  = "Low" if proba < 0.35 else ("Medium" if proba < 0.65 else "High")

    return {
        "churn_prediction":  pred,
        "churn_probability": round(proba, 4),
        "risk_label":        risk,
        "model_version":     version,
    }


def invalidate_cache() -> None:
    """Call this after a new model is trained to force reload."""
    global _cache
    _cache = {}
