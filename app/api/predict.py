"""
app/api/predict.py
POST /api/predict        — single applicant
POST /api/predict/batch  — up to 500 applicants
"""
from __future__ import annotations
import logging
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Application
from app.schemas import ApplicantIn, PredictionOut, BatchPredictIn, BatchPredictOut, ErrorOut
from app.errors import raise_error
from app.ml.predict import score_applicant

router = APIRouter()
logger = logging.getLogger(__name__)


async def _run_and_save(applicant: ApplicantIn, db: AsyncSession) -> PredictionOut:
    """Score one applicant and persist the result."""
    try:
        result = score_applicant(applicant.model_dump())
    except RuntimeError as exc:
        raise_error("ERR_004", str(exc))
    except Exception as exc:
        logger.exception("Inference error")
        raise_error("ERR_005", str(exc))

    rec = Application(
        customer_id=applicant.customer_id,
        age=applicant.age,
        income=applicant.income,
        credit_score=applicant.credit_score,
        tenure_months=applicant.tenure_months,
        monthly_charges=applicant.monthly_charges,
        num_products=applicant.num_products,
        support_calls=applicant.support_calls,
        complaints_last_6m=applicant.complaints_last_6m,
        avg_monthly_usage_gb=applicant.avg_monthly_usage_gb,
        payment_delay_days=applicant.payment_delay_days,
        gender=applicant.gender,
        education=applicant.education,
        marital_status=applicant.marital_status,
        contract=applicant.contract,
        signup_month=applicant.signup_month,
        signup_year=applicant.signup_year,
        churn_prediction=result["churn_prediction"],
        churn_probability=result["churn_probability"],
        risk_label=result["risk_label"],
        model_version=result["model_version"],
    )
    db.add(rec)
    await db.commit()
    await db.refresh(rec)

    return PredictionOut(
        record_id=rec.id,
        customer_id=applicant.customer_id,
        churn_prediction=result["churn_prediction"],
        churn_probability=result["churn_probability"],
        risk_label=result["risk_label"],
        model_version=result["model_version"],
    )


@router.post(
    "/predict",
    response_model=PredictionOut,
    responses={
        422: {"model": ErrorOut},
        503: {"model": ErrorOut},
        500: {"model": ErrorOut},
    },
    summary="Predict churn for a single applicant",
)
async def predict(applicant: ApplicantIn, db: AsyncSession = Depends(get_db)):
    return await _run_and_save(applicant, db)


@router.post(
    "/predict/batch",
    response_model=BatchPredictOut,
    responses={
        400: {"model": ErrorOut},
        503: {"model": ErrorOut},
    },
    summary="Batch predict churn for up to 500 applicants",
)
async def predict_batch(req: BatchPredictIn, db: AsyncSession = Depends(get_db)):
    if len(req.customers) > 500:
        raise_error("ERR_009")

    results = []
    for applicant in req.customers:
        results.append(await _run_and_save(applicant, db))

    return BatchPredictOut(
        results=results,
        total=len(results),
        model_version=results[0].model_version if results else "unknown",
    )
