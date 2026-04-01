"""
Error code registry — ERR_001 through ERR_013.
Every API error maps to one of these codes so the frontend
and any API client can handle errors programmatically.
"""
from dataclasses import dataclass
from fastapi import HTTPException


@dataclass(frozen=True)
class AppError:
    code: str           # e.g. "ERR_001"
    http_status: int    # HTTP status code
    message: str        # human-readable summary


# ── Registry ────────────────────────────────────────────────────
ERRORS = {
    # Validation errors
    "ERR_001": AppError("ERR_001", 422, "Missing required field"),
    "ERR_002": AppError("ERR_002", 422, "Invalid field value"),
    "ERR_003": AppError("ERR_003", 422, "Invalid field type"),

    # Model errors
    "ERR_004": AppError("ERR_004", 503, "No trained model available — run training first"),
    "ERR_005": AppError("ERR_005", 500, "Model inference failed"),
    "ERR_006": AppError("ERR_006", 500, "Model training failed"),
    "ERR_007": AppError("ERR_007", 404, "Training run not found"),

    # Request errors
    "ERR_008": AppError("ERR_008", 429, "Rate limit exceeded"),
    "ERR_009": AppError("ERR_009", 400, "Batch size exceeds maximum (500)"),
    "ERR_010": AppError("ERR_010", 404, "Application record not found"),

    # Data errors
    "ERR_011": AppError("ERR_011", 400, "CSV file is empty or malformed"),
    "ERR_012": AppError("ERR_012", 400, "CSV missing required columns"),

    # Server errors
    "ERR_013": AppError("ERR_013", 500, "Internal server error"),
}


def raise_error(code: str, detail: str = None) -> None:
    """Raise an HTTPException using a registered error code."""
    err = ERRORS.get(code)
    if err is None:
        err = ERRORS["ERR_013"]

    raise HTTPException(
        status_code=err.http_status,
        detail={
            "error_code": err.code,
            "message":    err.message,
            "detail":     detail or err.message,
        },
    )


def error_response(code: str, detail: str = None) -> dict:
    """Return an error dict without raising (useful for batch results)."""
    err = ERRORS.get(code, ERRORS["ERR_013"])
    return {
        "error_code": err.code,
        "message":    err.message,
        "detail":     detail or err.message,
    }
