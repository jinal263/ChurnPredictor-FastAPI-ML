"""
app/main.py — FastAPI application entry point.
Wires routers, middleware, exception handlers, lifespan, and HTMX routes.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import asyncio
from fastapi.responses import JSONResponse
from app.api import predict, train, applications, auth_router
from app.database import init_db
from app.errors import ERRORS
from app.middleware.rate_limit import RateLimitMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Lifespan ─────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Startup — initialising database…")
    await init_db()
    yield
    logger.info("Shutdown.")


# ── App ──────────────────────────────────────────────────────────

app = FastAPI(
    title="CC Underwriting — Churn Prediction API",
    version="1.0.0",
    description="Random Forest churn prediction with versioned model store.",
    lifespan=lifespan,
)

# ── Middleware ────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)
app.add_middleware(RateLimitMiddleware, max_requests=60, window_seconds=60)


# ── Exception handlers ────────────────────────────────────────────

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    from fastapi.responses import JSONResponse
    detail = exc.detail
    if isinstance(detail, dict):
        return JSONResponse(status_code=exc.status_code, content=detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error_code": "ERR_013", "message": str(detail), "detail": str(detail)},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    from fastapi.responses import JSONResponse
    errors = exc.errors()
    first  = errors[0] if errors else {}
    field  = " → ".join(str(x) for x in first.get("loc", []))
    return JSONResponse(
        status_code=422,
        content={
            "error_code": "ERR_002",
            "message":    ERRORS["ERR_002"].message,
            "detail":     f"Field '{field}': {first.get('msg', 'invalid')}",
            "all_errors": errors,
        },
    )


# ── API Routers ───────────────────────────────────────────────────

app.include_router(predict.router,      prefix="/api", tags=["predict"])
app.include_router(train.router,        prefix="/api", tags=["train"])
app.include_router(applications.router, prefix="/api", tags=["applications"])
app.include_router(auth_router.router, prefix="/api", tags=["auth"])


# ── Health ────────────────────────────────────────────────────────

@app.get("/health", tags=["health"])
async def health():
    from app.ml.train_pipeline import get_latest_version
    version = get_latest_version()
    return {"status": "ok", "active_model": version, "model_ready": version is not None}

# ── add a timeout middleware ────────────────────────────────────────────────────────
@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        return await asyncio.wait_for(call_next(request), timeout=30.0)
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=504,
            content={
                "error_code": "ERR_014",
                "message": "Request timed out",
                "detail": "The request took too long. Try again."
            }
        )
# ── HTMX frontend ─────────────────────────────────────────────────
import os
_BASE = os.path.dirname(os.path.abspath(__file__))

def _template(name: str) -> str:
    return os.path.join(_BASE, "..", "templates", name)

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index():
    with open(_template("index.html"), encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/partials/result", response_class=HTMLResponse, include_in_schema=False)
async def partial_result():
    with open(_template("partials/result.html"), encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/partials/history", response_class=HTMLResponse, include_in_schema=False)
async def partial_history():
    with open(_template("partials/history.html"), encoding="utf-8") as f:
        return HTMLResponse(f.read())
    
@app.get("/login", response_class=HTMLResponse, include_in_schema=False)
async def login_page():
    with open(_template("login.html"), encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/signup", response_class=HTMLResponse, include_in_schema=False)
async def signup_page():
    with open(_template("signup.html"), encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/train", response_class=HTMLResponse, include_in_schema=False)
async def train_page():
    with open(_template("train.html"), encoding="utf-8") as f:
        return HTMLResponse(f.read())