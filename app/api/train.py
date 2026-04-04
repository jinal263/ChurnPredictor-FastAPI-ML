"""
app/api/train.py
POST /api/train/start   — kick off background training
GET  /api/train/        — list all runs
GET  /api/train/{id}    — poll status of a training run

CHANGE FROM YOUR PREVIOUS VERSION:
  Moved GET /api/train/ to be defined BEFORE GET /api/train/{run_id}.

  WHY this matters:
    FastAPI matches routes in the order they are defined. The route
    GET /api/train/{run_id} uses a path parameter — it matches ANY
    string after /train/, including the empty string from /train/.
    So if {run_id} is defined first, /train/ is swallowed by it and
    the list endpoint is never reached, causing a 404.

    Rule to remember: always put exact/static routes BEFORE
    parameterised routes in the same router.
"""
from __future__ import annotations
import asyncio
import logging
from datetime import datetime
from fastapi import APIRouter, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from app.database import get_db
from app.models import TrainingRun
from app.schemas import TrainIn, TrainStartOut, TrainStatusOut, ErrorOut
from app.errors import raise_error
from app.ml.train_pipeline import run_training
from app.ml.predict import invalidate_cache
from app.auth import require_admin

router  = APIRouter()
logger  = logging.getLogger(__name__)


async def _background_train(run_id: int, params: TrainIn) -> None:
    """Run training in background and update DB record when done. UNCHANGED."""
    from app.database import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        try:
            loop = asyncio.get_event_loop()
            meta = await loop.run_in_executor(
                None,
                lambda: run_training(
                    n_samples=params.n_samples,
                    n_estimators=params.n_estimators,
                    max_depth=params.max_depth,
                    min_samples_split=params.min_samples_split,
                    min_samples_leaf=params.min_samples_leaf,
                )
            )

            await db.execute(update(TrainingRun).values(is_active=False))

            await db.execute(
                update(TrainingRun)
                .where(TrainingRun.id == run_id)
                .values(
                    status="done",
                    n_samples=meta["n_samples"],
                    n_features=meta["n_features"],
                    accuracy=meta["accuracy"],
                    roc_auc=meta["roc_auc"],
                    f1_score=meta["f1_score"],
                    oob_score=meta["oob_score"],
                    is_active=True,
                    finished_at=datetime.utcnow(),
                )
            )
            await db.commit()
            invalidate_cache()
            logger.info(f"Training run {run_id} done — {meta['version']}")

        except Exception as exc:
            logger.exception(f"Training run {run_id} failed")
            await db.execute(
                update(TrainingRun)
                .where(TrainingRun.id == run_id)
                .values(status="failed", error_message=str(exc), finished_at=datetime.utcnow())
            )
            await db.commit()


@router.post(
    "/train/start",
    response_model=TrainStartOut,
    responses={500: {"model": ErrorOut}, 401: {"model": ErrorOut}, 403: {"model": ErrorOut}},
    summary="Start a new training run — admin only",
)
async def start_training(
    params: TrainIn,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    user: str = Depends(require_admin),
):
    version = datetime.utcnow().strftime("v%Y%m%d_%H%M%S")
    run = TrainingRun(model_version=version, status="running")
    db.add(run)
    await db.commit()
    await db.refresh(run)
    background_tasks.add_task(_background_train, run.id, params)
    return TrainStartOut(
        model_version=version,
        status="running",
        message=f"Training started by {user}. Poll GET /api/train/{run.id} for status.",
    )


# ── LIST must come BEFORE /{run_id} ──────────────────────────────
@router.get(
    "/train/",
    response_model=list[TrainStatusOut],
    summary="List all training runs — admin only",
)
async def list_training_runs(
    db: AsyncSession = Depends(get_db),
    user: str = Depends(require_admin),
):
    result = await db.execute(
        select(TrainingRun).order_by(TrainingRun.created_at.desc()).limit(20)
    )
    return result.scalars().all()


# ── /{run_id} must come AFTER the static /train/ route ───────────
@router.get(
    "/train/{run_id}",
    response_model=TrainStatusOut,
    responses={404: {"model": ErrorOut}, 401: {"model": ErrorOut}, 403: {"model": ErrorOut}},
    summary="Get status of a training run — admin only",
)
async def get_training_status(
    run_id: int,
    db: AsyncSession = Depends(get_db),
    user: str = Depends(require_admin),
):
    result = await db.execute(select(TrainingRun).where(TrainingRun.id == run_id))
    run    = result.scalar_one_or_none()
    if run is None:
        raise_error("ERR_007", f"Training run {run_id} not found")
    return run