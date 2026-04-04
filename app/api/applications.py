"""
app/api/applications.py
GET    /api/applications/      — paginated history
DELETE /api/applications/{id}  — delete one record

CHANGES FROM YOUR ORIGINAL — only 4 lines added, nothing removed:
  1. Added `Depends` to the fastapi import (was missing)
  2. Added `from app.auth import require_auth, require_admin`
  3. GET  /api/applications/ → require_auth   (admin + viewer can see history)
  4. DELETE /api/applications/{id} → require_admin (only admin can delete)

WHY two different levels here:
  It makes sense for a viewer to be able to see the prediction history —
  that's read-only information. But deleting records is a destructive
  action that only an admin should be able to do.

  This is a common real-world pattern called "role-based access control"
  (RBAC): the same resource (application records) has different permission
  levels depending on what you want to DO with it.
"""
from __future__ import annotations
import math
from fastapi import APIRouter, Depends, Query                  # ← added Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, func

from app.database import get_db
from app.models import Application
from app.schemas import ApplicationOut, PaginatedApplications, ErrorOut
from app.errors import raise_error
from app.auth import require_auth, require_admin               # ← NEW

router = APIRouter()


@router.get(
    "/applications/",
    response_model=PaginatedApplications,
    summary="Paginated prediction history",
)
async def list_applications(
    page:      int = Query(1,  ge=1,         description="Page number (starts at 1)"),
    page_size: int = Query(10, ge=1, le=100, description="Results per page (max 100)"),
    db: AsyncSession = Depends(get_db),
    user: str = Depends(require_auth),                        # ← NEW (admin + viewer)
):
    # ── Count total records ──────────────────────────────────────
    count_result = await db.execute(select(func.count()).select_from(Application))
    total        = count_result.scalar_one()

    # ── Calculate pagination values ──────────────────────────────
    total_pages = max(1, math.ceil(total / page_size))
    page        = min(page, total_pages)
    offset      = (page - 1) * page_size

    # ── Fetch the page of records ────────────────────────────────
    result = await db.execute(
        select(Application)
        .order_by(Application.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    items = result.scalars().all()

    return PaginatedApplications(
        items       = items,
        total       = total,
        page        = page,
        page_size   = page_size,
        total_pages = total_pages,
        has_next    = page < total_pages,
        has_prev    = page > 1,
    )


@router.delete(
    "/applications/{app_id}",
    status_code=204,
    responses={
        404: {"model": ErrorOut},
        401: {"model": ErrorOut},
        403: {"model": ErrorOut},
    },
    summary="Delete a prediction record — admin only",
)
async def delete_application(
    app_id: int,
    db: AsyncSession = Depends(get_db),
    user: str = Depends(require_admin),                       # ← NEW (admin only)
):
    result = await db.execute(select(Application).where(Application.id == app_id))
    rec    = result.scalar_one_or_none()
    if rec is None:
        raise_error("ERR_010", f"Application {app_id} not found")

    await db.execute(delete(Application).where(Application.id == app_id))
    await db.commit()