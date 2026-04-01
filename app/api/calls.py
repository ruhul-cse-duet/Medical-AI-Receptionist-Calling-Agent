"""
Calls API — view call records (inbound only)
"""
from __future__ import annotations
import logging

from fastapi import APIRouter, HTTPException

from app.db.base import calls_col

router = APIRouter(prefix="/calls", tags=["Calls"])
logger = logging.getLogger(__name__)


@router.get("/{call_id}", summary="Get call record")
async def get_call(call_id: str):
    """Retrieve details of a specific call by ID."""
    doc = await calls_col().find_one({"id": call_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Call not found")
    return doc
