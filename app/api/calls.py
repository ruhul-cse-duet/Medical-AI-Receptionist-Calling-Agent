"""
Calls API — initiate outbound calls
"""
from __future__ import annotations
import uuid
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.db.base import calls_col
from app.db.models import Call, CallDirection, CallStatus, utcnow
from app.services.twilio_service import TwilioService

router = APIRouter(prefix="/calls", tags=["Calls"])
logger = logging.getLogger(__name__)


class OutboundCallRequest(BaseModel):
    patient_phone: str      # E.164 format
    patient_name: str = ""


@router.post("/outbound", summary="Initiate outbound AI call to a patient")
async def initiate_outbound_call(body: OutboundCallRequest):
    call_id = str(uuid.uuid4())
    call_doc = Call(
        id=call_id,
        patient_phone=body.patient_phone,
        direction=CallDirection.OUTBOUND,
        status=CallStatus.INITIATED,
        started_at=utcnow(),
    )
    await calls_col().insert_one(call_doc.model_dump())

    twilio = TwilioService()
    try:
        sid = await twilio.initiate_outbound_call(
            to_phone=body.patient_phone,
            call_id=call_id,
        )
        await calls_col().update_one(
            {"id": call_id},
            {"$set": {"twilio_call_sid": sid, "status": CallStatus.INITIATED.value}},
        )
    except Exception as exc:
        logger.error("Outbound call failed: %s", exc)
        await calls_col().update_one(
            {"id": call_id}, {"$set": {"status": CallStatus.FAILED.value}}
        )
        raise HTTPException(status_code=502, detail=f"Twilio error: {exc}")

    return {"call_id": call_id, "twilio_call_sid": sid, "status": "initiated"}


@router.get("/{call_id}", summary="Get call record")
async def get_call(call_id: str):
    doc = await calls_col().find_one({"id": call_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Call not found")
    return doc
