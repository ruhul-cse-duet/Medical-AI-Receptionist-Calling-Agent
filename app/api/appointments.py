"""
Appointments API — CRUD for medical appointments
"""
from __future__ import annotations
import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.db.base import appointments_col
from app.db.models import Appointment, AppointmentStatus, utcnow

router = APIRouter(prefix="/appointments", tags=["Appointments"])
logger = logging.getLogger(__name__)


class AppointmentCreate(BaseModel):
    patient_name: str
    patient_phone: str
    doctor_name: str
    treatment_title: str
    scheduled_at: datetime
    notes: Optional[str] = None


class AppointmentUpdate(BaseModel):
    doctor_name: Optional[str] = None
    treatment_title: Optional[str] = None
    scheduled_at: Optional[datetime] = None
    status: Optional[AppointmentStatus] = None
    notes: Optional[str] = None


@router.post("/", summary="Book a new appointment")
async def create_appointment(body: AppointmentCreate):
    appt = Appointment(**body.model_dump())
    await appointments_col().insert_one(appt.model_dump())
    return {"appointment_id": appt.id, "status": appt.status}


@router.get("/", summary="List all appointments")
async def list_appointments(
    patient_phone: Optional[str] = Query(None),
    status: Optional[AppointmentStatus] = Query(None),
    skip: int = 0,
    limit: int = 20,
):
    filt: dict = {}
    if patient_phone:
        filt["patient_phone"] = patient_phone
    if status:
        filt["status"] = status.value
    cursor = appointments_col().find(filt, {"_id": 0}).skip(skip).limit(limit)
    return await cursor.to_list(length=limit)


@router.get("/{appointment_id}", summary="Get appointment by ID")
async def get_appointment(appointment_id: str):
    doc = await appointments_col().find_one({"id": appointment_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Appointment not found")
    return doc


@router.patch("/{appointment_id}", summary="Update / reschedule appointment")
async def update_appointment(appointment_id: str, body: AppointmentUpdate):
    update = {k: v for k, v in body.model_dump().items() if v is not None}
    if not update:
        raise HTTPException(status_code=400, detail="Nothing to update")
    update["updated_at"] = utcnow()
    result = await appointments_col().update_one(
        {"id": appointment_id}, {"$set": update}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Appointment not found")
    return {"updated": True}


@router.delete("/{appointment_id}", summary="Cancel appointment")
async def cancel_appointment(appointment_id: str):
    result = await appointments_col().update_one(
        {"id": appointment_id},
        {"$set": {"status": AppointmentStatus.CANCELLED.value, "updated_at": utcnow()}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Appointment not found")
    return {"cancelled": True}
