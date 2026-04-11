"""
Doctors API — Register and manage doctors under a tenant (clinic).

Routes:
  POST   /v1/doctors/register        Register a new doctor
  GET    /v1/doctors?tenant_id=...   List doctors for a tenant
  GET    /v1/doctors/{doctor_id}     Get a single doctor
  PATCH  /v1/doctors/{doctor_id}     Update doctor details
  DELETE /v1/doctors/{doctor_id}     Deactivate (soft-delete) a doctor
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.db.base import doctors_col, tenants_col
from app.db.models import Doctor, utcnow

router = APIRouter(prefix="/doctors", tags=["Doctors"])
logger = logging.getLogger(__name__)


# ── Request / Response schemas ─────────────────────────────────────────────────
class DoctorRegisterRequest(BaseModel):
    tenant_id: str = Field(..., description="ID of the clinic/tenant this doctor belongs to")
    name: str = Field(..., min_length=1, description="Full name, e.g. 'Dr. Sarah Ahmed'")
    specialty: str = ""
    qualification: str = ""
    experience_years: Optional[int] = None
    available_days: str = ""          # e.g. "Sun–Thu"
    available_time: str = ""          # e.g. "9 AM – 5 PM"
    consultation_fee: str = ""        # e.g. "500 BDT"
    bio: str = ""
    phone: str = ""
    email: Optional[str] = None


class DoctorUpdateRequest(BaseModel):
    name: Optional[str] = None
    specialty: Optional[str] = None
    qualification: Optional[str] = None
    experience_years: Optional[int] = None
    available_days: Optional[str] = None
    available_time: Optional[str] = None
    consultation_fee: Optional[str] = None
    bio: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    is_active: Optional[bool] = None


# ── Endpoints ──────────────────────────────────────────────────────────────────
@router.post("/register", summary="Register a new doctor under a tenant")
async def register_doctor(body: DoctorRegisterRequest):
    """
    Register a doctor for a clinic (tenant). The doctor will be stored in the
    'doctors' collection and will appear when the AI receptionist lists available doctors.
    """
    # Verify tenant exists
    tenant = await tenants_col().find_one({"id": body.tenant_id})
    if not tenant:
        raise HTTPException(status_code=404, detail=f"Tenant '{body.tenant_id}' not found")

    doctor = Doctor(
        tenant_id=body.tenant_id,
        name=body.name.strip(),
        specialty=(body.specialty or "").strip(),
        qualification=(body.qualification or "").strip(),
        experience_years=body.experience_years,
        available_days=(body.available_days or "").strip(),
        available_time=(body.available_time or "").strip(),
        consultation_fee=(body.consultation_fee or "").strip(),
        bio=(body.bio or "").strip(),
        phone=(body.phone or "").strip(),
        email=body.email,
    )
    await doctors_col().insert_one(doctor.model_dump())
    logger.info("Doctor registered: id=%s name=%s tenant=%s", doctor.id, doctor.name, doctor.tenant_id)

    # Also sync into tenant's staff_list for backward compatibility
    staff_entry = {"name": doctor.name, "specialty": doctor.specialty, "doctor_id": doctor.id}
    await tenants_col().update_one(
        {"id": body.tenant_id},
        {"$push": {"staff_list": staff_entry}, "$set": {"updated_at": utcnow()}},
    )

    return {
        "doctor_id": doctor.id,
        "message": "Doctor registered successfully.",
        "doctor": {
            "id": doctor.id,
            "tenant_id": doctor.tenant_id,
            "name": doctor.name,
            "specialty": doctor.specialty,
            "qualification": doctor.qualification,
            "experience_years": doctor.experience_years,
            "available_days": doctor.available_days,
            "available_time": doctor.available_time,
            "consultation_fee": doctor.consultation_fee,
        },
    }


@router.get("", summary="List doctors for a tenant")
async def list_doctors(tenant_id: str = Query(..., description="Tenant ID to filter doctors")):
    cursor = doctors_col().find({"tenant_id": tenant_id, "is_active": True}, {"_id": 0})
    docs = await cursor.to_list(length=200)
    return {"tenant_id": tenant_id, "count": len(docs), "doctors": docs}


@router.get("/{doctor_id}", summary="Get a doctor by ID")
async def get_doctor(doctor_id: str):
    doc = await doctors_col().find_one({"id": doctor_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Doctor not found")
    return doc


@router.patch("/{doctor_id}", summary="Update doctor details")
async def update_doctor(doctor_id: str, body: DoctorUpdateRequest):
    existing = await doctors_col().find_one({"id": doctor_id})
    if not existing:
        raise HTTPException(status_code=404, detail="Doctor not found")
    update = {k: v for k, v in body.model_dump(exclude_none=True).items()}
    if not update:
        raise HTTPException(status_code=400, detail="No fields to update")
    update["updated_at"] = utcnow()
    await doctors_col().update_one({"id": doctor_id}, {"$set": update})

    # If name or specialty changed, sync tenant staff_list entry
    if "name" in update or "specialty" in update:
        tenant_id = existing.get("tenant_id", "")
        new_name = update.get("name", existing.get("name"))
        new_spec = update.get("specialty", existing.get("specialty"))
        await tenants_col().update_one(
            {"id": tenant_id, "staff_list.doctor_id": doctor_id},
            {"$set": {"staff_list.$.name": new_name, "staff_list.$.specialty": new_spec}},
        )
    return await doctors_col().find_one({"id": doctor_id}, {"_id": 0})


@router.delete("/{doctor_id}", summary="Deactivate a doctor (soft delete)")
async def deactivate_doctor(doctor_id: str):
    existing = await doctors_col().find_one({"id": doctor_id})
    if not existing:
        raise HTTPException(status_code=404, detail="Doctor not found")
    await doctors_col().update_one(
        {"id": doctor_id},
        {"$set": {"is_active": False, "updated_at": utcnow()}},
    )
    return {"message": "Doctor deactivated", "doctor_id": doctor_id}
