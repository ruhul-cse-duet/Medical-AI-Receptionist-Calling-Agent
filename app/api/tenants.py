"""
Tenants API — SaaS company registration and management
Companies (hotel, medical, dental, spa, etc.) create an account and get their AI receptionist.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.db.base import tenants_col, users_col
from app.db.models import CompanyType, Tenant, User, utcnow

router = APIRouter(prefix="/tenants", tags=["Tenants"])
logger = logging.getLogger(__name__)


def _hash_password(password: str) -> str:
    if not password:
        return ""
    return hashlib.sha256((settings.SECRET_KEY + password).encode()).hexdigest()


# ── Request/Response schemas ───────────────────────────────────────────────────
class TenantRegisterRequest(BaseModel):
    company_type: str = "medical_diagnostic"  # medical_diagnostic | dental_clinic | personal_doctor | hotel | spa_wellness | other
    name: str = Field(..., min_length=1)
    phone: str = Field(..., min_length=1)
    address: str = ""
    business_hours: str = ""
    country: str = "BD"                       # ISO 3166-1 alpha-2 (e.g. BD, US, GB)
    locale: str = "en-US"                     # e.g. en-US, bn-BD, en-GB
    timezone: str = "Asia/Dhaka"              # IANA timezone (e.g. Asia/Dhaka, America/New_York)
    twilio_phone_number: str = ""              # E.164; the number that will receive inbound calls for this tenant
    receptionist_name: str = "Lisa"
    staff_list: List[Dict[str, Any]] = Field(default_factory=list)  # e.g. [{"name": "Dr. X", "specialty": "Cardiology"}]
    extra_info: Dict[str, Any] = Field(default_factory=dict)
    # First admin user (optional)
    admin_email: Optional[str] = None
    admin_name: str = ""
    admin_password: Optional[str] = None


class TenantResponse(BaseModel):
    id: str
    company_type: str
    name: str
    phone: str
    address: str
    business_hours: str
    country: str
    locale: str
    timezone: str
    twilio_phone_number: str
    receptionist_name: str
    is_active: bool
    created_at: str


# ── Endpoints ──────────────────────────────────────────────────────────────────
@router.post("/register", summary="Register a new company (tenant)")
async def register_tenant(body: TenantRegisterRequest):
    """
    Create a new company account. After registration, configure your Twilio number
    to point to this app; set that number in twilio_phone_number so inbound calls
    are resolved to this tenant and the AI receptionist uses this company's info.
    """
    company_type = body.company_type.strip().lower()
    if company_type not in [e.value for e in CompanyType]:
        company_type = CompanyType.OTHER.value

    tenant = Tenant(
        company_type=CompanyType(company_type),
        name=body.name.strip(),
        phone=body.phone.strip(),
        address=(body.address or "").strip(),
        business_hours=(body.business_hours or "").strip(),
        country=(body.country or "BD").strip().upper()[:2],
        locale=(body.locale or "en-US").strip(),
        timezone=(body.timezone or "Asia/Dhaka").strip(),
        twilio_phone_number=(body.twilio_phone_number or "").strip().replace(" ", ""),
        receptionist_name=(body.receptionist_name or "Lisa").strip(),
        staff_list=body.staff_list or [],
        extra_info=body.extra_info or {},
    )
    await tenants_col().insert_one(tenant.model_dump())
    logger.info("Tenant registered: id=%s name=%s type=%s", tenant.id, tenant.name, tenant.company_type.value)

    if body.admin_email:
        user = User(
            tenant_id=tenant.id,
            email=body.admin_email.strip().lower(),
            name=(body.admin_name or "").strip(),
            password_hash=_hash_password(body.admin_password or ""),
            role="admin",
        )
        await users_col().insert_one(user.model_dump())
        logger.info("Admin user created for tenant=%s email=%s", tenant.id, user.email)

    return {
        "tenant_id": tenant.id,
        "message": (
            "Company registered successfully. "
            "Set your Twilio inbound number in twilio_phone_number and point it to this app's webhook."
        ),
        "tenant": {
            "id": tenant.id,
            "company_type": tenant.company_type.value,
            "name": tenant.name,
            "phone": tenant.phone,
            "country": tenant.country,
            "locale": tenant.locale,
            "timezone": tenant.timezone,
            "twilio_phone_number": tenant.twilio_phone_number,
            "receptionist_name": tenant.receptionist_name,
        },
    }


@router.get("/{tenant_id}", summary="Get tenant by ID")
async def get_tenant(tenant_id: str):
    doc = await tenants_col().find_one({"id": tenant_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Tenant not found")
    return doc


@router.patch("/{tenant_id}", summary="Update tenant (partial)")
async def update_tenant(tenant_id: str, body: dict):
    """Update tenant fields (name, phone, address, business_hours, staff_list, etc.)."""
    existing = await tenants_col().find_one({"id": tenant_id})
    if not existing:
        raise HTTPException(status_code=404, detail="Tenant not found")
    allowed = {
        "name", "phone", "address", "business_hours", "country", "locale", "timezone",
        "twilio_phone_number", "receptionist_name", "staff_list", "extra_info", "is_active",
    }
    update = {k: v for k, v in body.items() if k in allowed}
    if not update:
        raise HTTPException(status_code=400, detail="No allowed fields to update")
    update["updated_at"] = utcnow()
    await tenants_col().update_one({"id": tenant_id}, {"$set": update})
    doc = await tenants_col().find_one({"id": tenant_id}, {"_id": 0})
    return doc
