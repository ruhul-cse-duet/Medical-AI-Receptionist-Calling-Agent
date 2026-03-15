"""
Medical AI Receptionist — MongoDB Models (Pydantic)
Collections:  tenants | users | patients | appointments | calls
SaaS: each company (tenant) has its own data; calls/appointments/patients are scoped by tenant_id.
"""
from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import uuid


def new_id() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Enums ─────────────────────────────────────────────────────────────────────
class CompanyType(str, Enum):
    """Business type for tenant; drives agent behavior and templates."""
    MEDICAL_DIAGNOSTIC = "medical_diagnostic"
    DENTAL_CLINIC = "dental_clinic"
    PERSONAL_DOCTOR = "personal_doctor"
    HOSPITAL_CLINIC = "hospital_clinic"
    HOTEL = "hotel"
    SPA_WELLNESS = "spa_wellness"
    OTHER = "other"


class AppointmentStatus(str, Enum):
    SCHEDULED  = "scheduled"
    CONFIRMED  = "confirmed"
    CANCELLED  = "cancelled"
    COMPLETED  = "completed"
    NO_SHOW    = "no_show"


class CallDirection(str, Enum):
    INBOUND  = "inbound"
    OUTBOUND = "outbound"


class CallStatus(str, Enum):
    INITIATED   = "initiated"
    IN_PROGRESS = "in_progress"
    COMPLETED   = "completed"
    FAILED      = "failed"
    NO_ANSWER   = "no_answer"


# ── Tenant (Company / Organization) ─────────────────────────────────────────────
class Tenant(BaseModel):
    """One company/organization using the AI receptionist (SaaS tenant). International: region/country + timezone."""
    id: str = Field(default_factory=new_id)
    company_type: CompanyType = CompanyType.MEDICAL_DIAGNOSTIC
    name: str
    phone: str                             # E.164 main line
    address: str = ""
    business_hours: str = ""                # e.g. "Sat–Thu 9 AM–8 PM"
    # Region/country for international SaaS (ISO 3166-1 alpha-2)
    country: str = "BD"
    # Locale for date/time and language (e.g. en-US, bn-BD, en-GB)
    locale: str = "en-US"
    # IANA timezone (e.g. Asia/Dhaka, America/New_York, Europe/London)
    timezone: str = "Asia/Dhaka"
    # Twilio: the number that rings for this tenant (inbound → resolve tenant by this)
    twilio_phone_number: str = ""
    # Agent behavior
    agent_backend: str = "crewai"           # crewai | simple
    receptionist_name: str = "Lisa"         # persona name
    # Optional: staff/doctors/services (JSON or list); structure depends on company_type
    staff_list: List[Dict[str, Any]] = Field(default_factory=list)
    extra_info: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


# ── User (company admin / dashboard) ───────────────────────────────────────────
class User(BaseModel):
    """User belonging to a tenant (for dashboard login)."""
    id: str = Field(default_factory=new_id)
    tenant_id: str
    email: str
    password_hash: str = ""
    name: str = ""
    role: str = "admin"
    is_active: bool = True
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


# ── Patient ───────────────────────────────────────────────────────────────────
class Patient(BaseModel):
    id: str = Field(default_factory=new_id)
    tenant_id: str = ""                     # SaaS: which company this patient belongs to
    name: str
    phone: str                             # E.164
    email: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    address: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ── Appointment ───────────────────────────────────────────────────────────────
class Appointment(BaseModel):
    id: str = Field(default_factory=new_id)
    tenant_id: str = ""                     # SaaS: which company
    patient_id: str
    patient_name: str
    patient_phone: str
    doctor_name: str
    treatment_title: str                   # e.g. "General Checkup", "Cardiology Consult"
    scheduled_at: datetime                 # appointment date & time (UTC)
    status: AppointmentStatus = AppointmentStatus.SCHEDULED
    notes: Optional[str] = None
    reminder_sent: bool = False
    reminder_sent_at: Optional[datetime] = None
    call_id: Optional[str] = None          # call that booked this appointment
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


# ── Call ──────────────────────────────────────────────────────────────────────
class Call(BaseModel):
    id: str = Field(default_factory=new_id)
    tenant_id: str = ""                     # SaaS: which company this call belongs to
    patient_id: Optional[str] = None
    patient_phone: str
    direction: CallDirection
    status: CallStatus = CallStatus.INITIATED
    twilio_call_sid: Optional[str] = None
    duration_seconds: Optional[int] = None
    transcript: Optional[str] = None
    summary: Optional[str] = None
    appointment_booked: bool = False
    appointment_id: Optional[str] = None
    is_reminder_call: bool = False
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=utcnow)
