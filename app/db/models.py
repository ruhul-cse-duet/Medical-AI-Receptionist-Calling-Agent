"""
Medical AI Receptionist — MongoDB Models (Pydantic)
Collections: tenants | users | doctors | patients | appointments | calls
SaaS: each company (tenant) has its own data scoped by tenant_id.
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
    MEDICAL_DIAGNOSTIC = "medical_diagnostic"
    DENTAL_CLINIC      = "dental_clinic"
    PERSONAL_DOCTOR    = "personal_doctor"
    HOSPITAL_CLINIC    = "hospital_clinic"
    HOTEL              = "hotel"
    SPA_WELLNESS       = "spa_wellness"
    OTHER              = "other"


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


# ── Tenant ────────────────────────────────────────────────────────────────────
class Tenant(BaseModel):
    id: str = Field(default_factory=new_id)
    company_type: CompanyType = CompanyType.MEDICAL_DIAGNOSTIC
    name: str
    phone: str
    address: str = ""
    business_hours: str = ""
    country: str = "BD"
    locale: str = "en-US"
    timezone: str = "Asia/Dhaka"
    twilio_phone_number: str = ""
    agent_backend: str = "crewai"
    receptionist_name: str = "Lisa"
    staff_list: List[Dict[str, Any]] = Field(default_factory=list)
    extra_info: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


# ── User ──────────────────────────────────────────────────────────────────────
class User(BaseModel):
    id: str = Field(default_factory=new_id)
    tenant_id: str
    email: str
    password_hash: str = ""
    name: str = ""
    role: str = "admin"
    is_active: bool = True
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


# ── Doctor ────────────────────────────────────────────────────────────────────
class Doctor(BaseModel):
    """A doctor/staff member registered under a tenant (clinic)."""
    id: str = Field(default_factory=new_id)
    tenant_id: str
    name: str                              # e.g. "Dr. Sarah Ahmed"
    specialty: str = ""                    # e.g. "Cardiology"
    qualification: str = ""               # e.g. "MBBS, MD"
    experience_years: Optional[int] = None
    available_days: str = ""              # e.g. "Sun-Thu"
    available_time: str = ""              # e.g. "9:00 AM - 5:00 PM"
    consultation_fee: str = ""            # e.g. "500 BDT"
    bio: str = ""                         # short description
    phone: str = ""                        # direct line (optional)
    email: Optional[str] = None
    is_active: bool = True
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


# ── Patient ───────────────────────────────────────────────────────────────────
class Patient(BaseModel):
    id: str = Field(default_factory=new_id)
    tenant_id: str = ""
    name: str
    phone: str
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
    tenant_id: str = ""
    patient_id: str
    patient_name: str
    patient_phone: str
    doctor_id: Optional[str] = None        # references Doctor.id
    doctor_name: str
    treatment_title: str
    scheduled_at: datetime
    status: AppointmentStatus = AppointmentStatus.SCHEDULED
    notes: Optional[str] = None
    reminder_sent: bool = False
    reminder_sent_at: Optional[datetime] = None
    call_id: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


# ── Call ──────────────────────────────────────────────────────────────────────
class Call(BaseModel):
    id: str = Field(default_factory=new_id)
    tenant_id: str = ""
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
    updated_at: datetime = Field(default_factory=utcnow)
