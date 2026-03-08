"""
Medical AI Receptionist — MongoDB Models (Pydantic)
Collections:  patients | appointments | calls
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


# ── Patient ───────────────────────────────────────────────────────────────────
class Patient(BaseModel):
    id: str = Field(default_factory=new_id)
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
