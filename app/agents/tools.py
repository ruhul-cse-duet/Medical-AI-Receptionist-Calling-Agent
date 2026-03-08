"""
CrewAI Tools — MongoDB operations exposed as @tool for agents
Each tool is a standalone async-compatible function the agent calls directly.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from crewai.tools import tool

from app.config import settings

logger = logging.getLogger(__name__)

# Doctor roster — extend freely
DOCTORS = [
    {"name": "Dr. Rahman",    "specialty": "General Medicine"},
    {"name": "Dr. Sultana",   "specialty": "Cardiology"},
    {"name": "Dr. Hossain",   "specialty": "Orthopedics"},
    {"name": "Dr. Ahmed",     "specialty": "Pediatrics"},
    {"name": "Dr. Chowdhury", "specialty": "Dermatology"},
    {"name": "Dr. Islam",     "specialty": "Neurology"},
]


# ── Helper: get collections without importing at module-level ─────────────────
def _appointments_col():
    from app.db.base import appointments_col
    return appointments_col()

def _patients_col():
    from app.db.base import patients_col
    return patients_col()


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 1: Check available doctors
# ─────────────────────────────────────────────────────────────────────────────
@tool("check_available_doctors")
def check_available_doctors(query: str = "") -> str:
    """
    Returns the list of doctors available at the clinic with their specialties.
    Call this when the patient asks who is available or needs a specific specialty.
    Input: any string (e.g. 'all' or specialty keyword like 'cardiology').
    """
    if query and query.strip().lower() not in ("all", "", "list"):
        keyword = query.lower()
        filtered = [d for d in DOCTORS if keyword in d["specialty"].lower()]
        if filtered:
            lines = [f"• {d['name']} — {d['specialty']}" for d in filtered]
            return "Matching doctors:\n" + "\n".join(lines)

    lines = [f"• {d['name']} — {d['specialty']}" for d in DOCTORS]
    return f"Available doctors at {settings.CLINIC_NAME}:\n" + "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 2: Book appointment
# ─────────────────────────────────────────────────────────────────────────────
@tool("book_appointment")
def book_appointment(
    patient_name: str,
    patient_phone: str,
    doctor_name: str,
    treatment_title: str,
    scheduled_at_iso: str,
    notes: str = "",
) -> str:
    """
    Books a medical appointment and saves it to the database.
    IMPORTANT: Only call this AFTER confirming ALL details with the patient.

    Args:
        patient_name: Full name of the patient
        patient_phone: Phone number in E.164 format (e.g. +8801712345678)
        doctor_name: Name of the doctor (must match available doctors list)
        treatment_title: Reason for appointment (e.g. 'General Checkup')
        scheduled_at_iso: Appointment date/time in ISO 8601 UTC format (e.g. 2025-03-10T10:00:00Z)
        notes: Any extra notes (optional)
    """
    import asyncio

    async def _book():
        from app.db.models import Appointment, Patient

        try:
            scheduled_at = datetime.fromisoformat(scheduled_at_iso.replace("Z", "+00:00"))
        except ValueError:
            return "ERROR: Invalid date format. Use ISO 8601, e.g. 2025-03-10T10:00:00Z"

        # Validate doctor name
        valid_names = [d["name"].lower() for d in DOCTORS]
        if doctor_name.lower() not in valid_names:
            return f"ERROR: Doctor '{doctor_name}' not found. Please check available doctors."

        # Upsert patient
        existing = await _patients_col().find_one({"phone": patient_phone})
        if existing:
            patient_id = existing["id"]
        else:
            patient = Patient(name=patient_name, phone=patient_phone)
            await _patients_col().insert_one(patient.model_dump())
            patient_id = patient.id

        # Create appointment
        appt = Appointment(
            patient_id=patient_id,
            patient_name=patient_name,
            patient_phone=patient_phone,
            doctor_name=doctor_name,
            treatment_title=treatment_title,
            scheduled_at=scheduled_at,
            notes=notes or None,
        )
        await _appointments_col().insert_one(appt.model_dump())

        local_time = scheduled_at.strftime("%A, %B %d %Y at %I:%M %p UTC")
        return (
            f"✅ Appointment booked successfully!\n"
            f"  ID: {appt.id}\n"
            f"  Patient: {patient_name}\n"
            f"  Doctor: {doctor_name}\n"
            f"  Treatment: {treatment_title}\n"
            f"  Date/Time: {local_time}\n"
            f"  Notes: {notes or 'None'}"
        )

    # Run async from sync tool context
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _book())
                return future.result(timeout=10)
        else:
            return loop.run_until_complete(_book())
    except Exception as exc:
        logger.error("book_appointment tool error: %s", exc)
        return f"ERROR: Could not book appointment — {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 3: Get appointment details (for reminder calls)
# ─────────────────────────────────────────────────────────────────────────────
@tool("get_appointment_details")
def get_appointment_details(appointment_id: str) -> str:
    """
    Fetches appointment details from the database by appointment ID.
    Use this during reminder calls to get the patient's appointment info.

    Args:
        appointment_id: The UUID of the appointment to look up.
    """
    import asyncio

    async def _fetch():
        doc = await _appointments_col().find_one({"id": appointment_id}, {"_id": 0})
        if not doc:
            return f"ERROR: Appointment '{appointment_id}' not found."
        dt = doc.get("scheduled_at")
        if isinstance(dt, str):
            dt = datetime.fromisoformat(dt)
        time_str = dt.strftime("%A, %B %d %Y at %I:%M %p UTC") if dt else "Unknown"
        return (
            f"Appointment Details:\n"
            f"  Patient: {doc.get('patient_name')}\n"
            f"  Doctor: {doc.get('doctor_name')}\n"
            f"  Treatment: {doc.get('treatment_title')}\n"
            f"  Scheduled: {time_str}\n"
            f"  Status: {doc.get('status')}\n"
            f"  Notes: {doc.get('notes') or 'None'}"
        )

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _fetch())
                return future.result(timeout=10)
        else:
            return loop.run_until_complete(_fetch())
    except Exception as exc:
        logger.error("get_appointment_details tool error: %s", exc)
        return f"ERROR: Could not fetch appointment — {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 4: Get clinic info
# ─────────────────────────────────────────────────────────────────────────────
@tool("get_clinic_info")
def get_clinic_info(query: str = "") -> str:
    """
    Returns clinic information: name, address, phone, working hours.
    Use when patients ask about the clinic location, hours, or contact.

    Args:
        query: What the patient wants to know (e.g. 'hours', 'address', 'phone').
    """
    return (
        f"Clinic Information:\n"
        f"  Name: {settings.CLINIC_NAME}\n"
        f"  Phone: {settings.CLINIC_PHONE}\n"
        f"  Address: {settings.CLINIC_ADDRESS}\n"
        f"  Working Hours: {settings.CLINIC_HOURS}"
    )
