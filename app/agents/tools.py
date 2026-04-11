"""
Agent Tools — MongoDB operations for the AI receptionist.
Doctors are now fetched from the 'doctors' collection (DB-backed).
Falls back to tenant.staff_list, then DEFAULT_DOCTORS if DB is empty.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from crewai.tools import tool
from app.config import settings

logger = logging.getLogger(__name__)

DEFAULT_DOCTORS: List[Dict[str, Any]] = [
    {"name": "Dr. Rahman",    "specialty": "General Medicine"},
    {"name": "Dr. Sultana",   "specialty": "Cardiology"},
    {"name": "Dr. Hossain",   "specialty": "Orthopedics"},
    {"name": "Dr. Ahmed",     "specialty": "Pediatrics"},
    {"name": "Dr. Chowdhury", "specialty": "Dermatology"},
    {"name": "Dr. Islam",     "specialty": "Neurology"},
]


def _current_tenant():
    from app.services.tenant_service import get_current_tenant
    return get_current_tenant()


def _tenant_id() -> str:
    t = _current_tenant()
    return t.id if t else ""


def _appointments_col():
    from app.db.base import appointments_col
    return appointments_col()


def _patients_col():
    from app.db.base import patients_col
    return patients_col()


def _doctors_col():
    from app.db.base import doctors_col
    return doctors_col()


async def _get_doctors_from_db(tenant_id: str) -> List[Dict[str, Any]]:
    """Fetch active doctors from 'doctors' collection for this tenant."""
    if not tenant_id:
        return []
    cursor = _doctors_col().find({"tenant_id": tenant_id, "is_active": True}, {"_id": 0})
    return await cursor.to_list(length=200)


def _run_async(coro):
    """Run an async coroutine from a sync context."""
    import asyncio
    import concurrent.futures
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=10)
        else:
            return loop.run_until_complete(coro)
    except Exception as exc:
        logger.error("_run_async error: %s", exc)
        return None


def _get_doctors_sync() -> List[Dict[str, Any]]:
    """Return doctors list: DB → tenant.staff_list → DEFAULT_DOCTORS fallback."""
    tid = _tenant_id()
    if tid:
        result = _run_async(_get_doctors_from_db(tid))
        if result:
            return result
    # Fallback: tenant staff_list
    t = _current_tenant()
    if t and t.staff_list:
        return t.staff_list
    return DEFAULT_DOCTORS



# ─────────────────────────────────────────────────────────────────────────────
# TOOL 1: check_available_doctors  (DB-backed)
# ─────────────────────────────────────────────────────────────────────────────
@tool("check_available_doctors")
def check_available_doctors(query: str = "") -> str:
    """
    Returns the list of doctors available at the clinic with specialties,
    qualifications, availability, and fees. Fetches from database.
    Input: specialty keyword (e.g. 'cardiology') or 'all'.
    """
    t = _current_tenant()
    clinic_name = t.name if t else settings.CLINIC_NAME
    staff = _get_doctors_sync()

    keyword = (query or "").strip().lower()
    if keyword and keyword not in ("all", "list", ""):
        filtered = [d for d in staff if keyword in (d.get("specialty") or "").lower()
                    or keyword in (d.get("name") or "").lower()]
        if filtered:
            staff = filtered

    lines = []
    for d in staff:
        name  = d.get("name", "")
        spec  = d.get("specialty", "")
        qual  = d.get("qualification", "")
        avail = d.get("available_days", "") or d.get("available_time", "")
        fee   = d.get("consultation_fee", "")
        line  = f"• {name} — {spec}"
        if qual:
            line += f" ({qual})"
        if avail:
            line += f" | Available: {avail}"
        if fee:
            line += f" | Fee: {fee}"
        lines.append(line)

    header = f"Available doctors at {clinic_name}:\n"
    return header + "\n".join(lines) if lines else "No doctors found for the given query."


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 2: book_appointment  (stores doctor_id)
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
    Books a medical appointment and saves it to the database with doctor_id.
    IMPORTANT: Only call this AFTER confirming ALL details with the patient.
    """
    async def _book():
        from app.db.models import Appointment, Patient

        try:
            scheduled_at = datetime.fromisoformat(scheduled_at_iso.replace("Z", "+00:00"))
        except ValueError:
            return "ERROR: Invalid date format. Use ISO 8601, e.g. 2025-03-10T10:00:00Z"

        staff = _get_doctors_sync()
        matched_doctor = None
        for d in staff:
            if doctor_name.lower() in (d.get("name") or "").lower():
                matched_doctor = d
                break

        if not matched_doctor:
            names = [d.get("name", "") for d in staff]
            return f"ERROR: Doctor '{doctor_name}' not found. Available: {', '.join(names)}"

        resolved_name = matched_doctor.get("name", doctor_name)
        doctor_id     = matched_doctor.get("doctor_id") or matched_doctor.get("id")

        tid = _tenant_id()
        q = {"phone": patient_phone}
        if tid:
            q["tenant_id"] = tid
        existing = await _patients_col().find_one(q)
        if existing:
            patient_id = existing["id"]
            # Update patient name if missing
            if not existing.get("name") and patient_name:
                await _patients_col().update_one({"id": patient_id}, {"$set": {"name": patient_name}})
        else:
            patient = Patient(tenant_id=tid, name=patient_name, phone=patient_phone)
            await _patients_col().insert_one(patient.model_dump())
            patient_id = patient.id

        appt = Appointment(
            tenant_id=tid,
            patient_id=patient_id,
            patient_name=patient_name,
            patient_phone=patient_phone,
            doctor_id=doctor_id,
            doctor_name=resolved_name,
            treatment_title=treatment_title,
            scheduled_at=scheduled_at,
            notes=notes or None,
        )
        await _appointments_col().insert_one(appt.model_dump())

        from app.utils.datetime_utils import format_datetime_tenant
        from app.services.tenant_service import get_current_tenant
        local_time = format_datetime_tenant(scheduled_at, get_current_tenant())
        return (
            f"✅ Appointment booked successfully!\n"
            f"  ID: {appt.id}\n"
            f"  Patient: {patient_name}\n"
            f"  Doctor: {resolved_name}\n"
            f"  Doctor ID: {doctor_id or 'N/A'}\n"
            f"  Treatment: {treatment_title}\n"
            f"  Date/Time: {local_time}\n"
            f"  Notes: {notes or 'None'}"
        )

    return _run_async(_book()) or "ERROR: Could not book appointment."



# ─────────────────────────────────────────────────────────────────────────────
# TOOL 3: get_appointment_details
# ─────────────────────────────────────────────────────────────────────────────
@tool("get_appointment_details")
def get_appointment_details(query: str) -> str:
    """
    Fetches appointment details from the database by appointment ID or patient phone.
    """
    async def _fetch():
        tid = _tenant_id()
        # Try by appointment ID first, then by patient phone
        filters = [{"id": query}, {"patient_phone": query}]
        if tid:
            filters = [{"id": query, "tenant_id": tid}, {"patient_phone": query, "tenant_id": tid}]
        doc = None
        for f in filters:
            doc = await _appointments_col().find_one(f, {"_id": 0})
            if doc:
                break
        if not doc:
            return f"No appointment found for '{query}'."

        from app.utils.datetime_utils import format_datetime_tenant
        from app.services.tenant_service import get_current_tenant
        dt = doc.get("scheduled_at")
        if isinstance(dt, str):
            dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        time_str = format_datetime_tenant(dt, get_current_tenant()) if dt else "Unknown"
        return (
            f"Appointment Details:\n"
            f"  ID: {doc.get('id')}\n"
            f"  Patient: {doc.get('patient_name')}\n"
            f"  Doctor: {doc.get('doctor_name')} (ID: {doc.get('doctor_id') or 'N/A'})\n"
            f"  Treatment: {doc.get('treatment_title')}\n"
            f"  Scheduled: {time_str}\n"
            f"  Status: {doc.get('status')}\n"
            f"  Notes: {doc.get('notes') or 'None'}"
        )

    return _run_async(_fetch()) or "ERROR: Could not fetch appointment."


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 4: get_clinic_info
# ─────────────────────────────────────────────────────────────────────────────
@tool("get_clinic_info")
def get_clinic_info(query: str = "") -> str:
    """
    Returns clinic/business information: name, address, phone, hours, timezone.
    """
    from app.services.tenant_service import get_current_tenant
    t = get_current_tenant()
    name    = t.name            if t else settings.CLINIC_NAME
    phone   = t.phone           if t else settings.CLINIC_PHONE
    address = t.address         if t else settings.CLINIC_ADDRESS
    hours   = t.business_hours  if t else settings.CLINIC_HOURS
    tz      = t.timezone        if t else "Asia/Dhaka"
    return (
        f"Clinic Information:\n"
        f"  Name: {name}\n"
        f"  Phone: {phone}\n"
        f"  Address: {address}\n"
        f"  Hours: {hours}\n"
        f"  Timezone: {tz}"
    )
