"""
Celery Worker Tasks — Medical AI Receptionist
=============================================
Tasks:
  1. process_appointment_reminders  — runs every 30 min via Celery Beat
     Finds appointments whose scheduled_at is within (5h, 5.5h) from now
     and triggers reminder calls for those not yet reminded.

  2. send_reminder_call  — initiates a Twilio outbound call with appointment details

  3. dial_patient_outbound  — generic outbound dial task
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone

from celery import Celery
from celery.schedules import crontab

from app.config import settings

celery_app = Celery(
    "medical_receptionist",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_soft_time_limit=120,
    task_time_limit=300,
    beat_schedule={
        # Check every 30 minutes for upcoming appointments needing a reminder
        "process-appointment-reminders": {
            "task": "worker.tasks.process_appointment_reminders",
            "schedule": 1800.0,   # every 30 minutes
        },
    },
)

logger = logging.getLogger(__name__)


def _run(coro):
    """Run async coroutine from sync Celery task."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── 1. Reminder scan — runs every 30 minutes ──────────────────────────────────
@celery_app.task(name="worker.tasks.process_appointment_reminders")
def process_appointment_reminders():
    """
    Scans MongoDB for appointments that:
      - Are SCHEDULED or CONFIRMED
      - Have not been reminded yet
      - Are between (now + 5h) and (now + 5.5h) — 30-min window matches beat schedule
    Dispatches send_reminder_call task for each.
    """
    async def _scan():
        from app.db.base import connect_db, appointments_col

        await connect_db()
        now = datetime.now(timezone.utc)
        window_start = now + timedelta(hours=settings.REMINDER_HOURS_BEFORE)
        window_end   = now + timedelta(hours=settings.REMINDER_HOURS_BEFORE, minutes=30)

        cursor = appointments_col().find({
            "status":        {"$in": ["scheduled", "confirmed"]},
            "reminder_sent": False,
            "scheduled_at":  {"$gte": window_start, "$lt": window_end},
        })
        appointments = await cursor.to_list(length=200)
        logger.info(
            "Reminder scan: window [%s, %s] → %d appointments found",
            window_start.isoformat(), window_end.isoformat(), len(appointments),
        )
        for appt in appointments:
            send_reminder_call.delay(appt["id"])

    _run(_scan())


# ── 2. Send reminder call ──────────────────────────────────────────────────────
@celery_app.task(name="worker.tasks.send_reminder_call", bind=True, max_retries=3)
def send_reminder_call(self, appointment_id: str):
    """
    Places an outbound Twilio call to the patient with appointment reminder details.
    Marks reminder_sent=True on success.
    """
    async def _call():
        from app.db.base import connect_db, appointments_col, calls_col
        from app.db.models import Call, CallDirection, CallStatus, utcnow
        from app.services.twilio_service import TwilioService

        await connect_db()
        appt = await appointments_col().find_one({"id": appointment_id})
        if not appt:
            logger.warning("Reminder: appointment %s not found", appointment_id)
            return
        if appt.get("reminder_sent"):
            logger.info("Reminder already sent for %s", appointment_id)
            return

        call_id = str(uuid.uuid4())

        # Persist call record
        call_doc = Call(
            id=call_id,
            patient_phone=appt["patient_phone"],
            direction=CallDirection.OUTBOUND,
            status=CallStatus.INITIATED,
            is_reminder_call=True,
            appointment_id=appointment_id,
        )
        await calls_col().insert_one(call_doc.model_dump())

        # Initiate Twilio call
        twilio = TwilioService()
        try:
            sid = await twilio.initiate_outbound_call(
                to_phone=appt["patient_phone"],
                call_id=call_id,
                is_reminder=True,
            )
            await calls_col().update_one(
                {"id": call_id}, {"$set": {"twilio_call_sid": sid}}
            )
            # Mark reminder sent
            await appointments_col().update_one(
                {"id": appointment_id},
                {"$set": {
                    "reminder_sent": True,
                    "reminder_sent_at": utcnow(),
                    "updated_at": utcnow(),
                }},
            )
            logger.info(
                "Reminder call placed → patient=%s appt=%s sid=%s",
                appt["patient_phone"], appointment_id, sid,
            )
        except Exception as exc:
            logger.error("Reminder call failed appt=%s: %s", appointment_id, exc)
            await calls_col().update_one(
                {"id": call_id}, {"$set": {"status": CallStatus.FAILED.value}}
            )
            raise exc

    try:
        _run(_call())
    except Exception as exc:
        raise self.retry(exc=exc, countdown=300)   # retry in 5 minutes


# ── 3. Generic outbound dial ───────────────────────────────────────────────────
@celery_app.task(name="worker.tasks.dial_patient_outbound", bind=True, max_retries=2)
def dial_patient_outbound(self, patient_phone: str, patient_name: str = ""):
    """Initiate an AI receptionist outbound call to a patient."""
    async def _dial():
        from app.db.base import connect_db, calls_col
        from app.db.models import Call, CallDirection, CallStatus, utcnow
        from app.services.twilio_service import TwilioService

        await connect_db()
        call_id = str(uuid.uuid4())
        call_doc = Call(
            id=call_id,
            patient_phone=patient_phone,
            direction=CallDirection.OUTBOUND,
            status=CallStatus.INITIATED,
            started_at=utcnow(),
        )
        await calls_col().insert_one(call_doc.model_dump())

        twilio = TwilioService()
        try:
            sid = await twilio.initiate_outbound_call(
                to_phone=patient_phone, call_id=call_id
            )
            await calls_col().update_one(
                {"id": call_id}, {"$set": {"twilio_call_sid": sid}}
            )
        except Exception as exc:
            await calls_col().update_one(
                {"id": call_id}, {"$set": {"status": CallStatus.FAILED.value}}
            )
            raise exc

    try:
        _run(_dial())
    except Exception as exc:
        raise self.retry(exc=exc, countdown=60)
