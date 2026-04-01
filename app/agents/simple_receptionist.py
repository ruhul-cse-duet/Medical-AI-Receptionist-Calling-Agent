"""
Simple receptionist backend.
Fallback when CrewAI is unavailable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.config import settings

logger = logging.getLogger(__name__)


def _company_name(tenant: Optional[Any]) -> str:
    if tenant is not None and hasattr(tenant, "name"):
        return getattr(tenant, "name", settings.CLINIC_NAME)
    return settings.CLINIC_NAME


@dataclass
class CallContext:
    call_id: str
    patient_phone: str
    direction: str = "inbound"
    is_reminder_call: bool = False
    appointment_id: Optional[str] = None
    tenant: Optional[Any] = None  # SaaS: company for this call

    transcript_segments: List[Dict] = field(default_factory=list)
    patient_name: Optional[str] = None
    appointment_booked: bool = False
    booked_appointment_id: Optional[str] = None
    call_ended: bool = False
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    def add_turn(self, speaker: str, text: str) -> None:
        self.transcript_segments.append(
            {
                "speaker": speaker,
                "text": text,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        )

    @property
    def conversation_history(self) -> str:
        recent = self.transcript_segments[-10:]
        return "\n".join(f"{s['speaker'].upper()}: {s['text']}" for s in recent)

    @property
    def full_transcript(self) -> str:
        return "\n".join(
            f"{s['speaker'].upper()}: {s['text']}" for s in self.transcript_segments
        )

    def end_call(self) -> None:
        self.call_ended = True
        self.ended_at = datetime.now(timezone.utc)


class SimpleReceptionist:
    """
    Minimal fallback implementation that keeps the service running
    when CrewAI fails to import.
    """

    def __init__(self, ctx: CallContext):
        self.ctx = ctx

    def generate_greeting(self) -> str:
        name = _company_name(self.ctx.tenant)
        text = (
            f"Hello! Thank you for calling {name}. "
            "How can I help you today?"
        )
        self.ctx.add_turn("receptionist", text)
        return text

    def process_utterance(self, patient_text: str) -> str:
        self.ctx.add_turn("patient", patient_text)
        response = (
            "Thanks for sharing. Could you tell me your full name, "
            "preferred doctor, and a date/time that works for you?"
        )
        self.ctx.add_turn("receptionist", response)
        return response


_sessions: Dict[str, SimpleReceptionist] = {}


def create_session(ctx: CallContext) -> SimpleReceptionist:
    sess = SimpleReceptionist(ctx)
    _sessions[ctx.call_id] = sess
    logger.info(
        "Simple session created call_id=%s direction=%s",
        ctx.call_id,
        ctx.direction,
    )
    return sess


def get_session(call_id: str) -> Optional[SimpleReceptionist]:
    return _sessions.get(call_id)


def destroy_session(call_id: str) -> None:
    _sessions.pop(call_id, None)
    logger.info("Simple session destroyed call_id=%s", call_id)
