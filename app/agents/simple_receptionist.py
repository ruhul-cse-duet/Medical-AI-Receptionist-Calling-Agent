"""
Simple Receptionist — Fallback backend (no CrewAI dependency).
Uses direct OpenAI chat completions with conversation history.
Identical interface to crew_receptionist so call_webhook.py works unchanged.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.config import settings

logger = logging.getLogger(__name__)


# ── Call context ──────────────────────────────────────────────────────────────
@dataclass
class CallContext:
    call_id:       str
    patient_phone: str
    direction:     str            = "inbound"
    is_reminder_call: bool        = False
    appointment_id: Optional[str] = None
    tenant:         Optional[Any] = None

    transcript_segments:   List[Dict]    = field(default_factory=list)
    appointment_booked:    bool          = False
    booked_appointment_id: Optional[str] = None
    call_ended:            bool          = False
    started_at:            Optional[datetime] = None
    ended_at:              Optional[datetime] = None
    messages:              List[Dict]    = field(default_factory=list)

    def add_turn(self, speaker: str, text: str) -> None:
        self.transcript_segments.append({
            "speaker": speaker,
            "text": text,
            "ts": datetime.now(timezone.utc).isoformat(),
        })

    @property
    def full_transcript(self) -> str:
        return "\n".join(
            f"{s['speaker'].upper()}: {s['text']}" for s in self.transcript_segments
        )

    def end_call(self) -> None:
        self.call_ended = True
        self.ended_at = datetime.now(timezone.utc)


# ── Simple receptionist ───────────────────────────────────────────────────────
class SimpleReceptionist:
    def __init__(self, ctx: CallContext) -> None:
        self.ctx = ctx
        clinic = getattr(ctx.tenant, "name", None) or settings.CLINIC_NAME
        self.ctx.messages = [{
            "role": "system",
            "content": (
                f"You are Lisa, a receptionist at {clinic}. "
                "You are on a live phone call. "
                "Speak only in English. Keep replies to 1-2 sentences. "
                "Sound like a real human — warm, natural, helpful. "
                "Never say you are an AI or bot."
            ),
        }]

    def generate_greeting(self) -> str:
        clinic = getattr(self.ctx.tenant, "name", None) or settings.CLINIC_NAME
        text = f"Hello, thank you for calling {clinic}. How can I help you today?"
        self.ctx.messages.append({"role": "assistant", "content": text})
        self.ctx.add_turn("receptionist", text)
        return text

    def process_utterance(self, patient_text: str) -> str:
        self.ctx.add_turn("patient", patient_text)
        self.ctx.messages.append({"role": "user", "content": patient_text})
        response = self._complete()
        self.ctx.messages.append({"role": "assistant", "content": response})
        self.ctx.add_turn("receptionist", response)
        return response

    def _complete(self) -> str:
        if not settings.OPENAI_API_KEY:
            return "I'm sorry, I'm having a technical issue. Please call back shortly."
        try:
            from openai import OpenAI
            client = OpenAI(api_key=settings.OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=self.ctx.messages,
                temperature=0.7,
                max_tokens=120,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            logger.error("SimpleReceptionist OpenAI error | call=%s | %s", self.ctx.call_id, exc)
            return "Sorry, could you repeat that please?"


# ── Session registry ──────────────────────────────────────────────────────────
_sessions: Dict[str, SimpleReceptionist] = {}


def create_session(ctx: CallContext) -> SimpleReceptionist:
    sess = SimpleReceptionist(ctx)
    _sessions[ctx.call_id] = sess
    logger.info("Simple session created | call=%s", ctx.call_id)
    return sess


def get_session(call_id: str) -> Optional[SimpleReceptionist]:
    return _sessions.get(call_id)


def destroy_session(call_id: str) -> None:
    _sessions.pop(call_id, None)
    logger.info("Simple session destroyed | call=%s", call_id)
