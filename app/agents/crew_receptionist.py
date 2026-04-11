"""
Medical Receptionist — Conversation Agent
==========================================
Direct OpenAI chat completions with full conversation history + tool calling.
No CrewAI overhead. One API call per turn. Natural human-like conversation.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)

# ── OpenAI tool definitions ───────────────────────────────────────────────────
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_available_doctors",
            "description": "Get the list of doctors and their specialties at the clinic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "specialty": {
                        "type": "string",
                        "description": "Specialty to filter by, or 'all' to list everyone.",
                    }
                },
                "required": ["specialty"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": (
                "Book an appointment in the database. "
                "ONLY call this after the patient has explicitly confirmed all details."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_name":     {"type": "string"},
                    "patient_phone":    {"type": "string"},
                    "doctor_name":      {"type": "string"},
                    "treatment_title":  {"type": "string"},
                    "scheduled_at_iso": {
                        "type": "string",
                        "description": "ISO-8601 UTC, e.g. 2026-04-10T10:00:00Z",
                    },
                    "notes": {"type": "string"},
                },
                "required": [
                    "patient_name", "patient_phone", "doctor_name",
                    "treatment_title", "scheduled_at_iso",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_appointment_details",
            "description": "Look up an existing appointment by patient phone or appointment ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"],
            },
        },
    },
]


# ── Tool executor ─────────────────────────────────────────────────────────────
def _run_tool(name: str, args: Dict) -> str:
    from app.agents.tools import check_available_doctors, book_appointment, get_appointment_details
    try:
        if name == "check_available_doctors":
            return check_available_doctors(args.get("specialty", "all"))
        if name == "book_appointment":
            return book_appointment(
                patient_name    = args["patient_name"],
                patient_phone   = args["patient_phone"],
                doctor_name     = args["doctor_name"],
                treatment_title = args["treatment_title"],
                scheduled_at_iso= args["scheduled_at_iso"],
                notes           = args.get("notes", "Booked via phone call"),
            )
        if name == "get_appointment_details":
            return get_appointment_details(args["query"])
        return f"Unknown tool: {name}"
    except KeyError as exc:
        return f"Missing argument: {exc}"
    except Exception as exc:
        logger.error("Tool '%s' error: %s", name, exc)
        return f"Error: {exc}"


# ── System prompt ─────────────────────────────────────────────────────────────
def _fetch_doctors_for_prompt(tenant: Optional[Any]) -> str:
    """Build a compact doctor list string to embed in the system prompt."""
    try:
        from app.agents.tools import _get_doctors_sync
        from app.services.tenant_service import set_current_tenant
        if tenant:
            set_current_tenant(tenant)
        doctors = _get_doctors_sync()
        if not doctors:
            return ""
        lines = []
        for d in doctors:
            name = d.get("name", "")
            spec = d.get("specialty", "")
            avail = d.get("available_days") or d.get("available_time") or ""
            fee   = d.get("consultation_fee") or ""
            line  = f"  - {name} ({spec})"
            if avail:
                line += f", available {avail}"
            if fee:
                line += f", fee {fee}"
            lines.append(line)
        return "\n".join(lines)
    except Exception:
        return ""


def _system_prompt(ctx: Any, tenant: Optional[Any]) -> str:
    if tenant and hasattr(tenant, "name"):
        clinic   = getattr(tenant, "name",              settings.CLINIC_NAME)
        hours    = getattr(tenant, "business_hours",    None) or settings.CLINIC_HOURS
        address  = getattr(tenant, "address",           None) or settings.CLINIC_ADDRESS
        phone    = getattr(tenant, "phone",             None) or settings.CLINIC_PHONE
        rec_name = getattr(tenant, "receptionist_name", None) or "Lisa"
    else:
        clinic   = settings.CLINIC_NAME
        hours    = settings.CLINIC_HOURS
        address  = settings.CLINIC_ADDRESS
        phone    = settings.CLINIC_PHONE
        rec_name = "Lisa"

    doctor_list = _fetch_doctors_for_prompt(tenant)
    doctor_section = (
        f"\nCLINIC DOCTORS:\n{doctor_list}\n"
        if doctor_list else ""
    )

    return f"""\
You are {rec_name}, a professional medical receptionist at {clinic}. You are on a live phone call.

CLINIC INFORMATION:
  Name    : {clinic}
  Phone   : {phone}
  Address : {address}
  Hours   : {hours}
  Caller  : {ctx.patient_phone}
{doctor_section}
RULES:
1. You are a real human. Never admit to being an AI or automated system.
2. Speak in short, warm, natural sentences — 1 to 2 sentences per turn.
3. Never use bullet points, lists, or markdown in your spoken replies.
4. Sound like the best receptionist at a busy, professional clinic.
5. Do NOT repeat the same greeting or filler phrase twice.

APPOINTMENT BOOKING — collect one at a time, naturally:
  full name → preferred doctor or specialty → reason for visit
  → preferred date and time → confirm phone (you have {ctx.patient_phone})
Summarise once: "To confirm: [name], [doctor], [reason], [date/time] — shall I book that?"
Call book_appointment ONLY after the patient says YES.
After booking, read back the Appointment ID clearly.

Use check_available_doctors when the patient asks about doctors or specialties.
Use get_appointment_details to look up existing appointments."""


# ── Call context ──────────────────────────────────────────────────────────────
@dataclass
class CallContext:
    call_id:       str
    patient_phone: str
    direction:     str           = "inbound"
    is_reminder_call: bool       = False
    appointment_id: Optional[str]= None
    tenant:         Optional[Any]= None

    transcript_segments: List[Dict]  = field(default_factory=list)
    appointment_booked:  bool        = False
    booked_appointment_id: Optional[str] = None
    call_ended:  bool                = False
    started_at:  Optional[datetime]  = None
    ended_at:    Optional[datetime]  = None
    # Full OpenAI conversation history
    messages:    List[Dict]          = field(default_factory=list)

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


# ── Receptionist session ──────────────────────────────────────────────────────
class MedicalReceptionistCrew:
    """
    One instance per call. Wraps a stateful OpenAI conversation.
    Interface kept identical to the old CrewAI version so call_webhook.py
    needs zero changes.
    """

    def __init__(self, ctx: CallContext) -> None:
        self.ctx = ctx
        self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
        # Seed conversation with system prompt
        ctx.messages = [{"role": "system", "content": _system_prompt(ctx, ctx.tenant)}]

    # ── Opening greeting ──────────────────────────────────────────────────────
    def generate_greeting(self) -> str:
        # Hidden trigger — never stored in history as a real user message
        trigger = [
            *self.ctx.messages,
            {"role": "user", "content": "[The call just connected. Greet the caller now.]"},
        ]
        greeting = self._complete(trigger)
        if not greeting:
            from app.config import settings as s
            name = getattr(self.ctx.tenant, "name", None) or s.CLINIC_NAME
            greeting = f"Hello, thank you for calling {name}. How can I help you today?"
        # Store only the assistant reply, not the internal trigger
        self.ctx.messages.append({"role": "assistant", "content": greeting})
        self.ctx.add_turn("receptionist", greeting)
        return greeting

    # ── Per-utterance response ────────────────────────────────────────────────
    def process_utterance(self, patient_text: str) -> str:
        self.ctx.add_turn("patient", patient_text)
        self.ctx.messages.append({"role": "user", "content": patient_text})
        response = self._complete(self.ctx.messages)
        self.ctx.messages.append({"role": "assistant", "content": response})
        self.ctx.add_turn("receptionist", response)
        return response

    # ── Core chat completion (handles tool calls) ─────────────────────────────
    def _complete(self, messages: List[Dict]) -> str:
        try:
            resp = self._client.chat.completions.create(
                model       = settings.OPENAI_MODEL,
                messages    = messages,
                tools       = _TOOLS,
                tool_choice = "auto",
                temperature = 0.6,
                max_tokens  = 160,
            )
            msg = resp.choices[0].message

            # ── Tool call handling ────────────────────────────────────────────
            if msg.tool_calls:
                # Append the assistant's tool-call request to working history
                tool_msg = msg.model_dump(exclude_unset=True, exclude_none=True)
                working = [*messages, tool_msg]

                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}

                    result = _run_tool(tc.function.name, args)
                    logger.info("Tool %s -> %.80s", tc.function.name, result)

                    # Track successful bookings
                    if tc.function.name == "book_appointment" and "ERROR" not in result:
                        self.ctx.appointment_booked = True
                        m = re.search(r"\bID:\s*([a-f0-9\-]{8,})", result, re.IGNORECASE)
                        if m:
                            self.ctx.booked_appointment_id = m.group(1)

                    working.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "content":      result,
                    })

                # Second call — model reads tool results and speaks to patient
                resp2 = self._client.chat.completions.create(
                    model       = settings.OPENAI_MODEL,
                    messages    = working,
                    temperature = 0.6,
                    max_tokens  = 160,
                )
                # Persist tool interaction into conversation history
                for item in working[len(messages):]:
                    self.ctx.messages.append(item)

                return resp2.choices[0].message.content or ""

            return msg.content or ""

        except Exception as exc:
            logger.error("OpenAI error | call=%s | %s", self.ctx.call_id, exc)
            return "Sorry about that — could you say that again?"


# ── Session registry ──────────────────────────────────────────────────────────
_sessions: Dict[str, MedicalReceptionistCrew] = {}


def create_session(ctx: CallContext) -> MedicalReceptionistCrew:
    sess = MedicalReceptionistCrew(ctx)
    _sessions[ctx.call_id] = sess
    logger.info("Session created | call=%s | model=%s", ctx.call_id, settings.OPENAI_MODEL)
    return sess


def get_session(call_id: str) -> Optional[MedicalReceptionistCrew]:
    return _sessions.get(call_id)


def destroy_session(call_id: str) -> None:
    _sessions.pop(call_id, None)
    logger.info("Session destroyed | call=%s", call_id)
