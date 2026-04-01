"""
CrewAI Medical Receptionist
============================
Multi-agent crew for handling inbound medical calls.

Agents:
  1. ReceptionistAgent  â€” main conversational agent (greets, collects info)
  2. AppointmentAgent   â€” specialist for booking & managing appointments

LLM Support:
  - Cloud  â†’ OpenAI GPT-4o          (LLM_PROVIDER=openai in .env)
  - Local  â†’ LM Studio liquid/lfm2  (LLM_PROVIDER=lmstudio in .env)
    â€¢ Start LM Studio â†’ Server tab â†’ Load "liquid/lfm2-1.2b" â†’ Start Server
    â€¢ Default endpoint: http://localhost:1234/v1

Flow per utterance:
  Patient speech â†’ STT â†’ process_utterance(ctx, text) â†’ CrewAI Crew
  â†’ ReceptionistAgent delegates to AppointmentAgent if booking needed
  â†’ Response text â†’ TTS â†’ Twilio
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from crewai import Agent, Crew, Task, LLM, Process

from app.config import settings
from app.agents.tools import (
    DEFAULT_DOCTORS,
    check_available_doctors,
    book_appointment,
    get_appointment_details,
    get_clinic_info,
)

logger = logging.getLogger(__name__)


def _sanitize_spoken_reply(text: str) -> str:
    """
    Convert tool/meta output into plain spoken call text.
    """
    if not text:
        return ""

    cleaned = text.strip().replace("```", "")

    if "Final Answer:" in cleaned:
        cleaned = cleaned.split("Final Answer:", 1)[1].strip()

    lines: List[str] = []
    for raw in cleaned.splitlines():
        line = raw.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("thought:") or lower.startswith("action:") or lower.startswith("action input:") or lower.startswith("observation:"):
            continue
        lines.append(line)
    cleaned = " ".join(lines).strip()

    cleaned = re.sub(r"\bi am (an? )?ai\b", "I am the clinic receptionist", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bi'?m (an? )?ai\b", "I'm the clinic receptionist", cleaned, flags=re.IGNORECASE)

    return cleaned


def _tenant_company_info(tenant: Optional[Any]) -> tuple[str, str, str, str, str]:
    """Return (name, hours, address, phone, receptionist_name) from tenant or settings."""
    if tenant is not None and hasattr(tenant, "name"):
        return (
            getattr(tenant, "name", settings.CLINIC_NAME),
            getattr(tenant, "business_hours", None) or settings.CLINIC_HOURS,
            getattr(tenant, "address", None) or settings.CLINIC_ADDRESS,
            getattr(tenant, "phone", None) or settings.CLINIC_PHONE,
            getattr(tenant, "receptionist_name", None) or "Lisa",
        )
    return (
        settings.CLINIC_NAME,
        settings.CLINIC_HOURS,
        settings.CLINIC_ADDRESS,
        settings.CLINIC_PHONE,
        "Lisa",
    )


def _tenant_profile_summary(tenant: Optional[Any]) -> str:
    if tenant is None:
        return ""
    parts: List[str] = []
    if getattr(tenant, "name", None):
        parts.append(f"Clinic: {tenant.name}")
    if getattr(tenant, "business_hours", None):
        parts.append(f"Hours: {tenant.business_hours}")
    if getattr(tenant, "address", None):
        parts.append(f"Address: {tenant.address}")
    if getattr(tenant, "phone", None):
        parts.append(f"Phone: {tenant.phone}")
    if getattr(tenant, "twilio_phone_number", None):
        parts.append(f"Twilio: {tenant.twilio_phone_number}")
    staff = getattr(tenant, "staff_list", None) or []
    if staff:
        staff_str = "; ".join(
            f"{s.get('name', '').strip()} ({s.get('specialty', '').strip()})".strip()
            for s in staff
            if s.get("name") or s.get("specialty")
        )
        if staff_str:
            parts.append(f"Doctors: {staff_str}")
    extra = getattr(tenant, "extra_info", None) or {}
    if extra:
        extras = []
        for k in ["working_days", "start_time", "end_time", "consultation_fee", "patient_limit"]:
            v = extra.get(k)
            if v not in (None, ""):
                extras.append(f"{k.replace('_', ' ').title()}: {v}")
        if extras:
            parts.append("Extras: " + ", ".join(extras))
    return " | ".join(parts)


def _fast_faq_response(patient_text: str, tenant: Optional[Any] = None) -> Optional[str]:
    """
    Fast path for common clinic questions so callers get immediate answers.
    This bypasses LLM latency for frequent intents.
    """
    text = (patient_text or "").strip().lower()
    if not text:
        return None
    name, hours, address, phone, _ = _tenant_company_info(tenant)
    if any(k in text for k in ["hour", "open", "close", "time", "kokhon", "koyta", "working"]):
        return f"Our business hours are {hours}. Would you like me to book an appointment for you?"
    if any(k in text for k in ["address", "location", "kothay", "where"]):
        return f"Our address is {address}. Would you like directions by SMS?"
    if any(k in text for k in ["phone", "number", "contact", "jogajog"]):
        return f"You can reach us at {phone}. How else can I help you today?"
    if any(k in text for k in ["doctor", "specialist", "cardiology", "orthopedic", "pediatric", "dermatology", "neurology"]):
        try:
            doctors = check_available_doctors("all")
            compact = doctors.replace(f"Available doctors at {name}:\n", "").replace("\n", "; ")
            return f"Available doctors are: {compact}"
        except Exception:
            return "We have specialists in general medicine, cardiology, orthopedics, pediatrics, dermatology, and neurology. Which specialist do you need?"
    return None


YES_WORDS = {"yes", "y", "yeah", "yep", "sure", "ok", "okay", "hmm yes", "ji", "haan", "hya"}
NO_WORDS = {"no", "n", "nope", "nah", "cancel", "na"}
BOOKING_KEYWORDS = {
    "book", "booking", "appointment", "schedule", "serial", "visit", "see doctor",
    "book korte", "appointment nibo", "doctor dekhate", "serial nite",
}
GOODBYE_WORDS = {
    "bye", "bye bye", "goodbye", "take care", "see you", "thanks bye", "thank you bye",
    "allah hafez", "khoda hafez", "biday",
}
ACK_ONLY_WORDS = {
    "ok", "okay", "hmm", "hmmm", "yeah", "yep", "sure", "right", "alright",
    "thanks", "thank you", "got it", "i see", "understood", "hya", "haan", "ji",
}


@dataclass
class BookingState:
    active: bool = False
    awaiting_confirmation: bool = False
    patient_name: Optional[str] = None
    patient_phone: Optional[str] = None
    doctor_name: Optional[str] = None
    treatment_title: Optional[str] = None
    scheduled_at_iso: Optional[str] = None

    def is_complete(self) -> bool:
        return all(
            [
                self.patient_name,
                self.patient_phone,
                self.doctor_name,
                self.treatment_title,
                self.scheduled_at_iso,
            ]
        )


def _contains_booking_intent(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in BOOKING_KEYWORDS)


def _is_goodbye_intent(text: str) -> bool:
    t = (text or "").lower().strip()
    return any(w in t for w in GOODBYE_WORDS)


def _is_ack_only(text: str) -> bool:
    t = re.sub(r"[^a-z0-9\s]+", " ", (text or "").lower()).strip()
    if not t:
        return True
    if len(t.split()) > 4:
        return False
    return t in ACK_ONLY_WORDS


def _extract_name(text: str) -> Optional[str]:
    if not text:
        return None
    patterns = [
        r"(?:my name is|i am|this is)\s+([a-z][a-z .'-]{1,40})",
        r"(?:amar nam)\s+([a-z][a-z .'-]{1,40})",
    ]
    low = text.lower()
    for p in patterns:
        m = re.search(p, low, flags=re.IGNORECASE)
        if m:
            raw = m.group(1).strip(" .,-")
            return " ".join(w.capitalize() for w in raw.split())
    return None


def _extract_phone(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"(\+?\d[\d\s\-]{7,17}\d)", text)
    if not m:
        return None
    digits = re.sub(r"\D", "", m.group(1))
    if len(digits) < 10:
        return None
    if m.group(1).strip().startswith("+"):
        return "+" + digits
    if digits.startswith("88"):
        return "+" + digits
    return "+" + digits


def _extract_doctor(text: str) -> Optional[str]:
    if not text:
        return None
    low = text.lower()
    for d in DOCTORS:
        if d["name"].lower() in low:
            return d["name"]
    for d in DOCTORS:
        if d["specialty"].lower() in low:
            return d["name"]
    return None


def _extract_reason(text: str) -> Optional[str]:
    if not text:
        return None
    low = text.lower().strip()
    m = re.search(r"(?:for|because|reason is|problem is)\s+(.+)$", low)
    if m:
        reason = m.group(1).strip(" .")
        if len(reason) >= 3:
            return reason[:80]
    return None


def _extract_datetime_iso(text: str, tenant: Optional[Any] = None) -> Optional[str]:
    """Parse date/time from user text in tenant's local timezone; return ISO UTC."""
    if not text:
        return None
    from app.utils.datetime_utils import get_tenant_zoneinfo, now_in_tenant_tz
    low = text.lower()
    local_tz = get_tenant_zoneinfo(tenant)
    now = now_in_tenant_tz(tenant)

    # time
    tm = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", low)
    hour = 10
    minute = 0
    if tm:
        hour = int(tm.group(1))
        minute = int(tm.group(2) or "0")
        ampm = tm.group(3)
        if ampm == "pm" and hour < 12:
            hour += 12
        if ampm == "am" and hour == 12:
            hour = 0

    date_obj = None
    if "tomorrow" in low or "kal" in low:
        date_obj = now.date().fromordinal(now.date().toordinal() + 1)
    elif "today" in low or "aj" in low:
        date_obj = now.date()
    else:
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y"):
            m = re.search(r"\b\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\b", low)
            if not m:
                continue
            try:
                date_obj = datetime.strptime(m.group(0), fmt).date()
                break
            except Exception:
                continue

    if not date_obj:
        return None

    try:
        local_dt = datetime(
            date_obj.year,
            date_obj.month,
            date_obj.day,
            hour,
            minute,
            tzinfo=local_tz,
        )
    except Exception:
        return None
    utc_dt = local_dt.astimezone(timezone.utc)
    return utc_dt.isoformat().replace("+00:00", "Z")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# LLM FACTORY â€” picks OpenAI or LM Studio based on config
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _build_llm(temperature: float = 0.7) -> LLM:
    """
    Returns a CrewAI LLM instance.

    OpenAI (cloud):
      Set LLM_PROVIDER=openai and OPENAI_API_KEY in .env

    LM Studio (local):
      Set LLM_PROVIDER=lmstudio in .env
      Load 'liquid/lfm2-1.2b' in LM Studio â†’ Server tab â†’ Start
      No API key needed â€” LM Studio accepts any string.
    """
    if settings.LLM_PROVIDER == "lmstudio":
        logger.info(
            "Using LM Studio | model=%s | base_url=%s",
            settings.LMSTUDIO_MODEL, settings.LMSTUDIO_BASE_URL
        )
        # CrewAI uses LiteLLM under the hood.
        # "openai/" prefix tells LiteLLM to use OpenAI-compatible endpoint.
        return LLM(
            model=f"openai/{settings.LMSTUDIO_MODEL}",
            base_url=settings.LMSTUDIO_BASE_URL,
            api_key=settings.LMSTUDIO_API_KEY,
            temperature=temperature,
        )
    else:
        logger.info("Using OpenAI cloud | model=%s", settings.OPENAI_MODEL)
        return LLM(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=temperature,
        )


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CALL CONTEXT â€” shared state for one call session
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@dataclass
class CallContext:
    call_id: str
    patient_phone: str
    direction: str = "inbound"
    is_reminder_call: bool = False
    appointment_id: Optional[str] = None
    tenant: Optional[Any] = None         # SaaS: company (Tenant) for this call; None = single-tenant fallback

    # Conversation history (plain text for context injection)
    transcript_segments: List[Dict] = field(default_factory=list)
    patient_name: Optional[str] = None
    appointment_booked: bool = False
    booked_appointment_id: Optional[str] = None
    call_ended: bool = False
    booking: BookingState = field(default_factory=BookingState)
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    def add_turn(self, speaker: str, text: str) -> None:
        self.transcript_segments.append({
            "speaker": speaker,
            "text": text,
            "ts": datetime.now(timezone.utc).isoformat(),
        })

    @property
    def conversation_history(self) -> str:
        """Last 10 turns formatted for agent context."""
        recent = self.transcript_segments[-10:]
        return "\n".join(f"{s['speaker'].upper()}: {s['text']}" for s in recent)

    @property
    def full_transcript(self) -> str:
        return "\n".join(f"{s['speaker'].upper()}: {s['text']}" for s in self.transcript_segments)

    def end_call(self) -> None:
        self.call_ended = True
        self.ended_at = datetime.now(timezone.utc)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# AGENT DEFINITIONS (tenant-aware for SaaS)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _build_receptionist_agent(tenant: Optional[Any] = None) -> Agent:
    """
    Main front-desk agent. Handles greeting, understanding patient needs,
    and coordinating with the appointment specialist.
    """
    name, hours, address, _phone, _rec = _tenant_company_info(tenant)
    profile = _tenant_profile_summary(tenant)
    return Agent(
        role="Medical Receptionist",
        goal=(
            f"Help callers of {name} over the phone. "
            "Greet warmly, understand their need, collect required information, "
            "and guide them to book an appointment or answer their questions."
        ),
        backstory=(
            f"You are a warm, professional receptionist at {name}. "
            f"Business hours: {hours}. Address: {address}. "
            f"{('Clinic profile: ' + profile + '. ') if profile else ''}"
            "You speak naturally like a real human on the phone. "
            "Never mention that you are an AI, model, bot, assistant, or automated system unless the caller directly asks. "
            "Keep responses concise and conversational (1-2 sentences for most turns, max 3 only if needed). "
            "Sound like an experienced front-desk executive: calm, polite, and confident. "
            "Do not use repetitive filler phrases like 'How can I assist you today?' on every turn. "
            "Acknowledge the caller's exact point first, then ask at most one focused follow-up question. "
            "Strictly mirror the caller's language style: "
            "if they speak mostly English, reply only in English; "
            "if they speak mostly Bangla, reply only in Bangla; "
            "if they mix Bangla and English, reply in natural Banglish. "
            "Do not randomly switch languages inside one sentence; only mix when the caller mixes. "
            "When a caller wants to book an appointment, collect: their full name, phone number, "
            "preferred doctor, reason/treatment, and preferred date & time â€” then confirm before booking."
        ),
        tools=[check_available_doctors, get_clinic_info],
        llm=_build_llm(temperature=0.35),
        verbose=False,
        allow_delegation=False,
        max_iter=1,
    )


def _build_greeting_agent(tenant: Optional[Any] = None) -> Agent:
    name, _hours, _addr, _ph, rec_name = _tenant_company_info(tenant)
    return Agent(
        role="Front Desk Receptionist",
        goal=(
            f"Start calls naturally for {name}, sounding like a real human "
            "receptionist, not a bot, and using the caller's preferred language."
        ),
        backstory=(
            f"You are {rec_name}, the front-desk receptionist at {name}. "
            "You speak naturally like a real human on the phone. "
            "You can speak Bangla, English, or mixed Banglish. "
            "For the opening greeting, keep it very short and friendly (1â€“2 sentences). "
            "Always greet in a single clear language based on the caller's style. "
            "Avoid sounding scripted and use natural professional phone phrasing. "
            "For this greeting output, return only the exact words you would say out loud, "
            "with no labels, no reasoning, and no explanations."
        ),
        tools=[],
        llm=_build_llm(temperature=0.4),
        verbose=False,
        allow_delegation=False,
        max_iter=1,
    )


def _build_appointment_agent(tenant: Optional[Any] = None) -> Agent:
    """Specialist agent for appointment booking and retrieval."""
    return Agent(
        role="Appointment Booking Specialist",
        goal="Accurately book, retrieve, and manage patient appointments in the database.",
        backstory=(
            "You are an expert at managing medical appointments. "
            "You have direct access to the appointment database. "
            "When all details are confirmed, you book the appointment and return a confirmation. "
            "Always verify the doctor name matches available doctors before booking."
        ),
        tools=[book_appointment, get_appointment_details, check_available_doctors],
        llm=_build_llm(temperature=0.1),
        verbose=False,
        allow_delegation=False,
        max_iter=3,
    )


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CREW â€” orchestrates agents for one call session
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class MedicalReceptionistCrew:
    """
    One instance per active call. Processes utterances turn-by-turn.
    Uses CrewAI hierarchical process so ReceptionistAgent can delegate
    appointment booking to AppointmentAgent automatically.
    SaaS: ctx.tenant sets company-specific agents and tools.
    """

    def __init__(self, ctx: CallContext):
        self.ctx = ctx
        if ctx.patient_phone and not ctx.booking.patient_phone:
            ctx.booking.patient_phone = ctx.patient_phone
        self.receptionist = _build_receptionist_agent(ctx.tenant)
        self.greeter = _build_greeting_agent(ctx.tenant)
        self.appointment_specialist = _build_appointment_agent(ctx.tenant)

    # â”€â”€ Generate opening greeting â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def generate_greeting(self) -> str:
        from app.services.tenant_service import set_current_tenant
        ctx = self.ctx
        set_current_tenant(ctx.tenant)
        name, _h, _a, _p, _r = _tenant_company_info(ctx.tenant)
        description = (
            f"A caller just called {name}. "
            "Generate a warm, professional inbound greeting and ask how you can help."
        )
        agent = self.greeter

        greeting_task = Task(
            description=(
                f"{description}\n\n"
                "Return only the exact words to speak to the patient now. "
                "No headings, no reasoning, no tool traces."
            ),
            expected_output="Plain spoken greeting only (2-3 sentences max).",
            agent=agent,
        )
        crew = Crew(
            agents=[agent],
            tasks=[greeting_task],
            process=Process.sequential,
            verbose=False,
        )
        result = crew.kickoff()
        greeting = _sanitize_spoken_reply(str(result).strip())
        if not greeting:
            greeting = f"Hello, thank you for calling {name}. How may I help you today?"
        self.ctx.add_turn("receptionist", greeting)
        return greeting

    # â”€â”€ Process one patient utterance â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def process_utterance(self, patient_text: str) -> str:
        """
        Main turn handler. Called for every patient utterance.
        Runs a CrewAI hierarchical crew so the receptionist can
        automatically delegate booking to the appointment specialist.
        """
        from app.services.tenant_service import set_current_tenant
        set_current_tenant(self.ctx.tenant)
        self.ctx.add_turn("patient", patient_text)
        t = (patient_text or "").strip()
        low = t.lower()
        booking = self.ctx.booking

        if _is_goodbye_intent(low):
            response = "Thank you for calling HealthCare Medical Center. Have a great day. Goodbye."
            self.ctx.add_turn("receptionist", response)
            return response

        if _is_ack_only(low) and not booking.active:
            # Natural silence on backchannel-only utterances avoids robotic over-talking.
            return ""

        if any(k in low for k in ["cancel booking", "stop booking", "booking cancel"]):
            self.ctx.booking = BookingState()
            response = "Okay, booking canceled. How else can I help you today?"
            self.ctx.add_turn("receptionist", response)
            return response

        if _contains_booking_intent(low) or booking.active:
            booking.active = True

            name = _extract_name(t)
            phone = _extract_phone(t)
            doctor = _extract_doctor(t)
            reason = _extract_reason(t)
            when_iso = _extract_datetime_iso(t, self.ctx.tenant)

            if name:
                booking.patient_name = name
            if phone:
                booking.patient_phone = phone
            if doctor:
                booking.doctor_name = doctor
            if reason:
                booking.treatment_title = reason
            if when_iso:
                booking.scheduled_at_iso = when_iso

            if booking.awaiting_confirmation:
                if low in YES_WORDS or any(w in low for w in YES_WORDS):
                    result = book_appointment(
                        patient_name=booking.patient_name or "Unknown",
                        patient_phone=booking.patient_phone or self.ctx.patient_phone,
                        doctor_name=booking.doctor_name or "Dr. Rahman",
                        treatment_title=booking.treatment_title or "General Checkup",
                        scheduled_at_iso=booking.scheduled_at_iso or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                        notes="Booked via phone call",
                    )
                    clean = _sanitize_spoken_reply(result)
                    aid = None
                    m = re.search(r"\bID:\s*([a-f0-9\-]{8,})", result, flags=re.IGNORECASE)
                    if m:
                        aid = m.group(1)
                    self.ctx.appointment_booked = "ERROR" not in result
                    self.ctx.booked_appointment_id = aid
                    self.ctx.booking = BookingState()
                    response = clean if clean else "Great, your appointment is booked successfully."
                    self.ctx.add_turn("receptionist", response)
                    return response
                if low in NO_WORDS or any(w in low for w in NO_WORDS):
                    booking.awaiting_confirmation = False
                    response = "No problem. Which detail should I update: name, doctor, reason, phone, or date and time?"
                    self.ctx.add_turn("receptionist", response)
                    return response

            missing = []
            if not booking.patient_name:
                missing.append("name")
            if not booking.doctor_name:
                missing.append("doctor")
            if not booking.treatment_title:
                missing.append("reason")
            if not booking.scheduled_at_iso:
                missing.append("date_time")

            if missing:
                nxt = missing[0]
                if nxt == "name":
                    response = "Sure, I can help with booking. May I have your full name?"
                elif nxt == "doctor":
                    response = "Which doctor or specialty would you prefer?"
                elif nxt == "reason":
                    response = "What is the main reason for the visit?"
                else:
                    response = "Please tell me your preferred date and time, for example 2026-03-10 5:30 PM."
                self.ctx.add_turn("receptionist", response)
                return response

            booking.awaiting_confirmation = True
            try:
                from app.utils.datetime_utils import format_date_short_tenant
                dt_utc = datetime.fromisoformat((booking.scheduled_at_iso or "").replace("Z", "+00:00"))
                if dt_utc.tzinfo is None:
                    dt_utc = dt_utc.replace(tzinfo=timezone.utc)
                when_text = format_date_short_tenant(dt_utc, self.ctx.tenant)
            except Exception:
                when_text = booking.scheduled_at_iso or ""
            response = (
                f"Please confirm: Name {booking.patient_name}, Doctor {booking.doctor_name}, "
                f"Reason {booking.treatment_title}, Time {when_text}. Should I book it now?"
            )
            self.ctx.add_turn("receptionist", response)
            return response

        faq = _fast_faq_response(patient_text, self.ctx.tenant)
        if faq:
            self.ctx.add_turn("receptionist", faq)
            return faq

        task_description = (
            f"The patient just said: \"{patient_text}\"\n\n"
            f"Conversation so far:\n{self.ctx.conversation_history}\n\n"
            "Respond naturally as a real medical receptionist. "
            "First, directly answer the patient's exact question. Do not ignore the question. "
            "Return only spoken reply text, no labels and no reasoning. "
            "Do not mention being an AI/model unless the patient directly asks. "
            "Speak like a human receptionist in Bangladesh; Bangla, English, or mixed Banglish based on patient style. "
            "Keep your spoken response short and practical. "
            "Do not repeat the same stock sentence in consecutive turns. "
            "Use one clear response and, only if needed, one focused follow-up question."
        )

        response_task = Task(
            description=task_description,
            expected_output="A natural spoken response to the patient (1-3 sentences, phone-appropriate).",
            agent=self.receptionist,
        )
        crew = Crew(
            agents=[self.receptionist],
            tasks=[response_task],
            process=Process.sequential,
            verbose=False,
        )
        try:
            result = crew.kickoff()
            response = _sanitize_spoken_reply(str(result).strip())
        except Exception as exc:
            logger.error("CrewAI error on call %s: %s", self.ctx.call_id, exc)
            response = "I'm sorry, could you please repeat that?"
        if not response:
            response = "I'm sorry, I didn't catch that clearly. Could you please repeat?"
        self.ctx.add_turn("receptionist", response)
        return response


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SESSION REGISTRY
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_sessions: Dict[str, MedicalReceptionistCrew] = {}


def create_session(ctx: CallContext) -> MedicalReceptionistCrew:
    crew = MedicalReceptionistCrew(ctx)
    _sessions[ctx.call_id] = crew
    logger.info("Session created call_id=%s direction=%s provider=%s",
                ctx.call_id, ctx.direction, settings.LLM_PROVIDER)
    return crew


def get_session(call_id: str) -> Optional[MedicalReceptionistCrew]:
    return _sessions.get(call_id)


def destroy_session(call_id: str) -> None:
    _sessions.pop(call_id, None)
    logger.info("Session destroyed call_id=%s", call_id)

