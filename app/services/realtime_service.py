"""
OpenAI Realtime API Bridge
===========================
Bridges Twilio Media Stream WebSocket <-> OpenAI gpt-4o-realtime-preview.

Audio path (zero conversion — both sides use g711_ulaw @ 8 kHz):
    Twilio media payload  ->  input_audio_buffer.append  ->  OpenAI
    OpenAI response.audio.delta  ->  Twilio media event

Features:
    - Server-side VAD  : OpenAI detects speech start/end automatically
    - Barge-in support : ongoing TTS is cancelled when patient speaks
    - Function calling : appointment tools exposed directly to the model
    - Full transcript  : captured into CallContext for DB persistence
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional

import websockets
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from app.config import settings

logger = logging.getLogger(__name__)

REALTIME_URL = "wss://api.openai.com/v1/realtime"

# ── Tool definitions sent to OpenAI as session functions ──────────────────────
_REALTIME_TOOLS: List[Dict] = [
    {
        "type": "function",
        "name": "check_available_doctors",
        "description": (
            "Returns the list of doctors and their specialties available at the clinic. "
            "Call this when the patient asks about doctors, specialists, or availability."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "specialty": {
                    "type": "string",
                    "description": "Filter by specialty (e.g. 'cardiology'), or 'all' to list everyone.",
                }
            },
            "required": ["specialty"],
        },
    },
    {
        "type": "function",
        "name": "book_appointment",
        "description": (
            "Book a new appointment in the database. "
            "IMPORTANT: Call this ONLY after the patient has verbally confirmed all details."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "patient_name":     {"type": "string", "description": "Full name of the patient"},
                "patient_phone":    {"type": "string", "description": "Patient contact phone number"},
                "doctor_name":      {"type": "string", "description": "Full name of the doctor"},
                "treatment_title":  {"type": "string", "description": "Reason / treatment type"},
                "scheduled_at_iso": {
                    "type": "string",
                    "description": "Appointment datetime in ISO-8601 UTC format, e.g. 2026-04-10T10:00:00Z",
                },
                "notes": {"type": "string", "description": "Optional additional notes"},
            },
            "required": [
                "patient_name", "patient_phone", "doctor_name",
                "treatment_title", "scheduled_at_iso",
            ],
        },
    },
    {
        "type": "function",
        "name": "get_appointment_details",
        "description": "Look up an existing appointment by patient phone number or appointment ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Patient phone number or appointment ID to look up",
                }
            },
            "required": ["query"],
        },
    },
]


def _execute_tool(name: str, args: Dict) -> str:
    """Run a tool synchronously (called via asyncio.to_thread) and return a string result."""
    from app.agents.tools import (
        check_available_doctors,
        book_appointment,
        get_appointment_details,
    )
    try:
        if name == "check_available_doctors":
            return check_available_doctors(args.get("specialty", "all"))
        if name == "book_appointment":
            return book_appointment(
                patient_name=args["patient_name"],
                patient_phone=args["patient_phone"],
                doctor_name=args["doctor_name"],
                treatment_title=args["treatment_title"],
                scheduled_at_iso=args["scheduled_at_iso"],
                notes=args.get("notes", "Booked via phone (Realtime API)"),
            )
        if name == "get_appointment_details":
            return get_appointment_details(args["query"])
        return f"Unknown tool: {name}"
    except KeyError as exc:
        return f"Missing required argument: {exc}"
    except Exception as exc:
        logger.error("Tool '%s' raised an error: %s", name, exc)
        return f"Tool error: {exc}"


def _fetch_doctors_str(tenant: Optional[Any]) -> str:
    """Fetch DB doctors as a compact string for the system prompt."""
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
            name  = d.get("name", "")
            spec  = d.get("specialty", "")
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


def _build_instructions(ctx: Any, tenant: Optional[Any]) -> str:
    """Build the system prompt injected into the OpenAI Realtime session."""
    if tenant and hasattr(tenant, "name"):
        clinic_name  = getattr(tenant, "name",              settings.CLINIC_NAME)
        clinic_hours = getattr(tenant, "business_hours",    None) or settings.CLINIC_HOURS
        clinic_addr  = getattr(tenant, "address",           None) or settings.CLINIC_ADDRESS
        clinic_phone = getattr(tenant, "phone",             None) or settings.CLINIC_PHONE
        rec_name     = getattr(tenant, "receptionist_name", None) or "Lisa"
    else:
        clinic_name  = settings.CLINIC_NAME
        clinic_hours = settings.CLINIC_HOURS
        clinic_addr  = settings.CLINIC_ADDRESS
        clinic_phone = settings.CLINIC_PHONE
        rec_name     = "Lisa"

    doctor_list = _fetch_doctors_str(tenant)
    doctor_section = f"\nCLINIC DOCTORS (for reference):\n{doctor_list}\n" if doctor_list else ""

    return f"""You are {rec_name}, a warm and professional medical receptionist at {clinic_name}.

CLINIC INFORMATION:
- Name    : {clinic_name}
- Phone   : {clinic_phone}
- Address : {clinic_addr}
- Hours   : {clinic_hours}
- Caller  : {ctx.patient_phone}
{doctor_section}
YOUR ROLE & BEHAVIOUR RULES:
- You are a real human receptionist on a phone call. Never admit to being an AI.
- Speak in short, natural conversational sentences — 1 to 2 sentences per turn.
- Greet warmly, understand the caller's need, and guide them efficiently.
- Never repeat the same greeting or filler phrase twice.
- If the caller says goodbye/bye/allah hafez/biday, respond warmly and end the call.

APPOINTMENT BOOKING WORKFLOW:
1. Collect one at a time: full name → preferred doctor or specialty → reason for visit → date/time → confirm phone.
2. Summarise: "To confirm: [name], [doctor], [reason], [date/time] — shall I book that?"
3. Call book_appointment ONLY after the patient says YES.
4. After booking, read back the Appointment ID clearly.
5. Never ask for multiple details at once.

TOOL USAGE:
- check_available_doctors: when patient asks about doctors, specialties, or availability.
- get_appointment_details: when patient wants to check or cancel an existing appointment.
- book_appointment: only after explicit patient confirmation.

SPEAKING FORMAT:
- You are speaking aloud. Never use bullet points, markdown, asterisks, or numbered lists.
- Speak in complete natural sentences. Be concise, warm, and professional."""



class RealtimeBridge:
    """
    Manages one call's bidirectional bridge between Twilio and OpenAI Realtime API.

    Usage:
        bridge = RealtimeBridge(call_id, ctx, tenant)
        await bridge.run(twilio_websocket)
    """

    def __init__(self, call_id: str, ctx: Any, tenant: Optional[Any] = None) -> None:
        self.call_id   = call_id
        self.ctx       = ctx
        self.tenant    = tenant
        self.stream_sid: Optional[str] = None

        # Track in-flight function calls: {oai_call_id: {name, args_buffer}}
        self._pending_fn: Dict[str, Dict] = {}

        # Track whether a response is currently streaming (for barge-in)
        self._response_active       = False
        self._current_response_id:  Optional[str] = None

        # Set once session.update has been sent to avoid double-configure
        self._session_configured    = False

    # ── Public entry point ────────────────────────────────────────────────────
    async def run(self, twilio_ws) -> None:
        """Bridge loop — runs until the call ends or either WebSocket disconnects."""
        model   = settings.OPENAI_REALTIME_MODEL
        api_key = settings.OPENAI_API_KEY

        if not api_key:
            logger.error("OPENAI_API_KEY is not set — Realtime bridge cannot start | call=%s", self.call_id)
            return

        headers = {
            "Authorization": f"Bearer {api_key}",
            "OpenAI-Beta":   "realtime=v1",
        }
        url = f"{REALTIME_URL}?model={model}"
        logger.info("Realtime bridge starting | call=%s | model=%s", self.call_id, model)

        try:
            async with websockets.connect(url, additional_headers=headers) as oai_ws:
                logger.info("OpenAI Realtime WS connected | call=%s", self.call_id)
                # Run both directions concurrently; either side finishing stops both
                done, pending = await asyncio.wait(
                    [
                        asyncio.ensure_future(self._twilio_to_openai(twilio_ws, oai_ws)),
                        asyncio.ensure_future(self._openai_to_twilio(oai_ws, twilio_ws)),
                    ],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
        except ConnectionClosedOK:
            logger.info("Realtime WS closed normally | call=%s", self.call_id)
        except ConnectionClosedError as exc:
            logger.warning("Realtime WS closed with error | call=%s | %s", self.call_id, exc)
        except Exception as exc:
            logger.error("Realtime bridge error | call=%s | %s", self.call_id, exc, exc_info=True)
        finally:
            logger.info("Realtime bridge exited | call=%s", self.call_id)


    # ── Twilio → OpenAI ───────────────────────────────────────────────────────
    async def _twilio_to_openai(self, twilio_ws, oai_ws) -> None:
        """Receives Twilio WebSocket events and forwards audio to OpenAI."""
        async for raw in twilio_ws.iter_text():
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            event = data.get("event")

            # ── Stream start event ────────────────────────────────────────────
            if event == "start":
                start = data.get("start", {})
                self.stream_sid = start.get("streamSid") or start.get("stream_sid")
                logger.info(
                    "Twilio stream started | call=%s | stream_sid=%s",
                    self.call_id, self.stream_sid,
                )
                # Configure session on first start (session.created may beat us here)
                if not self._session_configured:
                    await self._configure_session(oai_ws)
                    self._session_configured = True

            # ── Inbound audio (patient's voice) ───────────────────────────────
            elif event == "media":
                media = data.get("media", {})
                # Only process inbound (patient) audio track
                track = media.get("track")
                if track and track not in ("inbound", ""):
                    continue
                payload = media.get("payload")
                if not payload:
                    continue
                # Payload is already base64-encoded g711_ulaw — pass directly
                await oai_ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": payload,
                }))

            # ── Stream stop ───────────────────────────────────────────────────
            elif event == "stop":
                logger.info("Twilio stream stopped | call=%s", self.call_id)
                self.ctx.end_call()
                break

            # Check if the call ended from the other side
            if self.ctx.call_ended:
                break


    # ── OpenAI → Twilio ───────────────────────────────────────────────────────
    async def _openai_to_twilio(self, oai_ws, twilio_ws) -> None:
        """Receives OpenAI Realtime events: streams audio to Twilio, handles tools & transcripts."""
        async for raw in oai_ws:
            try:
                ev = json.loads(raw)
            except json.JSONDecodeError:
                continue

            etype = ev.get("type", "")

            # ── Session ready ─────────────────────────────────────────────────
            if etype == "session.created":
                logger.info("OpenAI session created | call=%s", self.call_id)
                if not self._session_configured:
                    await self._configure_session(oai_ws)
                    self._session_configured = True

            # ── Barge-in: patient started speaking while AI is talking ────────
            elif etype == "input_audio_buffer.speech_started":
                logger.debug("VAD: speech started | call=%s", self.call_id)
                if self._response_active and self.stream_sid:
                    # Cancel the in-flight AI response
                    await oai_ws.send(json.dumps({"type": "response.cancel"}))
                    # Clear Twilio's audio buffer to stop playback immediately
                    try:
                        await twilio_ws.send_text(json.dumps({
                            "event": "clear",
                            "streamSid": self.stream_sid,
                        }))
                    except Exception:
                        pass
                    logger.debug("Barge-in: AI response cancelled | call=%s", self.call_id)

            elif etype == "input_audio_buffer.speech_stopped":
                logger.debug("VAD: speech stopped | call=%s", self.call_id)

            # ── Patient transcript (from Whisper) ─────────────────────────────
            elif etype == "conversation.item.input_audio_transcription.completed":
                text = ev.get("transcript", "").strip()
                if text:
                    logger.info("Patient[%s]: %s", self.call_id, text)
                    self.ctx.add_turn("patient", text)

            # ── AI audio response — stream directly to Twilio ─────────────────
            elif etype == "response.audio.delta":
                audio_b64 = ev.get("delta", "")
                if audio_b64 and self.stream_sid:
                    try:
                        await twilio_ws.send_text(json.dumps({
                            "event":     "media",
                            "streamSid": self.stream_sid,
                            "media":     {"payload": audio_b64},
                        }))
                    except Exception as exc:
                        logger.warning("WS send failed | call=%s | %s", self.call_id, exc)


            # ── AI text transcript ─────────────────────────────────────────────
            elif etype == "response.audio_transcript.done":
                text = ev.get("transcript", "").strip()
                if text:
                    logger.info("AI[%s]: %s", self.call_id, text[:150])
                    self.ctx.add_turn("receptionist", text)
                    # Detect goodbye to end call gracefully
                    self._check_goodbye(text)

            # ── Response lifecycle tracking ────────────────────────────────────
            elif etype == "response.created":
                self._response_active      = True
                self._current_response_id  = ev.get("response", {}).get("id")

            elif etype == "response.done":
                self._response_active     = False
                self._current_response_id = None
                # Handle function_call items that appear in the output list
                for item in ev.get("response", {}).get("output", []):
                    if item.get("type") == "function_call":
                        fn_id   = item.get("call_id", "")
                        fn_name = item.get("name", "")
                        fn_args = item.get("arguments", "")
                        # Skip if already handled via streaming delta events
                        if fn_id not in self._pending_fn:
                            await self._handle_function_call(fn_id, fn_name, fn_args, oai_ws)

            # ── Function call arguments (streaming) ────────────────────────────
            elif etype == "response.function_call_arguments.delta":
                fn_id = ev.get("call_id", "")
                if fn_id not in self._pending_fn:
                    self._pending_fn[fn_id] = {"name": "", "args_buffer": ""}
                self._pending_fn[fn_id]["args_buffer"] += ev.get("delta", "")

            elif etype == "response.function_call_arguments.done":
                fn_id   = ev.get("call_id", "")
                fn_name = ev.get("name", "")
                fn_args = ev.get("arguments", "")
                logger.info("Function call: %s | call=%s | args=%s", fn_name, self.call_id, fn_args[:80])
                await self._handle_function_call(fn_id, fn_name, fn_args, oai_ws)
                # Mark as handled so response.done doesn't double-execute
                self._pending_fn[fn_id] = {"name": fn_name, "args_buffer": fn_args}

            # ── Errors from OpenAI ─────────────────────────────────────────────
            elif etype == "error":
                err = ev.get("error", {})
                logger.error(
                    "OpenAI Realtime error | call=%s | code=%s | msg=%s",
                    self.call_id, err.get("code", "?"), err.get("message", ""),
                )

            if self.ctx.call_ended:
                try:
                    await oai_ws.close()
                except Exception:
                    pass
                break


    # ── Session configuration ─────────────────────────────────────────────────
    async def _configure_session(self, oai_ws) -> None:
        """Send session.update to set up the OpenAI Realtime session."""
        instructions = _build_instructions(self.ctx, self.tenant)

        await oai_ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities":            ["text", "audio"],
                "instructions":          instructions,
                "voice":                 settings.OPENAI_REALTIME_VOICE,
                "input_audio_format":    "g711_ulaw",
                "output_audio_format":   "g711_ulaw",
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": {
                    "type":                  "server_vad",
                    "threshold":             0.5,
                    "prefix_padding_ms":     300,
                    "silence_duration_ms":   500,
                    "create_response":       True,
                },
                "tools":       _REALTIME_TOOLS,
                "tool_choice": "auto",
                "temperature": 0.8,
                "max_response_output_tokens": 512,
            },
        }))
        logger.info("Session configured | call=%s | voice=%s | model=%s",
                    self.call_id, settings.OPENAI_REALTIME_VOICE, settings.OPENAI_REALTIME_MODEL)

        # Inject a hidden trigger so the AI speaks the opening greeting first
        await oai_ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": (
                        "[System: The patient just called. "
                        "Greet them warmly now in 1-2 natural sentences.]"
                    ),
                }],
            },
        }))
        await oai_ws.send(json.dumps({"type": "response.create"}))

    # ── Function call execution ───────────────────────────────────────────────
    async def _handle_function_call(
        self, fn_call_id: str, fn_name: str, args_str: str, oai_ws
    ) -> None:
        """Execute a tool function and return the result to OpenAI."""
        try:
            args = json.loads(args_str) if args_str.strip() else {}
        except json.JSONDecodeError:
            args = {}

        # Run potentially blocking tool in a thread pool
        result = await asyncio.to_thread(_execute_tool, fn_name, args)
        logger.info("Tool '%s' result | call=%s | %.120s", fn_name, self.call_id, result)

        # Update call context if an appointment was successfully booked
        if fn_name == "book_appointment" and "error" not in result.lower():
            self.ctx.appointment_booked = True
            m = re.search(r"\bID:\s*([a-f0-9\-]{8,})", result, re.IGNORECASE)
            if m:
                self.ctx.booked_appointment_id = m.group(1)

        # Return the result to OpenAI so it can continue the conversation
        await oai_ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type":    "function_call_output",
                "call_id": fn_call_id,
                "output":  result,
            },
        }))
        await oai_ws.send(json.dumps({"type": "response.create"}))

    # ── Goodbye detection ─────────────────────────────────────────────────────
    def _check_goodbye(self, text: str) -> None:
        """End the call context if the AI's response contains a farewell."""
        lower = text.lower()
        farewell_markers = [
            "goodbye", "bye bye", "have a great day", "take care",
            "allah hafez", "khoda hafez", "biday", "thanks for calling",
            "thank you for calling",
        ]
        if any(m in lower for m in farewell_markers):
            logger.info("Farewell detected in AI response — ending call | call=%s", self.call_id)
            self.ctx.end_call()
