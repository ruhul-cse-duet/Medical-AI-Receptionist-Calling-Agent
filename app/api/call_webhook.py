"""
Call Webhook API
================
Twilio webhook endpoints + WebSocket for real-time audio.

Routes:
  POST /webhooks/call/answer          — inbound call connects (TwiML)
  POST /webhooks/call/reminder        — reminder call TwiML
  POST /webhooks/call/status          — call lifecycle updates
  WS   /webhooks/call/ws/stream/{id}  — real-time audio stream (STT → CrewAI → TTS)
"""
from __future__ import annotations

import asyncio
import audioop
import base64
import io
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from fastapi import APIRouter, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

# ── CrewAI receptionist (replaces old single-agent receptionist.py) ───────────
from app.config import settings


def _load_agent_backend():
    """
    Lazy import to avoid CrewAI/LiteLLM import errors when using the simple backend.
    Falls back to simple backend if CrewAI cannot be imported.
    """
    log = logging.getLogger(__name__)
    if settings.AGENT_BACKEND == "simple":
        from app.agents.simple_receptionist import (
            CallContext,
            create_session,
            get_session,
            destroy_session,
        )
        return CallContext, create_session, get_session, destroy_session

    try:
        from app.agents.crew_receptionist import (
            CallContext,
            create_session,
            get_session,
            destroy_session,
        )
        return CallContext, create_session, get_session, destroy_session
    except Exception as exc:
        log.error("CrewAI backend failed to import; falling back to simple. Error: %s", exc)
        from app.agents.simple_receptionist import (
            CallContext,
            create_session,
            get_session,
            destroy_session,
        )
        return CallContext, create_session, get_session, destroy_session


CallContext, create_session, get_session, destroy_session = _load_agent_backend()
from app.db.base import calls_col, appointments_col
from app.db.models import Call, CallDirection, CallStatus, utcnow
from app.services.stt_service import STTService
from app.services.tts_service import TTSService
from app.services.twilio_service import TwilioService

router = APIRouter(prefix="/webhooks/call", tags=["Twilio Webhooks"])
logger = logging.getLogger(__name__)

# Thread pool for running CrewAI (sync) from async context
_executor = ThreadPoolExecutor(max_workers=4)


async def _validate_twilio(request: Request) -> None:
    if settings.APP_ENV == "development":
        return
    sig = request.headers.get("X-Twilio-Signature", "")
    form = dict(await request.form())
    if not TwilioService().validate_signature(str(request.url), form, sig):
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail="Invalid Twilio signature")


def _run_sync(fn, *args):
    """Run a sync function (CrewAI) in a thread pool so it doesn't block the event loop."""
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(_executor, fn, *args)


def _pcm_from_wav_bytes(wav_bytes: bytes) -> tuple[bytes, int, int, int]:
    import wave

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        rate = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())
    return pcm, rate, channels, sampwidth


def _to_mulaw_8k(audio_bytes: bytes, mime_type: str) -> bytes:
    """
    Convert audio bytes to 8kHz mono mu-law for Twilio Media Streams.
    Supports WAV natively. For MP3/other, tries pydub if available.
    """
    pcm = b""
    rate = 8000
    channels = 1
    sampwidth = 2

    if "wav" in (mime_type or "").lower():
        pcm, rate, channels, sampwidth = _pcm_from_wav_bytes(audio_bytes)
    else:
        m = (mime_type or "").lower()
        fmt = None
        if "mpeg" in m or "mp3" in m:
            fmt = "mp3"
        elif "ogg" in m:
            fmt = "ogg"
        elif "webm" in m:
            fmt = "webm"

        try:
            # Optional decode path for mp3/ogg/etc.
            from pydub import AudioSegment  # type: ignore

            seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
            pcm = seg.raw_data
            rate = seg.frame_rate
            channels = seg.channels
            sampwidth = seg.sample_width
        except Exception:
            # Fallback via ffmpeg if pydub/decoder is unavailable.
            import os
            import subprocess
            import tempfile

            suffix = f".{fmt}" if fmt else ".bin"
            in_path = ""
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
                    tf.write(audio_bytes)
                    in_path = tf.name
                proc = subprocess.run(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-i",
                        in_path,
                        "-ar",
                        "8000",
                        "-ac",
                        "1",
                        "-f",
                        "mulaw",
                        "pipe:1",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                if proc.returncode != 0 or not proc.stdout:
                    raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore") or "ffmpeg decode failed")
                return proc.stdout
            finally:
                if in_path:
                    try:
                        os.remove(in_path)
                    except Exception:
                        pass

    if channels > 1:
        pcm = audioop.tomono(pcm, sampwidth, 0.5, 0.5)
        channels = 1
    if sampwidth != 2:
        pcm = audioop.lin2lin(pcm, sampwidth, 2)
        sampwidth = 2
    if rate != 8000:
        pcm, _ = audioop.ratecv(pcm, 2, 1, rate, 8000, None)

    return audioop.lin2ulaw(pcm, 2)


async def _send_mulaw_over_ws(websocket: WebSocket, stream_sid: str, ulaw: bytes) -> None:
    # 20 ms chunks (160 bytes @ 8kHz mu-law)
    chunk_size = 160
    for i in range(0, len(ulaw), chunk_size):
        payload = base64.b64encode(ulaw[i:i + chunk_size]).decode("ascii")
        await websocket.send_text(
            json.dumps(
                {"event": "media", "streamSid": stream_sid, "media": {"payload": payload}}
            )
        )


async def _deliver_tts_ws(
    websocket: WebSocket,
    stream_sid: str,
    tts: TTSService,
    text: str,
) -> None:
    audio_bytes, mime_type = await tts.synthesize(text)
    ulaw = await asyncio.to_thread(_to_mulaw_8k, audio_bytes, mime_type)
    await _send_mulaw_over_ws(websocket, stream_sid, ulaw)


# ── Inbound call answer ────────────────────────────────────────────────────────
async def _call_answer_impl(request: Request, call_id: str = Query(...)):
    await _validate_twilio(request)
    form = await request.form()
    twilio_sid = form.get("CallSid", "")
    caller = form.get("From", "")
    effective_call_id = twilio_sid or call_id
    if call_id and call_id.upper() != "NEW":
        effective_call_id = call_id
    logger.info(
        "Inbound call received call_id=%s effective_call_id=%s sid=%s from=%s",
        call_id,
        effective_call_id,
        twilio_sid,
        caller,
    )

    call_doc = Call(
        id=effective_call_id,
        patient_phone=caller,
        direction=CallDirection.INBOUND,
        status=CallStatus.IN_PROGRESS,
        twilio_call_sid=twilio_sid,
        started_at=utcnow(),
    )
    await calls_col().insert_one(call_doc.model_dump())

    twiml = TwilioService().twiml_answer(effective_call_id)
    return Response(content=twiml, media_type="application/xml")


@router.get("/answer", operation_id="call_answer_get")
async def call_answer_get(request: Request, call_id: str = Query(...)):
    return await _call_answer_impl(request, call_id)


@router.post("/answer", operation_id="call_answer_post")
async def call_answer_post(request: Request, call_id: str = Query(...)):
    return await _call_answer_impl(request, call_id)


# ── Reminder call TwiML ────────────────────────────────────────────────────────
@router.post("/reminder")
async def reminder_answer(request: Request, call_id: str = Query(...)):
    await _validate_twilio(request)
    call = await calls_col().find_one({"id": call_id})
    appt = None
    if call and call.get("appointment_id"):
        appt = await appointments_col().find_one({"id": call["appointment_id"]})

    if appt:
        dt = appt["scheduled_at"]
        if isinstance(dt, str):
            dt = datetime.fromisoformat(dt)
        time_str = dt.strftime("%A %B %d at %I:%M %p")
        msg = (
            f"Hello {appt['patient_name']}, this is a reminder from {settings.CLINIC_NAME}. "
            f"You have an appointment with {appt['doctor_name']} for {appt['treatment_title']} "
            f"on {time_str}. We look forward to seeing you!"
        )
    else:
        msg = f"Hello, this is a reminder from {settings.CLINIC_NAME} about your upcoming appointment."

    return Response(content=TwilioService().twiml_reminder(msg), media_type="application/xml")


# ── Call status callback ───────────────────────────────────────────────────────
@router.post("/status")
async def call_status(request: Request, call_id: str = Query(...)):
    await _validate_twilio(request)
    form = await request.form()
    twilio_status = form.get("CallStatus", "").lower()
    duration = form.get("CallDuration")

    status_map = {
        "in-progress": CallStatus.IN_PROGRESS,
        "completed":   CallStatus.COMPLETED,
        "failed":      CallStatus.FAILED,
        "no-answer":   CallStatus.NO_ANSWER,
        "busy":        CallStatus.FAILED,
    }
    update = {"status": status_map.get(twilio_status, CallStatus.COMPLETED).value}
    if duration:
        update["duration_seconds"] = int(duration)
    if twilio_status in ("completed", "failed", "no-answer", "busy"):
        update["ended_at"] = utcnow()

    await calls_col().update_one({"id": call_id}, {"$set": update})

    if twilio_status == "completed":
        session = get_session(call_id)
        if session:
            session.ctx.end_call()
            if session.ctx.appointment_booked:
                await calls_col().update_one(
                    {"id": call_id},
                    {"$set": {
                        "appointment_booked": True,
                        "appointment_id": session.ctx.booked_appointment_id,
                        "transcript": session.ctx.full_transcript,
                    }},
                )
            destroy_session(call_id)

    return Response(content="", status_code=204)


# ── WebSocket: real-time audio stream ─────────────────────────────────────────
@router.websocket("/ws/stream/{call_id}")
async def audio_stream(websocket: WebSocket, call_id: str):
    """
    Twilio Media Streams WebSocket.

    Audio pipeline:
      Twilio micro-law audio -> STT -> CrewAI -> TTS -> Twilio
    """
    await websocket.accept()
    logger.info("WS opened | call_id=%s | provider=%s", call_id, settings.LLM_PROVIDER)

    stt = STTService()
    tts = TTSService()

    call_doc = await calls_col().find_one({"id": call_id}, sort=[("created_at", -1)])
    patient_phone = call_doc["patient_phone"] if call_doc else "unknown"
    direction = call_doc.get("direction", "inbound") if call_doc else "inbound"
    is_reminder = call_doc.get("is_reminder_call", False) if call_doc else False
    appt_id = call_doc.get("appointment_id") if call_doc else None
    twilio_sid = call_doc.get("twilio_call_sid") if call_doc else None
    stream_sid: str | None = None

    existing_session = get_session(call_id)
    is_new_session = existing_session is None
    if existing_session:
        crew_session = existing_session
        ctx = existing_session.ctx
        ctx.call_ended = False
    else:
        ctx = CallContext(
            call_id=call_id,
            patient_phone=patient_phone,
            direction=direction,
            is_reminder_call=is_reminder,
            appointment_id=appt_id,
            started_at=utcnow(),
        )
        crew_session = create_session(ctx)

    utterance_queue: asyncio.Queue[str] = asyncio.Queue()

    async def on_transcript(text: str, is_final: bool) -> None:
        if is_final and text.strip():
            await utterance_queue.put(text)

    stt_conn = await stt.start_streaming_session(on_transcript)

    async def process_loop():
        nonlocal twilio_sid, stream_sid
        while not ctx.call_ended:
            try:
                utterance = await asyncio.wait_for(utterance_queue.get(), timeout=1.0)
                logger.info("Patient[%s]: %s", call_id, utterance)

                ai_reply = await _run_sync(crew_session.process_utterance, utterance)
                logger.info("AI[%s]: %s", call_id, str(ai_reply)[:80])

                if ai_reply and stream_sid:
                    try:
                        await _deliver_tts_ws(websocket, stream_sid, tts, str(ai_reply))
                    except Exception as ws_tts_exc:
                        logger.warning("WS TTS failed (fallback to call update): %s", ws_tts_exc)
                        if twilio_sid:
                            await _deliver_tts(tts, twilio_sid, str(ai_reply), call_id)

            except asyncio.TimeoutError:
                continue
            except Exception as exc:
                logger.error("process_loop error call=%s: %s", call_id, exc)

    proc_task = asyncio.create_task(process_loop())

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            event = data.get("event")
            if event == "start":
                start = data.get("start", {})
                sid_from_start = start.get("callSid") or start.get("call_sid")
                if sid_from_start:
                    twilio_sid = sid_from_start
                stream_sid = start.get("streamSid") or start.get("stream_sid")
                if is_new_session and stream_sid:
                    greeting = await _run_sync(crew_session.generate_greeting)
                    logger.info("Greeting[%s]: %s", call_id, str(greeting)[:80])
                    try:
                        await _deliver_tts_ws(websocket, stream_sid, tts, str(greeting))
                    except Exception as ws_greet_exc:
                        logger.warning("WS greeting failed (fallback to call update): %s", ws_greet_exc)
                        if twilio_sid:
                            await _deliver_tts(tts, twilio_sid, str(greeting), call_id)
                    is_new_session = False
            elif event == "media":
                audio = base64.b64decode(data["media"]["payload"])
                stt_conn.send(audio)
            elif event == "stop":
                ctx.end_call()
                break
    except WebSocketDisconnect:
        logger.info("WS disconnected call_id=%s", call_id)
    except Exception as exc:
        logger.error("WS error call=%s: %s", call_id, exc)
    finally:
        proc_task.cancel()
        await stt_conn.finish()
        logger.info("WS closed call_id=%s", call_id)

async def _deliver_tts(tts: TTSService, twilio_sid: str, text: str, call_id: str) -> None:
    """Synthesize speech and inject into the live Twilio call."""
    try:
        tts_result = await tts.synthesize_and_upload(text)
        twilio = TwilioService()
        if tts_result.startswith("__TEXT_FALLBACK__|"):
            fallback = tts_result.split("|", 1)[1]
            from twilio.twiml.voice_response import VoiceResponse, Connect, Stream
            v = VoiceResponse()
            v.say(fallback, voice="Polly.Joanna")
            connect = Connect()
            connect.append(Stream(url=twilio._build_ws_url(f"/v1/webhooks/call/ws/stream/{call_id}")))
            v.append(connect)
            twilio.client.calls(twilio_sid).update(twiml=str(v))
        else:
            await twilio.inject_tts_to_call(twilio_sid, tts_result, call_id)
    except Exception as exc:
        logger.error("TTS delivery failed: %s", exc)
