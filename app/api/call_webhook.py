"""
Call Webhook API
================
Twilio webhook endpoints + WebSocket for real-time audio.

Routes:
  POST /webhooks/call/answer          — inbound call connects (TwiML)
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
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone  # noqa: F401 — timezone used in realtime block

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
) -> float:
    audio_bytes, mime_type = await tts.synthesize(text)
    ulaw = await asyncio.to_thread(_to_mulaw_8k, audio_bytes, mime_type)
    await _send_mulaw_over_ws(websocket, stream_sid, ulaw)
    # Approx duration in seconds for 8kHz mu-law (1 byte per sample)
    return len(ulaw) / 8000.0


# ── Inbound call answer ────────────────────────────────────────────────────────
async def _call_answer_impl(request: Request, call_id: str = Query(...)):
    await _validate_twilio(request)
    form = await request.form()
    twilio_sid = form.get("CallSid", "")
    caller = form.get("From", "")
    to_number = (form.get("To") or "").strip()
    effective_call_id = twilio_sid or call_id
    if call_id and call_id.upper() != "NEW":
        effective_call_id = call_id

    tenant_id = ""
    from app.services.tenant_service import get_tenant_by_twilio_phone
    tenant = await get_tenant_by_twilio_phone(to_number) if to_number else None
    if tenant:
        tenant_id = tenant.id
        logger.info("Inbound call resolved to tenant=%s (%s)", tenant_id, tenant.name)
    else:
        logger.info("Inbound call: no tenant for To=%s; using single-tenant fallback", to_number)

    logger.info(
        "Inbound call received call_id=%s effective_call_id=%s sid=%s from=%s to=%s",
        call_id,
        effective_call_id,
        twilio_sid,
        caller,
        to_number,
    )

    call_doc = Call(
        id=effective_call_id,
        tenant_id=tenant_id,
        patient_phone=caller,
        direction=CallDirection.INBOUND,
        status=CallStatus.IN_PROGRESS,
        twilio_call_sid=twilio_sid,
        started_at=utcnow(),
    )
    await calls_col().insert_one(call_doc.model_dump())

    twiml = TwilioService().twiml_answer(effective_call_id, tenant=tenant)
    return Response(content=twiml, media_type="application/xml")


@router.get("/answer", operation_id="call_answer_get")
async def call_answer_get(request: Request, call_id: str = Query(...)):
    return await _call_answer_impl(request, call_id)


@router.post("/answer", operation_id="call_answer_post")
async def call_answer_post(request: Request, call_id: str = Query(...)):
    return await _call_answer_impl(request, call_id)


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


# Stream status callback (Twilio Media Streams)
@router.post("/stream/status")
async def stream_status(request: Request):
    try:
        form = await request.form()
        logger.info("Stream status callback: %s", dict(form))
    except Exception as exc:
        logger.error("Stream status callback error: %s", exc)
    return Response(content="", status_code=204)


# ── WebSocket: real-time audio stream ─────────────────────────────────────────
@router.websocket("/ws/stream/{call_id}")
async def audio_stream(websocket: WebSocket, call_id: str):
    """
    Twilio Media Streams WebSocket.

    Audio pipeline (classic):
      Twilio mu-law audio -> STT -> CrewAI -> TTS -> Twilio

    Audio pipeline (Realtime, when USE_REALTIME_API=true):
      Twilio mu-law audio <-> OpenAI gpt-4o-realtime-preview (single WS, zero conversion)
    """
    await websocket.accept()
    logger.info(
        "WS opened | call_id=%s | realtime=%s | provider=%s",
        call_id, settings.USE_REALTIME_API, settings.LLM_PROVIDER,
    )

    # ── OpenAI Realtime API fast path ─────────────────────────────────────────
    if settings.USE_REALTIME_API:
        call_doc = await calls_col().find_one({"id": call_id}, sort=[("created_at", -1)])
        patient_phone = call_doc["patient_phone"] if call_doc else "unknown"
        direction     = call_doc.get("direction", "inbound") if call_doc else "inbound"
        appt_id       = call_doc.get("appointment_id")       if call_doc else None

        from app.services.tenant_service import get_tenant_for_call
        tenant = await get_tenant_for_call(call_id) if call_doc else None

        # Re-use or create a lightweight CallContext for transcript / booking state
        existing = get_session(call_id)
        if existing:
            ctx = existing.ctx
            ctx.call_ended = False
        else:
            ctx = CallContext(
                call_id=call_id,
                patient_phone=patient_phone,
                direction=direction,
                is_reminder_call=False,
                appointment_id=appt_id,
                started_at=utcnow(),
                tenant=tenant,
            )
            create_session(ctx)

        from app.services.realtime_service import RealtimeBridge
        bridge = RealtimeBridge(call_id=call_id, ctx=ctx, tenant=tenant)
        try:
            await bridge.run(websocket)
        except Exception as rt_exc:
            logger.error("Realtime bridge failed | call=%s | %s", call_id, rt_exc, exc_info=True)
        finally:
            if ctx.appointment_booked:
                await calls_col().update_one(
                    {"id": call_id},
                    {"$set": {
                        "appointment_booked":  True,
                        "appointment_id":      ctx.booked_appointment_id,
                        "transcript":          ctx.full_transcript,
                    }},
                )
            destroy_session(call_id)
            logger.info(
                "Realtime WS closed | call=%s | duration=%.1fs",
                call_id, (datetime.now(timezone.utc) - ctx.started_at).total_seconds()
                         if ctx.started_at else 0,
            )
        return   # ← do NOT fall through to the classic pipeline below
    # ─────────────────────────────────────────────────────────────────────────

    stt = STTService()
    tts = TTSService()

    call_doc = await calls_col().find_one({"id": call_id}, sort=[("created_at", -1)])
    patient_phone = call_doc["patient_phone"] if call_doc else "unknown"
    direction = call_doc.get("direction", "inbound") if call_doc else "inbound"
    appt_id = call_doc.get("appointment_id") if call_doc else None
    twilio_sid = call_doc.get("twilio_call_sid") if call_doc else None
    stream_sid: str | None = None

    from app.services.tenant_service import get_tenant_for_call
    tenant = await get_tenant_for_call(call_id) if call_doc else None

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
            is_reminder_call=False,
            appointment_id=appt_id,
            started_at=utcnow(),
            tenant=tenant,
        )
        crew_session = create_session(ctx)

    utterance_queue: asyncio.Queue[str] = asyncio.Queue()
    last_ai_reply: dict[str, str] = {"text": ""}
    last_patient_text: dict[str, str] = {"text": ""}
    last_patient_ts: dict[str, float] = {"ts": 0.0}
    tts_playing_until = 0.0
    tts_started_at = 0.0
    last_user_activity = time.time()
    pending_parts: list[str] = []
    pending_flush_task: asyncio.Task | None = None
    closing_requested = False
    closing_sent = False
    awaiting_user_turn = True

    def _normalize_text(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()

    def _is_goodbye_intent(text: str) -> bool:
        n = _normalize_text(text)
        if not n:
            return False
        keys = [
            "bye", "bye bye", "goodbye", "take care", "see you",
            "thank you bye", "thanks bye", "allah hafez", "khoda hafez", "biday",
        ]
        return any(k in n for k in keys)

    def _is_echo(text: str, last_reply: str) -> bool:
        if not text or not last_reply:
            return False
        u = _normalize_text(text)
        a = _normalize_text(last_reply)
        if not u or not a:
            return False
        # Direct containment is a strong echo signal.
        if (u in a or a in u) and len(u) >= 6:
            return True
        # Token overlap heuristic (guards against minor ASR drift).
        u_tokens = set(u.split())
        a_tokens = set(a.split())
        if not u_tokens or not a_tokens:
            return False
        overlap = len(u_tokens & a_tokens) / max(1, len(u_tokens))
        return overlap >= 0.6 and len(u_tokens) <= 18

    async def _flush_pending_user_utterance(delay: float = 1.0) -> None:
        await asyncio.sleep(delay)
        if not pending_parts:
            return
        merged = " ".join(pending_parts).strip()
        pending_parts.clear()
        if len(merged) < 2:
            return
        norm_merged = _normalize_text(merged)
        if norm_merged == _normalize_text(last_patient_text.get("text", "")) and time.time() - last_patient_ts["ts"] < 3.0:
            return
        last_patient_text["text"] = merged
        last_patient_ts["ts"] = time.time()
        await utterance_queue.put(merged)

    async def on_transcript(text: str, is_final: bool) -> None:
        nonlocal last_user_activity, pending_flush_task
        if is_final and text.strip():
            if closing_requested:
                return
            # Strict turn-taking: while AI is speaking, ignore incoming transcript.
            if (not awaiting_user_turn) or (time.time() < tts_playing_until + 0.2):
                logger.info("Ignoring transcript while AI turn is active on call %s: %s", call_id, text)
                return
            if _is_echo(text, last_ai_reply.get("text", "")):
                logger.info("Ignoring probable echo on call %s: %s", call_id, text)
                return
            last_user_activity = time.time()
            normalized = text.strip()
            if len(_normalize_text(normalized)) < 2:
                return
            if pending_parts:
                prev = pending_parts[-1].strip().lower()
                cur = normalized.lower()
                if cur == prev:
                    return
            pending_parts.append(normalized)
            if pending_flush_task and not pending_flush_task.done():
                pending_flush_task.cancel()
            pending_flush_task = asyncio.create_task(_flush_pending_user_utterance(1.0))

    stt_conn = await stt.start_streaming_session(on_transcript)

    async def process_loop():
        nonlocal twilio_sid, stream_sid, tts_started_at, tts_playing_until, closing_requested, closing_sent, awaiting_user_turn
        inactivity_warning_sent = False
        max_inactivity_seconds = 180  # 3 minutes of silence before warning
        
        while not ctx.call_ended:
            try:
                utterance = await asyncio.wait_for(utterance_queue.get(), timeout=1.0)
                if closing_requested:
                    continue
                awaiting_user_turn = False
                logger.info("Patient[%s]: %s", call_id, utterance)
                inactivity_warning_sent = False  # Reset warning flag on activity

                if _is_goodbye_intent(utterance):
                    ai_reply = "Thank you for calling HealthCare Medical Center. Have a great day. Goodbye."
                    closing_requested = True
                    closing_sent = True
                else:
                    ai_reply = await _run_sync(crew_session.process_utterance, utterance)
                logger.info("AI[%s]: %s", call_id, str(ai_reply)[:80])

                if ai_reply and stream_sid:
                    last_ai_reply["text"] = str(ai_reply)
                    try:
                        tts_started_at = time.time()
                        dur = await _deliver_tts_ws(websocket, stream_sid, tts, str(ai_reply))
                        tts_playing_until = time.time() + dur
                        awaiting_user_turn = True
                    except WebSocketDisconnect:
                        logger.warning("WebSocket disconnected during TTS delivery for call %s", call_id)
                        raise
                    except Exception as ws_tts_exc:
                        logger.warning("WS TTS failed (fallback to call update): %s", ws_tts_exc, exc_info=True)
                        if twilio_sid:
                            try:
                                await _deliver_tts(tts, twilio_sid, str(ai_reply), call_id)
                                awaiting_user_turn = True
                            except Exception as fallback_exc:
                                logger.error("Fallback TTS delivery also failed: %s", fallback_exc, exc_info=True)
                    if closing_sent:
                        ctx.end_call()
                        try:
                            await websocket.close()
                        except Exception:
                            pass
                        break
                else:
                    awaiting_user_turn = True

            except asyncio.TimeoutError:
                # Check for inactivity and send keepalive
                inactivity = time.time() - last_user_activity
                if inactivity > max_inactivity_seconds and not inactivity_warning_sent and stream_sid:
                    logger.info("Call %s: %d seconds of inactivity, sending keepalive message", call_id, int(inactivity))
                    try:
                        keepalive_msg = "I'm still here if you need anything. Are you still there?"
                        awaiting_user_turn = False
                        tts_started_at = time.time()
                        await _deliver_tts_ws(websocket, stream_sid, tts, keepalive_msg)
                        tts_playing_until = time.time() + 2.0
                        awaiting_user_turn = True
                        inactivity_warning_sent = True
                    except Exception as ka_exc:
                        logger.warning("Keepalive message failed: %s", ka_exc)
                continue
            except Exception as exc:
                logger.error("process_loop error call=%s: %s", call_id, exc, exc_info=True)
                awaiting_user_turn = True

    proc_task = asyncio.create_task(process_loop())

    try:
        while True:
            try:
                raw = await websocket.receive_text()
                data = json.loads(raw)
            except json.JSONDecodeError as json_exc:
                logger.warning("Invalid JSON from WebSocket call=%s: %s", call_id, json_exc)
                continue
            except RuntimeError as recv_exc:
                # Raised when process_loop closes the WS (normal goodbye path)
                if "not connected" in str(recv_exc).lower():
                    logger.info("WS closed normally after goodbye | call=%s", call_id)
                else:
                    logger.error("WS receive error call=%s: %s", call_id, recv_exc, exc_info=True)
                break
            except Exception as recv_exc:
                logger.error("Error receiving WebSocket data call=%s: %s", call_id, recv_exc, exc_info=True)
                break
                
            event = data.get("event")
            if event == "start":
                start = data.get("start", {})
                sid_from_start = start.get("callSid") or start.get("call_sid")
                if sid_from_start:
                    twilio_sid = sid_from_start
                stream_sid = start.get("streamSid") or start.get("stream_sid")
                logger.info("Stream started | call_id=%s | stream_sid=%s | twilio_sid=%s", call_id, stream_sid, twilio_sid)
                if is_new_session and stream_sid:
                    greeting = await _run_sync(crew_session.generate_greeting)
                    logger.info("Greeting[%s]: %s", call_id, str(greeting)[:80])
                    try:
                        awaiting_user_turn = False
                        tts_started_at = time.time()
                        dur = await _deliver_tts_ws(websocket, stream_sid, tts, str(greeting))
                        tts_playing_until = time.time() + dur
                        awaiting_user_turn = True
                    except Exception as ws_greet_exc:
                        logger.warning("WS greeting failed (fallback to call update): %s", ws_greet_exc, exc_info=True)
                        if twilio_sid:
                            try:
                                await _deliver_tts(tts, twilio_sid, str(greeting), call_id)
                                awaiting_user_turn = True
                            except Exception as fallback_greet_exc:
                                logger.error("Fallback greeting delivery failed: %s", fallback_greet_exc, exc_info=True)
                    is_new_session = False
            elif event == "media":
                try:
                    media = data.get("media", {})
                    track = media.get("track")
                    if track and track != "inbound":
                        continue
                    payload = media.get("payload")
                    if not payload:
                        continue
                    audio = base64.b64decode(payload)
                    stt_conn.send(audio)
                except Exception as media_exc:
                    logger.warning("Error processing media event call=%s: %s", call_id, media_exc)
            elif event == "stop":
                logger.info("Stream stop event received | call_id=%s", call_id)
                ctx.end_call()
                break
            else:
                logger.debug("Unknown WebSocket event call=%s: %s", call_id, event)
    except WebSocketDisconnect:
        logger.info("WS disconnected call_id=%s", call_id)
    except Exception as exc:
        logger.error("WS error call=%s: %s", call_id, exc, exc_info=True)
    finally:
        proc_task.cancel()
        if pending_flush_task and not pending_flush_task.done():
            pending_flush_task.cancel()
        try:
            await proc_task
        except asyncio.CancelledError:
            pass
        except Exception as task_exc:
            logger.error("Error cancelling process_loop task call=%s: %s", call_id, task_exc)
        
        try:
            await stt_conn.finish()
        except Exception as stt_exc:
            logger.error("Error finishing STT connection call=%s: %s", call_id, stt_exc)
        
        logger.info("WS closed call_id=%s | total_duration=%.1fs", call_id, time.time() - ctx.started_at.timestamp())

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
