"""
Twilio Service - inbound TwiML, stream management, and live TTS injection.
"""
from __future__ import annotations
import logging
from urllib.parse import urlparse, urlunparse
from typing import Any, Optional

import httpx
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream

from app.config import settings

logger = logging.getLogger(__name__)


class TwilioService:
    def __init__(self):
        self.client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
        self.from_number = settings.TWILIO_PHONE_NUMBER
        self.base_url = settings.TWILIO_WEBHOOK_BASE_URL.rstrip("/")

    def _build_ws_url(self, path: str) -> str:
        if not self.base_url:
            raise RuntimeError("TWILIO_WEBHOOK_BASE_URL is not set")

        raw = self.base_url
        parsed = urlparse(raw)
        if not parsed.scheme and not parsed.netloc and parsed.path:
            # Handle values like "example.ngrok-free.dev"
            raw = f"https://{raw}"
            parsed = urlparse(raw)

        if not parsed.scheme or not parsed.netloc:
            raise RuntimeError(f"Invalid TWILIO_WEBHOOK_BASE_URL: {self.base_url}")

        ws_scheme = "wss" if parsed.scheme == "https" else "ws"
        ws_url = urlunparse((ws_scheme, parsed.netloc, path, "", "", ""))
        return ws_url

    # ── Inbound call answer TwiML ──────────────────────────────────────────────
    def twiml_answer(self, call_id: str, tenant: Optional[Any] = None) -> str:
        """Return TwiML that opens a media stream for real-time audio processing."""
        company_name = getattr(tenant, "name", None) if tenant else None
        if not company_name:
            company_name = settings.CLINIC_NAME
        resp = VoiceResponse()
        resp.say(
            f"Hello, thank you for calling {company_name}. One moment please.",
            voice="Polly.Joanna",
            language="en-US",
        )
        resp.pause(length=1)
        try:
            ws_url = self._build_ws_url(f"/v1/webhooks/call/ws/stream/{call_id}")
            logger.info("Twilio Media Stream URL (inbound): %s", ws_url)
            stream = Stream(
                url=ws_url,
                status_callback=f"{self.base_url}/v1/webhooks/call/stream/status",
                status_callback_method="POST",
            )
            connect = Connect()
            connect.append(stream)
            resp.append(connect)
        except Exception as exc:
            logger.error("Failed to build WS URL for inbound call: %s", exc)
            resp.say(
                f"Hello, thank you for calling {company_name}.",
                voice="Polly.Joanna",
                language="en-US",
            )
        return str(resp)

    # ── Inject TTS audio into live call ───────────────────────────────────────
    async def inject_tts_to_call(self, call_sid: str, audio_url: str, call_id: str) -> None:
        twiml = VoiceResponse()
        twiml.play(audio_url)
        # Reconnect media stream so the conversation can continue after playback.
        connect = Connect()
        stream = Stream(url=self._build_ws_url(f"/v1/webhooks/call/ws/stream/{call_id}"))
        connect.append(stream)
        twiml.append(connect)
        self.client.calls(call_sid).update(twiml=str(twiml))

    # ── Validate Twilio webhook signature ─────────────────────────────────────
    def validate_signature(self, url: str, params: dict, signature: str) -> bool:
        from twilio.request_validator import RequestValidator
        validator = RequestValidator(settings.TWILIO_AUTH_TOKEN)
        return validator.validate(url, params, signature)
