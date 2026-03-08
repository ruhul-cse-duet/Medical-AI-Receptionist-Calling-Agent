"""
TTS Service â€” Edge TTS or ElevenLabs
Returns MP3 bytes and exposes cached audio for Twilio Play.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
import io
from typing import Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_AUDIO_CACHE: dict[str, tuple[bytes, str, float]] = {}
_CACHE_LOCK = threading.Lock()
_COQUI_LOCK = threading.Lock()
_COQUI_MODEL = None


def _cache_put(audio: bytes, mime_type: str, ttl_seconds: int) -> str:
    tts_id = str(uuid.uuid4())
    expires_at = time.time() + max(30, int(ttl_seconds))
    with _CACHE_LOCK:
        _AUDIO_CACHE[tts_id] = (audio, mime_type, expires_at)
    return tts_id


def get_cached_audio(tts_id: str) -> Optional[tuple[bytes, str]]:
    now = time.time()
    with _CACHE_LOCK:
        entry = _AUDIO_CACHE.get(tts_id)
        if not entry:
            return None
        audio, mime_type, expires_at = entry
        if expires_at <= now:
            _AUDIO_CACHE.pop(tts_id, None)
            return None
        return audio, mime_type


class TTSService:
    def __init__(self) -> None:
        self.provider = settings.TTS_PROVIDER
        self.headers = {
            "xi-api-key": settings.ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
        }
        self.eleven_url = (
            f"https://api.elevenlabs.io/v1/text-to-speech/{settings.ELEVENLABS_VOICE_ID}"
            "/stream"
        )

    async def synthesize(self, text: str) -> tuple[bytes, str]:
        """
        Synthesize text and return (audio_bytes, mime_type).
        """
        if self.provider == "none":
            raise RuntimeError("TTS disabled (provider=none)")

        if self.provider == "edge":
            import edge_tts

            communicate = edge_tts.Communicate(
                text,
                voice=settings.EDGE_TTS_VOICE,
                rate=settings.EDGE_TTS_RATE,
                pitch=settings.EDGE_TTS_PITCH,
            )
            chunks = []
            try:
                async for chunk in communicate.stream():
                    if chunk.get("type") == "audio":
                        chunks.append(chunk.get("data", b""))
                audio = b"".join(chunks)
                if not audio:
                    raise RuntimeError("Edge TTS returned empty audio")
                logger.info("Edge TTS synthesized %d chars â†’ %d bytes", len(text), len(audio))
                return audio, "audio/mpeg"
            except Exception as exc:
                logger.error("Edge TTS failed: %s", exc)
                if settings.ELEVENLABS_API_KEY:
                    logger.info("Falling back to ElevenLabs TTS")
                    return await self._synthesize_elevenlabs(text)
                raise

        if self.provider == "piper":
            return self._synthesize_piper(text)

        if self.provider == "gtts":
            return self._synthesize_gtts(text)

        if self.provider == "coqui":
            return self._synthesize_coqui(text)

        return await self._synthesize_elevenlabs(text)

    async def _synthesize_elevenlabs(self, text: str) -> tuple[bytes, str]:
        payload = {
            "text": text,
            "model_id": settings.ELEVENLABS_MODEL_ID,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": True,
            },
        }
        async with httpx.AsyncClient(timeout=30) as c:
            resp = await c.post(self.eleven_url, json=payload, headers=self.headers)
            resp.raise_for_status()
            audio = resp.content
            logger.info("ElevenLabs synthesized %d chars â†’ %d bytes", len(text), len(audio))
            return audio, "audio/mpeg"

    def _synthesize_piper(self, text: str) -> tuple[bytes, str]:
        import os
        import subprocess
        import tempfile

        model_path = settings.PIPER_MODEL_PATH
        if not model_path:
            raise RuntimeError("PIPER_MODEL_PATH is required for Piper TTS")
        if not os.path.exists(model_path):
            raise RuntimeError(f"Piper model not found: {model_path}")

        config_path = settings.PIPER_CONFIG_PATH or ""
        piper_bin = settings.PIPER_BIN or "piper"

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_wav = tmp.name

        try:
            args = [piper_bin, "--model", model_path, "--output_file", out_wav, "--text", text]
            if config_path:
                args.extend(["--config", config_path])
            proc = subprocess.run(args, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                raise RuntimeError(f"Piper failed: {proc.stderr.strip() or proc.stdout.strip()}")
            with open(out_wav, "rb") as f:
                audio = f.read()
            if not audio:
                raise RuntimeError("Piper returned empty audio")
            logger.info("Piper synthesized %d chars â†’ %d bytes", len(text), len(audio))
            return audio, "audio/wav"
        finally:
            try:
                os.remove(out_wav)
            except Exception:
                pass

    def _synthesize_gtts(self, text: str) -> tuple[bytes, str]:
        try:
            from gtts import gTTS
        except Exception as exc:
            raise RuntimeError(f"gTTS import failed: {exc}") from exc

        mp3_io = io.BytesIO()
        tts = gTTS(text=text, lang=settings.GTTS_LANG or "en")
        tts.write_to_fp(mp3_io)
        audio = mp3_io.getvalue()
        if not audio:
            raise RuntimeError("gTTS returned empty audio")
        logger.info("gTTS synthesized %d chars → %d bytes", len(text), len(audio))
        return audio, "audio/mpeg"

    def _synthesize_coqui(self, text: str) -> tuple[bytes, str]:
        global _COQUI_MODEL
        try:
            from TTS.api import TTS
        except Exception as exc:
            raise RuntimeError(f"Coqui TTS import failed: {exc}") from exc

        with _COQUI_LOCK:
            if _COQUI_MODEL is None:
                _COQUI_MODEL = TTS(
                    model_name=settings.COQUI_TTS_MODEL,
                    progress_bar=False,
                    gpu=bool(settings.COQUI_TTS_USE_GPU),
                )

        # Coqui can return numpy or save to file; use in-memory if available
        try:
            wav = _COQUI_MODEL.tts(text)
            import numpy as np
            import wave

            if isinstance(wav, np.ndarray):
                wav = (wav * 32767).astype("int16")
                buf = io.BytesIO()
                with wave.open(buf, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(_COQUI_MODEL.synthesizer.output_sample_rate)
                    wf.writeframes(wav.tobytes())
                audio = buf.getvalue()
                if not audio:
                    raise RuntimeError("Coqui TTS returned empty audio")
                logger.info("Coqui TTS synthesized %d chars → %d bytes", len(text), len(audio))
                return audio, "audio/wav"
        except Exception as exc:
            raise RuntimeError(f"Coqui TTS failed: {exc}") from exc

    async def synthesize_and_upload(self, text: str) -> str:
        """
        Synthesize and store in memory cache. Returns public URL for Twilio Play.
        Falls back to text if no public base URL is configured.
        """
        try:
            audio_bytes, mime_type = await self.synthesize(text)
        except Exception as exc:
            logger.error("TTS synth failed: %s", exc)
            return f"__TEXT_FALLBACK__|{text}"

        base_url = settings.TWILIO_WEBHOOK_BASE_URL.rstrip("/")
        if not base_url:
            return f"__TEXT_FALLBACK__|{text}"

        tts_id = _cache_put(audio_bytes, mime_type, settings.TTS_AUDIO_TTL_SECONDS)
        prefix = settings.API_V1_PREFIX
        if not prefix.startswith("/"):
            prefix = "/" + prefix
        return f"{base_url}{prefix}/tts/{tts_id}"
