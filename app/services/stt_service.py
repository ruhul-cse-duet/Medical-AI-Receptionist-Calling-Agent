"""
STT Service â€” Deepgram or Faster-Whisper streaming transcription
"""
from __future__ import annotations

import asyncio
import audioop
import io
import logging
import queue
import threading
import time
from typing import Awaitable, Callable, Optional
import wave

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)
OnTranscript = Callable[[str, bool], Awaitable[None]]


class _FasterWhisperStreamingSession:
    def __init__(
        self,
        model,
        on_transcript: OnTranscript,
        loop: asyncio.AbstractEventLoop,
        language: str,
        chunk_seconds: float,
        min_silence_ms: int,
    ) -> None:
        self._model = model
        self._on_transcript = on_transcript
        self._loop = loop
        self._language = language or "en"
        self._chunk_seconds = max(0.5, float(chunk_seconds))
        self._min_silence_ms = int(min_silence_ms)

        self._queue: "queue.Queue[bytes]" = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)

        self._rate_state = None
        self._processed_seconds = 0.0
        self._last_end_time = 0.0
        self._chunk_bytes = int(self._chunk_seconds * 16000 * 2)
        self._min_chunk_bytes = int(0.4 * 16000 * 2)

        self._thread.start()

    def send(self, data: bytes) -> None:
        if self._stop.is_set():
            return
        if data:
            self._queue.put(data)

    async def finish(self) -> None:
        self._stop.set()
        await asyncio.to_thread(self._thread.join, 2.0)

    def _worker(self) -> None:
        pcm_buffer = bytearray()
        while not self._stop.is_set() or not self._queue.empty():
            try:
                data = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            pcm = audioop.ulaw2lin(data, 2)
            pcm16k, self._rate_state = audioop.ratecv(
                pcm, 2, 1, 8000, 16000, self._rate_state
            )
            if pcm16k:
                pcm_buffer.extend(pcm16k)

            while len(pcm_buffer) >= self._chunk_bytes:
                chunk = bytes(pcm_buffer[: self._chunk_bytes])
                del pcm_buffer[: self._chunk_bytes]
                self._transcribe_chunk(chunk)

        if len(pcm_buffer) >= self._min_chunk_bytes:
            self._transcribe_chunk(bytes(pcm_buffer))

    def _transcribe_chunk(self, pcm_bytes: bytes) -> None:
        if not pcm_bytes:
            return
        try:
            audio_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)
            if audio_i16.size == 0:
                return
            audio_f32 = audio_i16.astype(np.float32) / 32768.0
            segments, _info = self._model.transcribe(
                audio_f32,
                language=self._language,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": self._min_silence_ms},
                word_timestamps=False,
            )

            new_parts = []
            for seg in segments:
                text = (seg.text or "").strip()
                if not text:
                    continue
                absolute_end = self._processed_seconds + (seg.end or 0.0)
                if absolute_end > self._last_end_time + 0.01:
                    new_parts.append(text)
                    self._last_end_time = max(self._last_end_time, absolute_end)

            if new_parts:
                final_text = " ".join(new_parts).strip()
                if final_text:
                    fut = asyncio.run_coroutine_threadsafe(
                        self._on_transcript(final_text, True), self._loop
                    )
                    fut.add_done_callback(self._log_future_error)
        except Exception as exc:
            logger.error("Faster-Whisper error: %s", exc)
        finally:
            self._processed_seconds += len(pcm_bytes) / (16000 * 2)

    @staticmethod
    def _log_future_error(fut: "asyncio.Future[None]") -> None:
        try:
            fut.result()
        except Exception as exc:
            logger.error("on_transcript callback failed: %s", exc)


class _NoopStreamingSession:
    def send(self, data: bytes) -> None:
        return

    async def finish(self) -> None:
        return


class _OpenAIWhisperSession:
    def __init__(
        self,
        on_transcript: OnTranscript,
        loop: asyncio.AbstractEventLoop,
        language: str,
    ) -> None:
        """
        Pseudo-streaming OpenAI Whisper session.

        We receive continuous 8kHz mu-law bytes from Twilio, buffer a few
        seconds of audio, then send that chunk to OpenAI Whisper as a WAV
        file. When a transcription comes back, we treat it as a final
        utterance and call on_transcript(text, True).

        This is not true low-latency streaming, but it gives near real-time
        turn-by-turn behavior with OpenAI STT.
        """
        self._on_transcript = on_transcript
        self._loop = loop
        # Normalize language to ISO-639-1 (e.g. "en-US" -> "en")
        raw_lang = language or getattr(settings, "OPENAI_STT_LANGUAGE", None) or "en"
        norm_lang = raw_lang.split("-")[0].split("_")[0].lower()
        self._language = norm_lang or "en"

        # Buffer of raw 8kHz mu-law bytes from Twilio
        self._buffer_ulaw = bytearray()
        self._rate_state = None

        # Chunking parameters
        self._chunk_seconds = max(1.5, float(getattr(settings, "STT_CHUNK_SECONDS", 1.0)))
        # At 8kHz, 1 byte/sample for mu-law
        self._chunk_ulaw_bytes = int(self._chunk_seconds * 8000)
        self._min_chunk_ulaw_bytes = int(0.8 * self._chunk_ulaw_bytes)

        self._queue: "queue.Queue[bytes]" = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)

        self._thread.start()

    def send(self, data: bytes) -> None:
        if self._stop.is_set():
            return
        if data:
            self._queue.put(data)

    async def finish(self) -> None:
        self._stop.set()
        # Wait briefly for worker to finish
        await asyncio.to_thread(self._thread.join, 2.0)
        # Final flush of remaining audio if big enough
        if len(self._buffer_ulaw) >= self._min_chunk_ulaw_bytes:
            await asyncio.to_thread(self._transcribe_blocking, bytes(self._buffer_ulaw))

    def _worker(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                data = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if data:
                self._buffer_ulaw.extend(data)

            # When we have enough buffered audio, send a chunk to Whisper
            while len(self._buffer_ulaw) >= self._chunk_ulaw_bytes:
                chunk = bytes(self._buffer_ulaw[: self._chunk_ulaw_bytes])
                del self._buffer_ulaw[: self._chunk_ulaw_bytes]
                try:
                    self._transcribe_blocking(chunk)
                except Exception as exc:
                    logger.error("OpenAI STT chunk error: %s", exc)

        # After stop signal, flush any remaining audio
        if len(self._buffer_ulaw) >= self._min_chunk_ulaw_bytes:
            try:
                self._transcribe_blocking(bytes(self._buffer_ulaw))
            except Exception as exc:
                logger.error("OpenAI STT final chunk error: %s", exc)

    def _transcribe_blocking(self, ulaw_bytes: bytes) -> None:
        try:
            from openai import OpenAI
        except Exception as exc:
            logger.error("OpenAI import failed: %s", exc)
            return

        if not settings.OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY is required for STT_PROVIDER=openai")
            return

        try:
            # Convert 8kHz mu-law → 16-bit PCM mono
            pcm = audioop.ulaw2lin(ulaw_bytes, 2)
            pcm16k, self._rate_state = audioop.ratecv(
                pcm, 2, 1, 8000, 16000, self._rate_state
            )
            if not pcm16k:
                return

            wav_io = io.BytesIO()
            with wave.open(wav_io, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(pcm16k)
            wav_io.seek(0)

            client = OpenAI(api_key=settings.OPENAI_API_KEY)
            # Let Whisper auto-detect language from audio so callers can speak
            # Bangla, English, or mixed Banglish naturally.
            resp = client.audio.transcriptions.create(
                model=settings.OPENAI_STT_MODEL,
                file=("audio.wav", wav_io, "audio/wav"),
            )
            text = (resp.text or "").strip()
            if text:
                fut = asyncio.run_coroutine_threadsafe(
                    self._on_transcript(text, True), self._loop
                )
                fut.add_done_callback(_FasterWhisperStreamingSession._log_future_error)
        except Exception as exc:
            logger.error("OpenAI STT error: %s", exc)


class STTService:
    def __init__(self) -> None:
        self.provider = settings.STT_PROVIDER
        self.dg = None
        self.model = None

        if self.provider == "none":
            return
        if self.provider == "deepgram":
            from deepgram import DeepgramClient
            self.dg = DeepgramClient(api_key=settings.DEEPGRAM_API_KEY)
        elif self.provider == "faster_whisper":
            from faster_whisper import WhisperModel
            self.model = WhisperModel(
                settings.FASTER_WHISPER_MODEL,
                device=settings.FASTER_WHISPER_DEVICE,
                compute_type=settings.FASTER_WHISPER_COMPUTE_TYPE,
            )
        elif self.provider == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is required for STT_PROVIDER=openai")
        else:
            raise ValueError(f"Unknown STT_PROVIDER: {self.provider}")

    async def start_streaming_session(
        self,
        on_transcript: OnTranscript,
        language: str = "en-US",
    ):
        """
        Opens a streaming session and returns a connection object
        with .send(bytes) and .finish() methods.
        """
        if self.provider == "none":
            logger.info("STT disabled (provider=none)")
            return _NoopStreamingSession()

        if self.provider == "openai":
            loop = asyncio.get_running_loop()
            logger.info("OpenAI STT started (model=%s)", settings.OPENAI_STT_MODEL)
            return _OpenAIWhisperSession(on_transcript, loop, language)

        if self.provider == "deepgram":
            from deepgram import LiveTranscriptionEvents, LiveOptions

            dg_connection = self.dg.listen.asynclive.v("1")

            async def on_message(self_inner, result, **kwargs):
                transcript = result.channel.alternatives[0].transcript
                is_final = result.is_final
                if transcript:
                    await on_transcript(transcript, is_final)

            dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

            options = LiveOptions(
                model=settings.DEEPGRAM_MODEL,
                language=language,
                smart_format=True,
                encoding="mulaw",
                channels=1,
                sample_rate=8000,
                interim_results=True,
                utterance_end_ms=1000,
                vad_events=True,
            )
            started = await dg_connection.start(options)
            if not started:
                raise RuntimeError("Deepgram STT connection failed to start")

            logger.info("Deepgram STT session started (lang=%s)", language)
            return dg_connection

        loop = asyncio.get_running_loop()
        stt_lang = settings.STT_LANGUAGE or language
        logger.info(
            "Faster-Whisper STT started (model=%s, lang=%s)",
            settings.FASTER_WHISPER_MODEL,
            stt_lang,
        )
        return _FasterWhisperStreamingSession(
            self.model,
            on_transcript,
            loop,
            stt_lang,
            settings.STT_CHUNK_SECONDS,
            settings.STT_MIN_SILENCE_MS,
        )
