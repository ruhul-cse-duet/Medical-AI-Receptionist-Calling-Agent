"""
Medical AI Receptionist — Configuration
Supports: OpenAI (cloud) | LM Studio (local)
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List, Literal


class Settings(BaseSettings):
    # ── App ───────────────────────────────────────────────────────────────────
    APP_NAME: str = "Medical AI Receptionist"
    APP_ENV: str = "development"
    DEBUG: bool = True
    API_V1_PREFIX: str = "/v1"
    ALLOWED_HOSTS: List[str] = ["*"]

    # Agent backend: "crewai" (default) | "simple" (no CrewAI, direct OpenAI/LM Studio)
    AGENT_BACKEND: Literal["crewai", "simple"] = "crewai"

    # ── LLM Provider: "openai" (cloud) | "lmstudio" (local) ──────────────────
    LLM_PROVIDER: Literal["openai", "lmstudio"] = "openai"

    # Speech Providers
    STT_PROVIDER: Literal["none", "deepgram", "faster_whisper", "openai"] = "faster_whisper"
    TTS_PROVIDER: Literal["none", "elevenlabs", "edge", "piper", "gtts", "coqui", "openai"] = "edge"

    # ── OpenAI (cloud) ────────────────────────────────────────────────────────
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o"
    OPENAI_STT_MODEL: str = "whisper-1"
    OPENAI_STT_LANGUAGE: str = "en"
    OPENAI_TTS_MODEL: str = "gpt-4o-mini-tts"
    OPENAI_TTS_VOICE: str = "shimmer"  # alloy, echo, fable, onyx, nova, shimmer, etc.

    # ── LM Studio (local) — OpenAI-compatible API ─────────────────────────────
    # LM Studio → Server tab → start → default port 1234
    LMSTUDIO_BASE_URL: str = "http://localhost:1234/v1"
    LMSTUDIO_API_KEY: str = "lm-studio"          # LM Studio doesn't need a real key
    LMSTUDIO_MODEL: str = "liquid/lfm2-1.2b"     # model loaded in LM Studio

    # ── MongoDB ───────────────────────────────────────────────────────────────
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "medical_receptionist"

    # ── Redis / Celery ────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"

    # ── Twilio ────────────────────────────────────────────────────────────────
    TWILIO_ACCOUNT_SID: str = ""
    TWILIO_AUTH_TOKEN: str = ""
    TWILIO_PHONE_NUMBER: str = ""
    TWILIO_WEBHOOK_BASE_URL: str = ""

    # ── Deepgram (STT) ────────────────────────────────────────────────────────
    DEEPGRAM_API_KEY: str = ""
    DEEPGRAM_MODEL: str = "nova-2"
    DEEPGRAM_LANGUAGE: str = "en-US"

    # Faster-Whisper (STT local)
    FASTER_WHISPER_MODEL: str = "base"
    FASTER_WHISPER_DEVICE: str = "cpu"
    FASTER_WHISPER_COMPUTE_TYPE: str = "int8"
    STT_LANGUAGE: str = "en"
    STT_CHUNK_SECONDS: float = 1.0
    STT_MIN_SILENCE_MS: int = 800

    # ── ElevenLabs (TTS) ──────────────────────────────────────────────────────
    ELEVENLABS_API_KEY: str = ""
    ELEVENLABS_VOICE_ID: str = "21m00Tcm4TlvDq8ikWAM"
    ELEVENLABS_MODEL_ID: str = "eleven_turbo_v2"

    # Edge TTS (local)
    EDGE_TTS_VOICE: str = "en-US-JennyNeural"
    EDGE_TTS_RATE: str = "+0%"
    EDGE_TTS_PITCH: str = "+0Hz"
    TTS_AUDIO_TTL_SECONDS: int = 600

    # Piper TTS (local CLI)
    PIPER_BIN: str = "piper"
    PIPER_MODEL_PATH: str = ""
    PIPER_CONFIG_PATH: str = ""

    # gTTS (Google TTS)
    GTTS_LANG: str = "en"

    # Coqui TTS (local)
    COQUI_TTS_MODEL: str = "tts_models/en/ljspeech/tacotron2-DDC"
    COQUI_TTS_USE_GPU: bool = False

    # ── Clinic Info ───────────────────────────────────────────────────────────
    CLINIC_NAME: str = "HealthCare Medical Center"
    CLINIC_PHONE: str = "+8801XXXXXXXXX"
    CLINIC_ADDRESS: str = "123 Medical Street, Dhaka, Bangladesh"
    CLINIC_HOURS: str = "Saturday to Thursday, 9 AM to 8 PM"

    # ── Reminder ──────────────────────────────────────────────────────────────
    REMINDER_HOURS_BEFORE: int = 5

    # ── Security ──────────────────────────────────────────────────────────────
    SECRET_KEY: str = "change-me-in-production-min-32-chars"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
