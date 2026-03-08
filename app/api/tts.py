"""
TTS API â€” serve cached audio for Twilio Play
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from app.services.tts_service import get_cached_audio, TTSService

router = APIRouter(prefix="/tts", tags=["TTS"])


@router.get("/test", summary="Synthesize test audio without a call")
async def tts_test(
    text: str = Query(
        "Hello! This is a test of the receptionist voice.",
        min_length=1,
        max_length=500,
    )
):
    tts = TTSService()
    try:
        audio, mime_type = await tts.synthesize(text)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"TTS failed: {exc}")
    return Response(content=audio, media_type=mime_type)


@router.get("/{tts_id}", summary="Fetch cached TTS audio")
async def fetch_tts_audio(tts_id: str):
    cached = get_cached_audio(tts_id)
    if not cached:
        raise HTTPException(status_code=404, detail="Audio not found or expired")
    audio, mime_type = cached
    return Response(content=audio, media_type=mime_type)
