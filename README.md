# Medical AI Receptionist Calling Agent

AI-powered phone receptionist for a clinic.

It handles inbound/outbound calls, answers common questions, does slot-filling appointment booking (name, doctor, reason, date/time, confirmation), stores data in MongoDB, and triggers reminder calls via Celery.

## Features
- Inbound call handling through Twilio voice webhooks + Media Streams WebSocket.
- Outbound call API (`/v1/calls/outbound`).
- Conversational receptionist with local LM Studio or OpenAI.
- Deterministic slot-filling appointment booking flow with confirm-before-book.
- Appointment CRUD APIs.
- Reminder scheduler (`Celery Beat`) + worker tasks.
- STT providers: `faster_whisper`, `deepgram`, `openai`, `none`.
- TTS providers: `edge`, `gtts`, `elevenlabs`, `piper`, `coqui`, `none`.
- Full-duplex WS audio return path (preferred) with fallback call update path.

## Project Structure
- `app/main.py`: FastAPI app bootstrap.
- `app/config.py`: environment settings.
- `app/api/call_webhook.py`: Twilio webhook + WS call loop.
- `app/agents/crew_receptionist.py`: receptionist logic + booking state machine.
- `app/agents/tools.py`: doctor list + booking/info tools.
- `app/services/stt_service.py`: speech-to-text streaming.
- `app/services/tts_service.py`: text-to-speech + cache.
- `app/services/twilio_service.py`: Twilio helper functions.
- `app/api/appointments.py`: appointment CRUD.
- `app/api/calls.py`: outbound call API.
- `worker/tasks.py`: reminder and outbound Celery tasks.

## Requirements
- Python 3.11+
- MongoDB
- Redis
- Twilio account + phone number
- Public HTTPS URL for webhooks (ngrok/cloudflared)
- Optional but recommended for full-duplex MP3 decode:
  - `ffmpeg` (system binary)
  - `pydub` (already in requirements)

## Environment Setup
1. Create and activate virtual env.
2. Install dependencies.
3. Copy env template and fill values.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
Copy-Item .env.example .env
```

Important `.env` keys:
- `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_PHONE_NUMBER`
- `TWILIO_WEBHOOK_BASE_URL=https://<public-domain>`
- `MONGODB_URL`, `MONGODB_DB_NAME`
- `REDIS_URL`, `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND`
- `LLM_PROVIDER=openai|lmstudio`
- `LMSTUDIO_BASE_URL`, `LMSTUDIO_MODEL` (if local LLM)
- `STT_PROVIDER`, `TTS_PROVIDER`
- `CLINIC_NAME`, `CLINIC_PHONE`, `CLINIC_ADDRESS`, `CLINIC_HOURS`

## Local LLM (LM Studio)
If you use local model:
1. Open LM Studio.
2. Download/load your model (default: `liquid/lfm2-1.2b`).
3. Start local server (default `http://localhost:1234/v1`).
4. Set in `.env`:
   - `LLM_PROVIDER=lmstudio`
   - `LMSTUDIO_BASE_URL=http://localhost:1234/v1`
   - `LMSTUDIO_MODEL=<loaded-model-id>`

Note: smaller local models can be slower or less accurate in phone conversation than cloud models.

## FFmpeg Setup (Windows)
Recommended for full-duplex audio reliability.

```powershell
winget install --id Gyan.FFmpeg -e
ffmpeg -version
ffprobe -version
where ffmpeg
```

## Run (Local, 3 terminals)
Terminal 1 - API:
```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Terminal 2 - Celery worker:
```powershell
celery -A worker.tasks worker --loglevel=info --concurrency=4
```

Terminal 3 - Celery beat:
```powershell
celery -A worker.tasks beat --loglevel=info
```

Health/docs:
- `http://localhost:8000/health`
- `http://localhost:8000/docs`

## Run with Docker Compose
```bash
docker-compose up -d
```

Optional dev profile with Mongo Express:
```bash
docker-compose --profile dev up -d
```

## Expose Local Server for Twilio
Example with ngrok:
```powershell
ngrok http 8000
```
Set `.env`:
- `TWILIO_WEBHOOK_BASE_URL=https://<ngrok-domain>`

## Twilio Webhook Configuration
In Twilio Console (Phone Number -> Voice):
- A call comes in:
  - `https://<public-domain>/v1/webhooks/call/answer?call_id=NEW`
  - Method: `POST` (GET also supported)
- Status callback:
  - `https://<public-domain>/v1/webhooks/call/status?call_id=NEW`
  - Method: `POST`

## How Conversation Works
Inbound call flow:
1. Twilio hits `/v1/webhooks/call/answer`.
2. App returns TwiML with `<Connect><Stream>` to `/v1/webhooks/call/ws/stream/{call_id}`.
3. WS receives caller audio (`media` events).
4. STT converts speech -> text.
5. `crew_receptionist` processes text:
   - Fast FAQ answers for common intents.
   - Explicit booking state machine for appointments.
   - LLM fallback for general conversation.
6. TTS generates response audio.
7. Audio is sent back over the same WS (full-duplex). If needed, fallback path updates call TwiML.

## Booking Conversation (Slot-Filling)
Required slots:
- Patient name
- Patient phone
- Preferred doctor/specialty
- Reason/treatment
- Preferred date/time

Behavior:
- Asks only missing slot(s), one step at a time.
- Summarizes all details.
- Asks explicit confirmation.
- Books appointment only after user confirms.

## API Summary
Base prefix: `/v1`

Calls:
- `POST /calls/outbound`
- `GET /calls/{call_id}`

Appointments:
- `POST /appointments/`
- `GET /appointments/`
- `GET /appointments/{appointment_id}`
- `PATCH /appointments/{appointment_id}`
- `DELETE /appointments/{appointment_id}`

Twilio webhooks:
- `GET|POST /webhooks/call/answer`
- `POST /webhooks/call/reminder`
- `POST /webhooks/call/status`
- `WS /webhooks/call/ws/stream/{call_id}`

TTS helper:
- `GET /tts/test`
- `GET /tts/{tts_id}`

## Test Quickly
### 1) Outbound test call
```bash
curl -X POST http://localhost:8000/v1/calls/outbound \
  -H "Content-Type: application/json" \
  -d '{"patient_phone":"+8801XXXXXXXXX","patient_name":"Test User"}'
```

### 2) TTS test
Open:
- `http://localhost:8000/v1/tts/test?text=Hello%20from%20medical%20receptionist`

### 3) Appointment APIs
Use `/docs` to create, list, update, and cancel appointments.

## Common Issues and Fixes
1. Agent receives call but says nothing.
- Check `TWILIO_WEBHOOK_BASE_URL` is public HTTPS.
- Check WS route is reachable from Twilio.
- Install `ffmpeg` and verify with `ffmpeg -version`.

2. Response is slow.
- Local LLM/STT/TTS on CPU is slower.
- Use faster model, reduce STT model size, or switch to OpenAI/Deepgram/ElevenLabs.

3. Twilio error `21220 Call is not in-progress`.
- Usually stale `CallSid`/timing issue; keep call/session IDs unique and ensure active stream session.

4. No reminder calls.
- Ensure worker + beat are both running.
- Verify Redis connection and `REMINDER_HOURS_BEFORE`.

## Security Notes
- Do not commit `.env`.
- Use real webhook signature validation in production (`APP_ENV != development`).
- Rotate Twilio/OpenAI keys if leaked.

## License
Internal project / add your preferred license.
