# Medical AI Receptionist (Inbound Only)

AI-powered phone receptionist for clinics using Twilio Voice webhooks and Media Streams.

This project is configured for inbound calls only:
- Caller dials your Twilio number
- Twilio hits `/v1/webhooks/call/answer`
- App opens the media stream websocket
- STT -> receptionist agent -> TTS response loop
- Optional appointment booking and CRUD in MongoDB

## Features
- Inbound call handling with Twilio Voice webhooks.
- Real-time audio via Twilio Media Streams websocket.
- Multi-tenant routing by Twilio `To` number.
- Conversational receptionist (CrewAI or simple backend).
- Deterministic appointment slot-filling (name, doctor, reason, date/time, confirmation).
- Appointment CRUD API.
- TTS test endpoint.

## Project Structure
- `app/main.py` - FastAPI application bootstrap.
- `app/api/call_webhook.py` - Twilio answer/status endpoints and media websocket.
- `app/agents/crew_receptionist.py` - main receptionist logic and booking flow.
- `app/agents/simple_receptionist.py` - fallback receptionist backend.
- `app/services/stt_service.py` - streaming STT.
- `app/services/tts_service.py` - TTS synthesis and audio serving.
- `app/services/twilio_service.py` - inbound TwiML and Twilio helpers.
- `app/api/appointments.py` - appointment CRUD.
- `app/api/calls.py` - call record retrieval.
- `app/api/tenants.py` - tenant registration and management.

## Requirements
- Python 3.11+
- MongoDB
- Twilio account + phone number
- Public HTTPS URL for Twilio webhooks (ngrok/cloudflared)

Optional for better audio decoding:
- `ffmpeg`
- `pydub`

## Setup
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set at least:
- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`
- `TWILIO_PHONE_NUMBER`
- `TWILIO_WEBHOOK_BASE_URL`
- `MONGODB_URL`
- `MONGODB_DB_NAME`
- `OPENAI_API_KEY` (if `LLM_PROVIDER=openai`)

## Run
```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Health/docs:
- `http://localhost:8000/health`
- `http://localhost:8000/docs`

## Docker Compose (Inbound Stack)
```bash
docker-compose up -d
```

This starts:
- `api`
- `mongo`
- `mongo-express` (only with `--profile dev`)

## Twilio Configuration
In Twilio Console for your phone number (Voice):

- A call comes in:
  - `https://<public-domain>/v1/webhooks/call/answer?call_id=NEW`
  - Method: `POST` (GET also supported)
- Status callback:
  - `https://<public-domain>/v1/webhooks/call/status?call_id=NEW`
  - Method: `POST`

## Inbound Call Flow
1. Twilio sends webhook to `/v1/webhooks/call/answer`.
2. App returns TwiML that starts websocket media stream.
3. Incoming audio is transcribed by STT.
4. Receptionist generates a reply.
5. Reply is synthesized with TTS and streamed back to caller.

## Multi-Tenant Inbound Routing
- Register tenants using `POST /v1/tenants/register`.
- Set tenant `twilio_phone_number` in E.164 format.
- When calls arrive, tenant is resolved from Twilio `To`.

## API Summary
Base prefix: `/v1`

Calls:
- `GET /calls/{call_id}`

Appointments:
- `POST /appointments/`
- `GET /appointments/`
- `GET /appointments/{appointment_id}`
- `PATCH /appointments/{appointment_id}`
- `DELETE /appointments/{appointment_id}`

Tenants:
- `POST /tenants/register`
- `GET /tenants/{tenant_id}`
- `PATCH /tenants/{tenant_id}`

Webhooks:
- `GET|POST /webhooks/call/answer`
- `POST /webhooks/call/status`
- `WS /webhooks/call/ws/stream/{call_id}`

TTS:
- `GET /tts/test`
- `GET /tts/{tts_id}`

## Quick Test
1. Start the API.
2. Expose it publicly (for example: `ngrok http 8000`).
3. Set `TWILIO_WEBHOOK_BASE_URL` to your public URL.
4. Configure Twilio voice webhook as above.
5. Call your Twilio number from your phone.

## Notes
- This repository is intentionally inbound-only.
- Worker/reminder outbound stack has been removed from compose/runtime path.
