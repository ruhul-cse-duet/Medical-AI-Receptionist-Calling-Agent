"""
Medical AI Receptionist — Main FastAPI Application
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.db.base import connect_db, close_db

from app.api.calls import router as calls_router
from app.api.appointments import router as appointments_router
from app.api.call_webhook import router as webhook_router
from app.api.tts import router as tts_router

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger("python_multipart").setLevel(logging.INFO)
logging.getLogger("pymongo").setLevel(logging.INFO)
logging.getLogger("urllib3").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.INFO)
logging.getLogger("LiteLLM").setLevel(logging.INFO)
logging.getLogger("httpcore").setLevel(logging.INFO)
logging.getLogger("openai").setLevel(logging.INFO)
logging.getLogger("gtts").setLevel(logging.INFO)
logging.getLogger("twilio.http_client").setLevel(logging.INFO)
logging.getLogger("crewai").setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s (env=%s)", settings.APP_NAME, settings.APP_ENV)
    await connect_db()
    logger.info("Application ready")
    yield
    await close_db()
    logger.info("Application shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version="1.0.0",
        description=(
            "AI-powered medical receptionist — handles inbound & outbound calls, "
            "books appointments in MongoDB, and sends reminder calls 5 hours before."
        ),
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url=None,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_HOSTS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        import uuid
        request.state.request_id = str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(Exception)
    async def global_exc(request: Request, exc: Exception):
        logger.error("Unhandled: %s %s → %s", request.method, request.url, exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"},
        )

    @app.get("/health", tags=["Health"])
    async def health():
        return {"status": "ok", "service": settings.APP_NAME}

    prefix = settings.API_V1_PREFIX
    app.include_router(calls_router,        prefix=prefix)
    app.include_router(appointments_router, prefix=prefix)
    app.include_router(webhook_router,      prefix=prefix)
    app.include_router(tts_router,          prefix=prefix)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)


# python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 192.168.0.194
