"""
MongoDB async client using Motor
"""
from __future__ import annotations
import logging
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from app.config import settings

logger = logging.getLogger(__name__)

_client: AsyncIOMotorClient | None = None


async def connect_db() -> None:
    global _client
    _client = AsyncIOMotorClient(settings.MONGODB_URL)
    logger.info("MongoDB connected: %s / %s", settings.MONGODB_URL, settings.MONGODB_DB_NAME)


async def close_db() -> None:
    global _client
    if _client:
        _client.close()
        logger.info("MongoDB connection closed")


def get_db() -> AsyncIOMotorDatabase:
    if _client is None:
        raise RuntimeError("MongoDB not connected. Call connect_db() first.")
    return _client[settings.MONGODB_DB_NAME]


# Collection helpers
def tenants_col():
    return get_db()["tenants"]


def users_col():
    return get_db()["users"]


def patients_col():
    return get_db()["patients"]


def appointments_col():
    return get_db()["appointments"]


def calls_col():
    return get_db()["calls"]
