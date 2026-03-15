"""
Tenant resolution for SaaS — load tenant by ID or by Twilio phone number.
Used by webhooks and agents to get company-specific config (name, hours, agent behavior).
"""
from __future__ import annotations

import contextvars
import logging
from typing import Optional

from app.db.base import tenants_col
from app.db.models import Tenant

logger = logging.getLogger(__name__)

# Request-scoped current tenant (set in webhook/API, read by tools and agents)
_current_tenant: contextvars.ContextVar[Optional[Tenant]] = contextvars.ContextVar(
    "current_tenant", default=None
)


def set_current_tenant(tenant: Optional[Tenant]) -> None:
    _current_tenant.set(tenant)


def get_current_tenant() -> Optional[Tenant]:
    return _current_tenant.get()


async def get_tenant_by_id(tenant_id: str) -> Optional[Tenant]:
    """Load tenant by ID."""
    if not tenant_id:
        return None
    doc = await tenants_col().find_one({"id": tenant_id, "is_active": True})
    if not doc:
        return None
    doc.pop("_id", None)
    return Tenant.model_validate(doc)


def _normalize_phone(value: str) -> dict:
    raw = (value or "").strip()
    digits = "".join(ch for ch in raw if ch.isdigit())
    e164 = ""
    if raw.startswith("+") and digits:
        e164 = "+" + digits
    elif digits.startswith("88"):
        e164 = "+" + digits
    elif digits:
        e164 = "+" + digits
    last10 = digits[-10:] if len(digits) >= 10 else digits
    return {"raw": raw, "digits": digits, "e164": e164, "last10": last10}


async def _find_tenant_by_phone_fallback(normalized: dict) -> Optional[Tenant]:
    """
    Fallback: scan active tenants with phone numbers and match by digits/last10.
    This is used when exact E.164 match fails (e.g., formatting differences).
    """
    if not normalized.get("digits"):
        return None
    cursor = tenants_col().find(
        {"is_active": True, "twilio_phone_number": {"$ne": ""}},
        {"_id": 0},
    )
    async for doc in cursor:
        candidate = _normalize_phone(doc.get("twilio_phone_number", ""))
        if not candidate.get("digits"):
            continue
        if normalized["digits"] == candidate["digits"] or normalized["last10"] == candidate["last10"]:
            return Tenant.model_validate(doc)

    # Optional fallback: match against main clinic phone if Twilio number not set
    cursor = tenants_col().find(
        {"is_active": True, "phone": {"$ne": ""}},
        {"_id": 0},
    )
    async for doc in cursor:
        candidate = _normalize_phone(doc.get("phone", ""))
        if not candidate.get("digits"):
            continue
        if normalized["digits"] == candidate["digits"] or normalized["last10"] == candidate["last10"]:
            return Tenant.model_validate(doc)
    return None


async def get_tenant_by_twilio_phone(to_number: str) -> Optional[Tenant]:
    """
    Resolve tenant by the Twilio number that was called (inbound).
    Normalize for comparison: strip spaces, use E.164 if needed.
    """
    if not to_number:
        return None
    normalized = _normalize_phone(to_number)
    e164 = normalized.get("e164", "")
    if e164:
        doc = await tenants_col().find_one(
            {"twilio_phone_number": e164, "is_active": True}
        )
        if doc:
            doc.pop("_id", None)
            return Tenant.model_validate(doc)

    fallback = await _find_tenant_by_phone_fallback(normalized)
    if fallback:
        return fallback
    return None


async def get_tenant_for_call(call_id: str) -> Optional[Tenant]:
    """Load tenant from call document (call has tenant_id)."""
    from app.db.base import calls_col
    call = await calls_col().find_one({"id": call_id})
    if not call or not call.get("tenant_id"):
        return None
    return await get_tenant_by_id(call["tenant_id"])
