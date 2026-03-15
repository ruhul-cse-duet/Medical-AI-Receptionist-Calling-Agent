"""
Date/time utilities for international SaaS — region/country and timezone aware.
All stored times are UTC; display and parse using tenant's timezone and locale.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

# Safe default if tenant timezone is invalid
DEFAULT_TZ = "UTC"


def get_tenant_timezone(tenant: Optional[Any]) -> str:
    """Return IANA timezone string for tenant; fallback to DEFAULT_TZ."""
    if tenant is None or not getattr(tenant, "timezone", None):
        return DEFAULT_TZ
    tz = (tenant.timezone or "").strip()
    if not tz:
        return DEFAULT_TZ
    try:
        ZoneInfo(tz)
        return tz
    except Exception:
        return DEFAULT_TZ


def get_tenant_zoneinfo(tenant: Optional[Any]) -> ZoneInfo:
    """Return ZoneInfo for tenant's timezone."""
    return ZoneInfo(get_tenant_timezone(tenant))


def utc_to_tenant_local(dt_utc: datetime, tenant: Optional[Any]) -> datetime:
    """Convert UTC datetime to tenant's local time (naive in that TZ, or with tzinfo)."""
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    z = get_tenant_zoneinfo(tenant)
    return dt_utc.astimezone(z)


def format_datetime_tenant(
    dt_utc: datetime,
    tenant: Optional[Any],
    fmt: str = "%A, %B %d %Y at %I:%M %p",
) -> str:
    """
    Format a UTC datetime in the tenant's local timezone for display (e.g. confirmations, reminders).
    Uses tenant.timezone; no locale-based month names (keep English for simplicity unless we add babel).
    """
    if dt_utc is None:
        return "Unknown"
    if isinstance(dt_utc, str):
        dt_utc = datetime.fromisoformat(dt_utc.replace("Z", "+00:00"))
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    local = utc_to_tenant_local(dt_utc, tenant)
    return local.strftime(fmt)


def format_date_short_tenant(dt_utc: datetime, tenant: Optional[Any]) -> str:
    """Short format for inline display: e.g. 15 Mar 2026, 2:30 PM."""
    return format_datetime_tenant(dt_utc, tenant, fmt="%d %b %Y, %I:%M %p")


def now_in_tenant_tz(tenant: Optional[Any]) -> datetime:
    """Current time in tenant's timezone (for parsing 'today' / 'tomorrow')."""
    z = get_tenant_zoneinfo(tenant)
    return datetime.now(z)
