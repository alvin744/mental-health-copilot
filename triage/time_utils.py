from datetime import datetime, timezone


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_iso_datetime(value: str):
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value.replace("Z", "+00:00")
        return datetime.fromisoformat(value)
    except Exception:
        return None


def case_age_hours(timestamp: str) -> float:
    dt = parse_iso_datetime(timestamp)
    if not dt:
        return 0.0
    now = datetime.now(timezone.utc)
    return max((now - dt).total_seconds() / 3600, 0.0)
