import json
import os
import uuid

from .config import DATA_FILE
from .policy import risk_to_priority
from .time_utils import utc_now_iso


def normalize_case_record(item: dict) -> dict:
    risk = item.get("risk") or item.get("risk_level", "LOW")
    timestamp = item.get("timestamp", utc_now_iso())
    case_id = item.get("id", str(uuid.uuid4())[:8])
    status = item.get("status", "OPEN")
    user_input = item.get("input", "")
    source = item.get("source", "unknown")
    recommended_action = item.get("recommended_action", "")
    priority = item.get("priority") or risk_to_priority(risk)

    return {
        "id": case_id,
        "timestamp": timestamp,
        "input": user_input,
        "risk": risk,
        "status": status,
        "source": source,
        "recommended_action": recommended_action,
        "priority": priority,
    }


def load_history() -> list:
    if not os.path.exists(DATA_FILE):
        return []

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if not isinstance(raw, list):
            return []

        return [normalize_case_record(item) for item in raw if isinstance(item, dict)]
    except Exception:
        return []


def save_history(history: list) -> None:
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
