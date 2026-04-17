from .time_utils import case_age_hours


def risk_to_priority(risk: str) -> str:
    mapping = {
        "HIGH": "P1",
        "MEDIUM": "P2",
        "LOW": "P3",
    }
    return mapping.get(risk, "P3")


def apply_policy_adjustment(base_risk: str, text: str, policy_mode: str):
    text_lower = text.lower().strip()
    reason = "No policy adjustment applied."

    medium_signal_words = [
        "anxious", "anxiety", "panic", "stressed", "stress", "overwhelmed",
        "can't sleep", "cant sleep", "cannot sleep", "trouble sleeping",
        "hopeless", "depressed", "burned out", "burnt out", "exhausted", "alone"
    ]

    high_signal_words = [
        "kill myself", "end my life", "suicide", "want to die",
        "don't want to live", "dont want to live", "hurt myself", "self harm"
    ]

    has_medium_signal = any(word in text_lower for word in medium_signal_words)
    has_high_signal = any(word in text_lower for word in high_signal_words)

    if policy_mode == "Balanced":
        return base_risk, reason

    if policy_mode == "Conservative":
        if base_risk == "LOW" and has_medium_signal:
            return "MEDIUM", "Conservative mode upshifted LOW to MEDIUM due to distress signals."
        if base_risk == "MEDIUM" and (
            "hopeless" in text_lower or "alone" in text_lower or "exhausted" in text_lower
        ):
            return "HIGH", "Conservative mode upshifted MEDIUM to HIGH due to stronger vulnerability indicators."
        return base_risk, reason

    if policy_mode == "Lenient":
        if base_risk == "HIGH" and not has_high_signal:
            return "MEDIUM", "Lenient mode downshifted HIGH to MEDIUM because no explicit crisis phrase was detected."
        if base_risk == "MEDIUM" and (
            "stressed" in text_lower or "stress" in text_lower
        ) and not any(
            word in text_lower for word in ["anxious", "anxiety", "panic", "hopeless", "depressed", "sleep"]
        ):
            return "LOW", "Lenient mode downshifted MEDIUM to LOW for mild stress-only signals."
        return base_risk, reason

    return base_risk, reason


def simulated_metrics(history: list, policy_mode: str) -> dict:
    if not history:
        return {
            "review_rate": 0.0,
            "escalation_rate": 0.0,
            "high_risk_share": 0.0,
            "estimated_review_load": 0,
        }

    adjusted_risks = []
    for item in history:
        base_risk = item.get("risk", "LOW")
        text = item.get("input", "")
        adjusted_risk, _ = apply_policy_adjustment(base_risk, text, policy_mode)
        adjusted_risks.append(adjusted_risk)

    total = len(adjusted_risks)
    review_cases = sum(1 for r in adjusted_risks if r in {"MEDIUM", "HIGH"})
    escalated_cases = sum(1 for r in adjusted_risks if r == "HIGH")
    high_share = escalated_cases / total if total else 0

    return {
        "review_rate": round((review_cases / total) * 100, 1) if total else 0.0,
        "escalation_rate": round((escalated_cases / total) * 100, 1) if total else 0.0,
        "high_risk_share": round(high_share * 100, 1) if total else 0.0,
        "estimated_review_load": review_cases,
    }


def sla_bucket(timestamp: str, status: str, risk: str) -> str:
    if status in {"CLOSED", "ESCALATED"}:
        return "Resolved / Routed"

    age = case_age_hours(timestamp)

    if risk == "HIGH":
        return "Overdue" if age > 1 else "Within SLA"
    if risk == "MEDIUM":
        return "Overdue" if age > 8 else "Within SLA"
    return "Overdue" if age > 24 else "Within SLA"
