import re

HIGH_RISK_PATTERNS = [
    r"\bkill myself\b",
    r"\bend my life\b",
    r"\bsuicide\b",
    r"\bwant to die\b",
    r"\bdon'?t want to live\b",
    r"\bhurt myself\b",
    r"\bself[- ]?harm\b",
    r"\bno reason to live\b",
    r"\bi want to disappear forever\b",
]

MEDIUM_RISK_PATTERNS = [
    r"\bpanic\b",
    r"\banxious\b",
    r"\banxiety\b",
    r"\bdepressed\b",
    r"\boverwhelmed\b",
    r"\bcan'?t sleep\b",
    r"\bcannot sleep\b",
    r"\btrouble sleeping\b",
    r"\bhopeless\b",
    r"\bstressed\b",
    r"\bburned out\b",
    r"\bburnt out\b",
    r"\bexhausted\b",
    r"\balone\b",
]


def detect_signals(text: str) -> dict:
    text_lower = text.lower()
    high_matches = []
    medium_matches = []

    for pattern in HIGH_RISK_PATTERNS:
        if re.search(pattern, text_lower):
            high_matches.append(pattern)

    for pattern in MEDIUM_RISK_PATTERNS:
        if re.search(pattern, text_lower):
            medium_matches.append(pattern)

    if high_matches:
        risk = "HIGH"
        concerns = ["self-harm or suicide risk"]
    elif medium_matches:
        risk = "MEDIUM"
        concerns = ["emotional distress", "possible anxiety, sleep disruption, or depressive symptoms"]
    else:
        risk = "LOW"
        concerns = ["general emotional discomfort"]

    return {
        "risk": risk,
        "concerns": concerns,
        "high_matches": high_matches,
        "medium_matches": medium_matches,
        "confidence_note": "Rule-based fallback result.",
        "explanation": (
            "This result was produced by rule-based phrase matching because the AI service "
            "was unavailable, misconfigured, or returned an invalid response."
        ),
    }


def generate_response(risk: str) -> str:
    if risk == "HIGH":
        return (
            "I’m really sorry you’re going through this. You deserve immediate support right now. "
            "Please contact emergency services, a crisis hotline, or a trusted person immediately. "
            "If you might act on these thoughts, seek urgent help now."
        )
    if risk == "MEDIUM":
        return (
            "I’m sorry you’re dealing with this. It sounds like you may be going through significant distress. "
            "It may help to reach out to a trusted friend, counselor, or mental health professional. "
            "You could also try one small next step today, like taking a short walk, slowing your breathing, "
            "or writing down what you’re feeling."
        )
    return (
        "Thank you for sharing. It sounds like you may be having a difficult moment. "
        "Taking a short pause, doing a calming activity, or checking in with someone you trust may help."
    )


def recommended_action(risk: str) -> str:
    if risk == "HIGH":
        return "Escalate immediately to human review and crisis-support flow."
    if risk == "MEDIUM":
        return "Recommend human review and supportive follow-up if symptoms persist or worsen."
    return "Provide supportive resources and continue monitoring."


def build_fallback_result(text: str) -> dict:
    result = detect_signals(text)
    risk = result["risk"]
    return {
        "risk_level": risk,
        "confidence_note": result["confidence_note"],
        "detected_concerns": result["concerns"],
        "explanation": result["explanation"],
        "suggested_response": generate_response(risk),
        "recommended_action": recommended_action(risk),
        "safeguards_triggered": ["rule_based_fallback"],
        "matched_patterns": {
            "high_risk": result["high_matches"],
            "medium_risk": result["medium_matches"],
        },
        "source": "fallback_rules",
    }
