import json
import re
import uuid

from .config import client
from .policy import apply_policy_adjustment, risk_to_priority
from .rules import build_fallback_result, generate_response, recommended_action
from .time_utils import utc_now_iso

SYSTEM_PROMPT = """
You are an AI safety assistant for a mental health triage demo.

Your job:
1. Assess the user's message for mental health risk in a cautious, safety-first way.
2. Return ONLY valid JSON.
3. Use one of these risk levels exactly: LOW, MEDIUM, HIGH.
4. Provide a supportive, non-diagnostic response.
5. If there are signs of self-harm, suicide intent, desire to die, or immediate danger, mark HIGH and recommend urgent escalation.
6. Do not provide a medical diagnosis.
7. Do not provide harmful instructions.
8. Keep suggested_response concise, supportive, and safe.
9. Include explainability suitable for demo purposes.
10. Be conservative in safety-sensitive contexts.

Risk level guidance:
- LOW: mild stress, frustration, temporary sadness, or general emotional discomfort without ongoing distress, sleep disruption, safety concerns, or strong impairment signals.
- MEDIUM: clear anxiety, panic, sleep disruption, depressed mood, hopelessness, overwhelm, repeated distress, or signs the person may benefit from human review or professional support, but no explicit self-harm intent.
- HIGH: any indication of self-harm, suicide intent, desire to die, harming others, or immediate danger.

Important calibration rule:
- If the message includes anxiety, panic, hopelessness, inability to sleep, ongoing distress, or similar symptoms, prefer MEDIUM over LOW.
- Do not classify anxiety plus sleep difficulty as LOW unless the message clearly indicates very mild and temporary discomfort.

Return JSON with this exact schema:
{
  "risk_level": "LOW or MEDIUM or HIGH",
  "confidence_note": "brief note",
  "detected_concerns": ["concern 1", "concern 2"],
  "explanation": "brief explanation of why this level was assigned",
  "suggested_response": "safe supportive response",
  "recommended_action": "clear operational recommendation",
  "safeguards_triggered": ["safeguard 1", "safeguard 2"]
}
"""


def parse_json_content(content: str) -> dict:
    content = content.strip()

    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?", "", content.strip())
        content = re.sub(r"```$", "", content.strip())
        content = content.strip()

    return json.loads(content)


def llm_triage(text: str) -> dict:
    if not client:
        return build_fallback_result(text)

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Assess this message:\n\n{text}"},
            ],
            temperature=0.2,
        )

        raw_text = response.output_text
        parsed = parse_json_content(raw_text)

        required_keys = [
            "risk_level",
            "confidence_note",
            "detected_concerns",
            "explanation",
            "suggested_response",
            "recommended_action",
            "safeguards_triggered",
        ]

        for key in required_keys:
            if key not in parsed:
                raise ValueError(f"Missing required key: {key}")

        if parsed["risk_level"] not in {"LOW", "MEDIUM", "HIGH"}:
            raise ValueError("Invalid risk_level returned by model.")

        parsed["source"] = "llm"
        return parsed

    except Exception as e:
        fallback = build_fallback_result(text)
        fallback["llm_error"] = str(e)
        return fallback


def apply_policy_to_result(result: dict, user_input: str, policy_mode: str) -> dict:
    original_risk = result["risk_level"]
    adjusted_risk, adjustment_reason = apply_policy_adjustment(
        original_risk,
        user_input,
        policy_mode,
    )

    policy_adjusted = adjusted_risk != original_risk

    if policy_adjusted:
        result["confidence_note"] = (
            f"{result['confidence_note']} Policy mode adjusted risk from {original_risk} to {adjusted_risk}."
        )
        result["recommended_action"] = recommended_action(adjusted_risk)
        result["suggested_response"] = generate_response(adjusted_risk)
        result["risk_level"] = adjusted_risk
        safeguards = result.get("safeguards_triggered", [])
        safeguards.append("policy_threshold_adjustment")
        result["safeguards_triggered"] = safeguards

    result["policy_adjustment_reason"] = adjustment_reason
    result["policy_adjusted"] = policy_adjusted
    return result


def create_case_record(user_input: str, result: dict, policy_mode: str) -> dict:
    timestamp = utc_now_iso()
    case_id = str(uuid.uuid4())[:8]
    risk = result["risk_level"]
    priority = risk_to_priority(risk)

    return {
        "timestamp": timestamp,
        "input": user_input,
        "result": result,
        "case_id": case_id,
        "priority": priority,
        "policy_mode": policy_mode,
        "case_record": {
            "id": case_id,
            "timestamp": timestamp,
            "input": user_input,
            "risk": risk,
            "status": "OPEN",
            "source": result.get("source", "unknown"),
            "recommended_action": result.get("recommended_action", ""),
            "priority": priority,
        },
    }
