import json
import os
import re
import uuid
from datetime import datetime, timezone

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------
# Setup
# -----------------------------
load_dotenv()

st.set_page_config(
    page_title="AI Safety Triage Console",
    page_icon="📋",
    layout="wide",
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

DATA_FILE = "triage_data.json"

# -----------------------------
# Time helpers
# -----------------------------
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


# -----------------------------
# Policy / priority helpers
# -----------------------------
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


# -----------------------------
# Persistence helpers
# -----------------------------
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


# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
        .app-subtitle {
            opacity: 0.82;
            margin-bottom: 0.85rem;
        }

        .soft-box {
            padding: 0.95rem 1rem;
            border-radius: 12px;
            border: 1px solid rgba(128, 128, 128, 0.22);
            background: rgba(127, 127, 127, 0.04);
            margin-bottom: 0.9rem;
        }

        .risk-banner {
            padding: 1rem 1.1rem;
            border-radius: 14px;
            margin-bottom: 1rem;
            border: 1px solid rgba(128, 128, 128, 0.25);
        }

        .risk-low {
            background: rgba(34, 197, 94, 0.10);
            border-left: 6px solid #22c55e;
        }

        .risk-medium {
            background: rgba(245, 158, 11, 0.14);
            border-left: 6px solid #f59e0b;
        }

        .risk-high {
            background: rgba(239, 68, 68, 0.12);
            border-left: 6px solid #ef4444;
        }

        .label-text {
            font-size: 0.85rem;
            opacity: 0.75;
            margin-bottom: 0.2rem;
        }

        .value-text {
            font-size: 1.4rem;
            font-weight: 700;
            margin-bottom: 0.8rem;
        }

        .pill-wrap {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.25rem;
        }

        .pill {
            display: inline-flex;
            align-items: center;
            padding: 0.32rem 0.68rem;
            border-radius: 999px;
            border: 1px solid rgba(128, 128, 128, 0.25);
            background: rgba(127, 127, 127, 0.08);
            font-size: 0.88rem;
            line-height: 1.2;
            white-space: nowrap;
        }

        .queue-card {
            padding: 0.85rem 0.95rem;
            border-radius: 12px;
            border: 1px solid rgba(128, 128, 128, 0.22);
            background: rgba(127, 127, 127, 0.04);
            margin-bottom: 0.65rem;
        }

        .queue-title {
            font-weight: 600;
            margin-bottom: 0.15rem;
        }

        .queue-subtext {
            opacity: 0.84;
            font-size: 0.93rem;
            margin-bottom: 0.2rem;
        }

        .queue-note {
            opacity: 0.72;
            font-size: 0.88rem;
            margin-bottom: 0.15rem;
        }

        .status-open {
            color: #f59e0b;
            font-weight: 600;
        }

        .status-in-review {
            color: #3b82f6;
            font-weight: 600;
        }

        .status-escalated {
            color: #ef4444;
            font-weight: 600;
        }

        .status-closed {
            color: #22c55e;
            font-weight: 600;
        }

        .kpi-caption {
            opacity: 0.75;
            font-size: 0.9rem;
            margin-top: -0.35rem;
            margin-bottom: 0.8rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📋 AI Safety Triage Console")
st.markdown(
    '<div class="app-subtitle">Demo MVP for safe, AI-assisted triage with human review, case management, and policy simulation.</div>',
    unsafe_allow_html=True,
)

st.warning(
    "This is a demo prototype and not a substitute for professional medical or mental health care. "
    "If someone is in immediate danger or may harm themselves or others, contact local emergency services immediately."
)

# -----------------------------
# UI helpers
# -----------------------------
def risk_class_name(risk: str) -> str:
    mapping = {
        "LOW": "risk-banner risk-low",
        "MEDIUM": "risk-banner risk-medium",
        "HIGH": "risk-banner risk-high",
    }
    return mapping.get(risk, "risk-banner")


def status_class_name(status: str) -> str:
    mapping = {
        "OPEN": "status-open",
        "IN REVIEW": "status-in-review",
        "ESCALATED": "status-escalated",
        "CLOSED": "status-closed",
    }
    return mapping.get(status, "")


def render_risk_banner(risk: str, confidence_note: str, recommended_action: str, priority: str) -> None:
    css_class = risk_class_name(risk)
    st.markdown(
        f"""
        <div class="{css_class}">
            <div class="label-text">Risk Level</div>
            <div class="value-text">{risk} · {priority}</div>
            <div class="label-text">Confidence Note</div>
            <div>{confidence_note}</div>
            <br>
            <div class="label-text">Recommended Action</div>
            <div>{recommended_action}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pills(items) -> None:
    if not items:
        st.write("None")
        return

    safe_items = [str(item) for item in items if str(item).strip()]
    if not safe_items:
        st.write("None")
        return

    html = "".join([f'<span class="pill">{item}</span>' for item in safe_items])
    st.markdown(f'<div class="pill-wrap">{html}</div>', unsafe_allow_html=True)


def soft_box_start() -> None:
    st.markdown('<div class="soft-box">', unsafe_allow_html=True)


def soft_box_end() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Rule-based fallback
# -----------------------------
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


# -----------------------------
# LLM-assisted triage
# -----------------------------
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


# -----------------------------
# Session state
# -----------------------------
if "triage_history" not in st.session_state:
    st.session_state.triage_history = load_history()

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

if "current_input" not in st.session_state:
    st.session_state.current_input = ""

if "policy_mode" not in st.session_state:
    st.session_state.policy_mode = "Balanced"

sample_cases = {
    "Select a sample case...": "",
    "Low distress": "I've been stressed about work this week and need a break.",
    "Medium distress": "I feel anxious and cant sleep lately.",
    "High risk": "I don't want to live anymore and I feel like hurting myself.",
}

if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = "Select a sample case..."


def on_sample_change():
    selected = st.session_state.selected_sample
    if selected in sample_cases and selected != "Select a sample case...":
        st.session_state.current_input = sample_cases[selected]


# -----------------------------
# Controls
# -----------------------------
with st.expander("System Status"):
    st.write("API key loaded:", bool(OPENAI_API_KEY))
    st.write("LLM client ready:", client is not None)

control_left, control_mid, control_right = st.columns([3.2, 1.2, 1.2])

with control_mid:
    policy_mode = st.selectbox(
        "Policy mode",
        ["Conservative", "Balanced", "Lenient"],
        index=["Conservative", "Balanced", "Lenient"].index(st.session_state.policy_mode),
        key="policy_mode",
    )

with control_right:
    st.selectbox(
        "Load sample",
        list(sample_cases.keys()),
        key="selected_sample",
        on_change=on_sample_change,
    )

with control_left:
    user_input = st.text_area(
        "Enter a user message",
        key="current_input",
        height=180,
        placeholder="Example: I feel anxious and can't sleep lately.",
    )

action_col1, action_col2 = st.columns([1, 1])
with action_col1:
    analyze_clicked = st.button("Analyze", type="primary")
with action_col2:
    clear_history_clicked = st.button("Clear history")

if clear_history_clicked:
    st.session_state.triage_history = []
    st.session_state.latest_result = None
    save_history([])

# -----------------------------
# Policy simulation panel
# -----------------------------
soft_box_start()
st.subheader("Policy Simulation")
st.write(
    "Simulate how policy strictness affects review workload and escalation volume. "
    "This is a simplified demo for product and operations trade-off discussions."
)

simulation = simulated_metrics(st.session_state.triage_history, st.session_state.policy_mode)
sim1, sim2, sim3, sim4 = st.columns(4)
sim1.metric("Review Rate", f"{simulation['review_rate']}%")
sim2.metric("Escalation Rate", f"{simulation['escalation_rate']}%")
sim3.metric("High-Risk Share", f"{simulation['high_risk_share']}%")
sim4.metric("Est. Review Load", simulation["estimated_review_load"])

policy_note = {
    "Conservative": "Conservative mode prioritizes safety sensitivity and tends to increase review volume.",
    "Balanced": "Balanced mode aims for a practical trade-off between safety coverage and reviewer workload.",
    "Lenient": "Lenient mode reduces review load but may under-classify borderline distress cases.",
}
st.caption(policy_note[st.session_state.policy_mode])
soft_box_end()

# -----------------------------
# Main analysis
# -----------------------------
if analyze_clicked:
    if not user_input.strip():
        st.error("Please enter a message.")
    else:
        with st.spinner("Analyzing message..."):
            result = llm_triage(user_input)

        original_risk = result["risk_level"]
        adjusted_risk, adjustment_reason = apply_policy_adjustment(
            original_risk,
            user_input,
            st.session_state.policy_mode,
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

        timestamp = utc_now_iso()
        case_id = str(uuid.uuid4())[:8]
        risk = result["risk_level"]
        priority = risk_to_priority(risk)

        st.session_state.latest_result = {
            "timestamp": timestamp,
            "input": user_input,
            "result": result,
            "case_id": case_id,
            "priority": priority,
            "policy_mode": st.session_state.policy_mode,
        }

        new_case = {
            "id": case_id,
            "timestamp": timestamp,
            "input": user_input,
            "risk": risk,
            "status": "OPEN",
            "source": result.get("source", "unknown"),
            "recommended_action": result.get("recommended_action", ""),
            "priority": priority,
        }

        st.session_state.triage_history.append(new_case)
        save_history(st.session_state.triage_history)

latest = st.session_state.latest_result

if latest:
    result = latest["result"]

    left, right = st.columns([1.35, 1])

    with left:
        render_risk_banner(
            result["risk_level"],
            result["confidence_note"],
            result["recommended_action"],
            latest["priority"],
        )

        soft_box_start()
        st.subheader("Suggested Support Response")
        st.write(result["suggested_response"])
        soft_box_end()

        soft_box_start()
        st.subheader("Explanation")
        st.write(result["explanation"])
        soft_box_end()

        if result.get("policy_adjusted"):
            soft_box_start()
            st.subheader("Policy Adjustment")
            st.write(f"**Mode:** {latest['policy_mode']}")
            st.write(result["policy_adjustment_reason"])
            soft_box_end()

    with right:
        soft_box_start()
        st.subheader("Case Info")
        st.write("**Case ID:**", latest["case_id"])
        st.write("**Timestamp:**", latest["timestamp"])
        st.write("**Priority:**", latest["priority"])
        st.write("**Policy Mode:**", latest["policy_mode"])
        soft_box_end()

        soft_box_start()
        st.subheader("Detected Concerns")
        render_pills(result.get("detected_concerns", []))
        soft_box_end()

        soft_box_start()
        st.subheader("Safeguards Triggered")
        render_pills(result.get("safeguards_triggered", []))
        soft_box_end()

        st.subheader("Reviewer Action")
        st.selectbox(
            "Reviewer decision",
            ["Approve system recommendation", "Escalate", "Modify response", "Defer"],
            key="reviewer_decision",
        )
        st.text_area(
            "Reviewer notes",
            placeholder="Add rationale, escalation notes, or follow-up actions...",
            key="reviewer_notes",
            height=120,
        )

    if result.get("source") == "fallback_rules":
        st.info("AI service unavailable or invalid response received. Showing rule-based fallback result.")

    if "llm_error" in result:
        st.error(f"LLM error: {result['llm_error']}")
        with st.expander("LLM error details"):
            st.code(result["llm_error"])

    if "matched_patterns" in result:
        with st.expander("Rule-based match details"):
            st.json(result["matched_patterns"])

    st.subheader("Audit Log Preview")
    st.json(
        {
            "case_id": latest["case_id"],
            "timestamp": latest["timestamp"],
            "input": latest["input"],
            "risk_level": result["risk_level"],
            "priority": latest["priority"],
            "policy_mode": latest["policy_mode"],
            "recommended_action": result["recommended_action"],
            "reviewer_decision": st.session_state.get("reviewer_decision", ""),
            "reviewer_notes": st.session_state.get("reviewer_notes", ""),
            "source": result.get("source", "unknown"),
        }
    )

# -----------------------------
# Dashboard / Case queue
# -----------------------------
st.divider()
st.subheader("Operational Dashboard")

history = st.session_state.triage_history

filter_col1, filter_col2 = st.columns([1, 3])
with filter_col1:
    status_filter = st.selectbox(
        "Filter by status",
        ["ALL", "OPEN", "IN REVIEW", "ESCALATED", "CLOSED"],
    )

if history:
    for item in history:
        if "priority" not in item or not item["priority"]:
            item["priority"] = risk_to_priority(item.get("risk", "LOW"))

    low_count = sum(1 for item in history if item.get("risk", "LOW") == "LOW")
    medium_count = sum(1 for item in history if item.get("risk", "LOW") == "MEDIUM")
    high_count = sum(1 for item in history if item.get("risk", "LOW") == "HIGH")

    open_count = sum(1 for item in history if item.get("status", "OPEN") in {"OPEN", "IN REVIEW"})
    escalated_count = sum(1 for item in history if item.get("status", "OPEN") == "ESCALATED")
    escalation_rate = round((escalated_count / len(history)) * 100, 1) if history else 0.0

    overdue_count = sum(
        1
        for item in history
        if sla_bucket(item.get("timestamp", ""), item.get("status", "OPEN"), item.get("risk", "LOW")) == "Overdue"
    )

    avg_age_open = [
        case_age_hours(item.get("timestamp", ""))
        for item in history
        if item.get("status", "OPEN") in {"OPEN", "IN REVIEW"}
    ]
    avg_age_open_hours = round(sum(avg_age_open) / len(avg_age_open), 1) if avg_age_open else 0.0

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Open Backlog", open_count)
    d2.metric("Escalation Rate", f"{escalation_rate}%")
    d3.metric("Overdue Cases", overdue_count)
    d4.metric("Avg Open Age", f"{avg_age_open_hours}h")

    st.markdown(
        '<div class="kpi-caption">Backlog and SLA-style metrics for demo purposes only.</div>',
        unsafe_allow_html=True,
    )

    risk1, risk2, risk3, risk4 = st.columns(4)
    risk1.metric("LOW", low_count)
    risk2.metric("MEDIUM", medium_count)
    risk3.metric("HIGH", high_count)
    risk4.metric("Total Cases", len(history))

    visible_cases = []
    for item in history:
        case_status = item.get("status", "OPEN")
        if status_filter == "ALL" or case_status == status_filter:
            visible_cases.append(item)

    st.subheader("Case Queue")

    updated = False

    for item in visible_cases[::-1][:10]:
        case_id = item.get("id", "unknown")
        risk = item.get("risk", item.get("risk_level", "LOW"))
        priority = item.get("priority", risk_to_priority(risk))
        user_text = item.get("input", "")
        source = item.get("source", "unknown")
        recommended = item.get("recommended_action", "")
        timestamp = item.get("timestamp", "")
        status = item.get("status", "OPEN")
        age_hours = round(case_age_hours(timestamp), 1)
        sla_status = sla_bucket(timestamp, status, risk)

        st.markdown(
            f"""
            <div class="queue-card">
                <div class="queue-title">{risk} · {priority} · {source}</div>
                <div class="queue-subtext">{user_text[:140]}</div>
                <div class="queue-note">Case ID: {case_id} · {timestamp}</div>
                <div class="queue-note">Age: {age_hours}h · SLA: {sla_status}</div>
                <div class="queue-note">{recommended}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        queue_col1, queue_col2 = st.columns([1, 3])
        with queue_col1:
            new_status = st.selectbox(
                f"Status for case {case_id}",
                ["OPEN", "IN REVIEW", "ESCALATED", "CLOSED"],
                index=["OPEN", "IN REVIEW", "ESCALATED", "CLOSED"].index(status),
                key=f"status_{case_id}",
            )
        with queue_col2:
            st.markdown(
                f'<span class="{status_class_name(new_status)}">{new_status}</span> · Priority {priority} · {risk} case',
                unsafe_allow_html=True,
            )

        if new_status != status:
            item["status"] = new_status
            updated = True

    if updated:
        save_history(history)

    with st.expander("Recent Analyses (raw)"):
        st.json(history[-10:])
else:
    st.caption("No cases yet.")