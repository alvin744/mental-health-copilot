from html import escape

import streamlit as st

st.set_page_config(
    page_title="AI Safety Triage Console",
    page_icon="📋",
    layout="wide",
)

from triage.config import (
    CASE_STATUSES,
    OPENAI_API_KEY,
    POLICY_MODES,
    REVIEWER_DECISIONS,
    client,
)
from triage.llm import apply_policy_to_result, create_case_record, llm_triage
from triage.persistence import load_history, save_history
from triage.policy import risk_to_priority, simulated_metrics, sla_bucket
from triage.time_utils import case_age_hours
from ui.components import (
    inject_styles,
    render_pills,
    render_risk_banner,
    soft_box_end,
    soft_box_start,
    status_class_name,
)

inject_styles()

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
        POLICY_MODES,
        index=POLICY_MODES.index(st.session_state.policy_mode),
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

        result = apply_policy_to_result(result, user_input, st.session_state.policy_mode)
        case_bundle = create_case_record(user_input, result, st.session_state.policy_mode)

        st.session_state.latest_result = {
            key: case_bundle[key]
            for key in ["timestamp", "input", "result", "case_id", "priority", "policy_mode"]
        }
        st.session_state.triage_history.append(case_bundle["case_record"])
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
            REVIEWER_DECISIONS,
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

filter_col = st.columns([1, 3])[0]
with filter_col:
    status_filter = st.selectbox(
        "Filter by status",
        ["ALL"] + CASE_STATUSES,
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
                <div class="queue-title">{escape(risk)} · {escape(priority)} · {escape(source)}</div>
                <div class="queue-subtext">{escape(user_text[:140])}</div>
                <div class="queue-note">Case ID: {escape(case_id)} · {escape(timestamp)}</div>
                <div class="queue-note">Age: {age_hours}h · SLA: {escape(sla_status)}</div>
                <div class="queue-note">{escape(recommended)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        queue_col1, queue_col2 = st.columns([1, 3])
        with queue_col1:
            new_status = st.selectbox(
                f"Status for case {case_id}",
                CASE_STATUSES,
                index=CASE_STATUSES.index(status),
                key=f"status_{case_id}",
            )
        with queue_col2:
            st.markdown(
                f'<span class="{status_class_name(new_status)}">{escape(new_status)}</span> · Priority {escape(priority)} · {escape(risk)} case',
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
