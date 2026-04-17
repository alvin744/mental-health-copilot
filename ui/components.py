from html import escape

import streamlit as st

STYLES = """
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
"""


def inject_styles() -> None:
    st.markdown(STYLES, unsafe_allow_html=True)


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
            <div class="value-text">{escape(risk)} · {escape(priority)}</div>
            <div class="label-text">Confidence Note</div>
            <div>{escape(confidence_note)}</div>
            <br>
            <div class="label-text">Recommended Action</div>
            <div>{escape(recommended_action)}</div>
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

    html = "".join([f'<span class="pill">{escape(item)}</span>' for item in safe_items])
    st.markdown(f'<div class="pill-wrap">{html}</div>', unsafe_allow_html=True)


def soft_box_start() -> None:
    st.markdown('<div class="soft-box">', unsafe_allow_html=True)


def soft_box_end() -> None:
    st.markdown("</div>", unsafe_allow_html=True)
