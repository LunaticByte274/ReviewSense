"""
ReviewSense - UI Components & Design System
Provides reusable styling, KPI metric cards, badges, banners,
empty-state displays, and sidebar headers.
"""

from typing import Optional, Dict, Any
import streamlit as st
from src.config import AppConfig, THEME_TOKENS


def inject_custom_css():
    """Injects refined SaaS styling matching Linear/Vercel design guidelines."""
    st.markdown("""
    <style>
    /* Global Typography & Font Family */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif;
    }
    
    /* Main App Background */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }

    /* Top Header Bar */
    header[data-testid="stHeader"] {
        background-color: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(8px);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* KPI Metric Cards */
    .metric-card {
        background: #1E293B;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2), 0 2px 4px -2px rgba(0, 0, 0, 0.2);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .metric-card:hover {
        border-color: rgba(255, 255, 255, 0.18);
        transform: translateY(-1px);
    }
    .metric-title {
        color: #94A3B8;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        color: #F8FAFC;
        font-size: 1.75rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .metric-subtitle {
        color: #64748B;
        font-size: 0.78rem;
        margin-top: 4px;
    }

    /* Sentiment Badges */
    .badge-pos {
        display: inline-flex;
        align-items: center;
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .badge-neg {
        display: inline-flex;
        align-items: center;
        background-color: rgba(244, 63, 94, 0.15);
        color: #F43F5E;
        border: 1px solid rgba(244, 63, 94, 0.3);
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .badge-neu {
        display: inline-flex;
        align-items: center;
        background-color: rgba(148, 163, 184, 0.15);
        color: #94A3B8;
        border: 1px solid rgba(148, 163, 184, 0.3);
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
    }

    /* Hero / Feature Banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
    }

    /* Active Dataset Banner */
    .dataset-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99, 102, 241, 0.12);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #A5B4FC;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 500;
    }

    /* Code & Pre */
    code {
        font-family: 'JetBrains Mono', monospace;
        color: #38BDF8 !important;
        background: rgba(15, 23, 42, 0.6) !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
    }
    </style>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Renders persistent branding, active dataset status, and global actions in the sidebar."""
    with st.sidebar:
        st.markdown(f"""
        <div style="padding: 10px 0 20px 0;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 26px;">⚡</span>
                <div>
                    <h2 style="margin: 0; font-size: 1.3rem; font-weight: 700; color: #F8FAFC;">{AppConfig.APP_NAME}</h2>
                    <span style="font-size: 0.75rem; color: #6366F1; font-weight: 600;">v{AppConfig.APP_VERSION}</span>
                </div>
            </div>
            <p style="margin: 6px 0 0 0; font-size: 0.75rem; color: #94A3B8;">NLP Product Review Intelligence</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Active Dataset Status
        if "dataset_metadata" in st.session_state and st.session_state["dataset_metadata"]:
            meta = st.session_state["dataset_metadata"]
            is_demo = meta.get("is_demo", False)
            source_label = meta.get("source", "Demo Dataset")
            row_count = meta.get("total_rows", 0)

            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 12px; margin-bottom: 16px;">
                <div style="font-size: 0.72rem; color: #94A3B8; font-weight: 600; text-transform: uppercase;">Active Dataset</div>
                <div style="font-size: 0.88rem; font-weight: 600; color: #F8FAFC; margin: 2px 0;">{source_label}</div>
                <div style="font-size: 0.75rem; color: #64748B;">{row_count:,} reviews processed</div>
            </div>
            """, unsafe_allow_html=True)

            if not is_demo:
                if st.button("↺ Restore Demo Data", use_container_width=True, help="Reset active session data back to the bundled demonstration dataset."):
                    from src.data_manager import DataManager
                    DataManager.restore_demonstration_data()
                    st.success("Demonstration data restored!")
                    st.rerun()
        else:
            st.info("Loading active dataset...")

        st.markdown("---")
        st.markdown(
            "<div style='font-size: 0.72rem; color: #64748B; text-align: center;'>"
            "ReviewSense Platform<br>TextBlob & VADER Dual-Engine NLP"
            "</div>",
            unsafe_allow_html=True
        )


def render_kpi_card(title: str, value: Any, subtitle: str = "", border_color: Optional[str] = None):
    """Renders a styled KPI metric card."""
    border_style = f"border-left: 3px solid {border_color};" if border_color else ""
    st.markdown(f"""
    <div class="metric-card" style="{border_style}">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        {f'<div class="metric-subtitle">{subtitle}</div>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)


def render_sentiment_badge(sentiment: str) -> str:
    """Returns HTML markup for a sentiment badge."""
    if sentiment == "Positive":
        return '<span class="badge-pos">● Positive</span>'
    elif sentiment == "Negative":
        return '<span class="badge-neg">● Negative</span>'
    else:
        return '<span class="badge-neu">● Neutral</span>'


def render_empty_state(message: str = "No reviews match the specified filters."):
    """Displays a clean empty state card."""
    st.markdown(f"""
    <div style="text-align: center; padding: 48px 20px; background: rgba(30, 41, 59, 0.4); border: 1px dashed rgba(255, 255, 255, 0.12); border-radius: 12px; margin: 20px 0;">
        <span style="font-size: 36px;">🔍</span>
        <h4 style="margin: 12px 0 6px 0; color: #F8FAFC;">No Records Found</h4>
        <p style="color: #94A3B8; font-size: 0.88rem; max-width: 400px; margin: 0 auto;">{message}</p>
    </div>
    """, unsafe_allow_html=True)
