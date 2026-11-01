"""
ReviewSense - UI Components & Design System
Provides reusable glassmorphism styling, KPI metric cards, badges, banners,
empty-state displays, theme customizers, and interactive sidebar telemetry.
"""

from typing import Optional, Dict, Any, List
import streamlit as st
from src.config import AppConfig, THEME_TOKENS

# Available UI Accent Themes
ACCENT_THEMES = {
    "Electric Indigo (Default)": {"primary": "#6366F1", "glow": "rgba(99, 102, 241, 0.25)", "border": "rgba(99, 102, 241, 0.4)"},
    "Cyberpunk Cyan": {"primary": "#06B6D4", "glow": "rgba(6, 182, 212, 0.25)", "border": "rgba(6, 182, 212, 0.4)"},
    "Emerald Mint": {"primary": "#10B981", "glow": "rgba(16, 185, 129, 0.25)", "border": "rgba(16, 185, 129, 0.4)"},
    "Sunset Rose": {"primary": "#F43F5E", "glow": "rgba(244, 63, 94, 0.25)", "border": "rgba(244, 63, 94, 0.4)"},
}


def inject_custom_css():
    """Injects ultra-premium SaaS styling matching Linear/Vercel/Stripe design aesthetics."""
    # Retrieve current accent theme
    theme_key = st.session_state.get("ui_accent_theme", "Electric Indigo (Default)")
    theme = ACCENT_THEMES.get(theme_key, ACCENT_THEMES["Electric Indigo (Default)"])
    accent_hex = theme["primary"]
    glow_rgba = theme["glow"]
    border_rgba = theme["border"]

    st.markdown(f"""
    <style>
    /* Global Typography & Font Family */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif;
    }}
    
    /* Main App Background with Subtle Mesh Gradient */
    .stApp {{
        background: radial-gradient(circle at 15% 15%, rgba(30, 41, 59, 0.5) 0%, transparent 40%),
                    radial-gradient(circle at 85% 85%, rgba(15, 23, 42, 0.8) 0%, transparent 40%),
                    #090D16;
        color: #F8FAFC;
    }}

    /* Top Header Bar */
    header[data-testid="stHeader"] {{
        background-color: rgba(9, 13, 22, 0.85);
        backdrop-filter: blur(16px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }}

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background-color: #0D131F;
        border-right: 1px solid rgba(255, 255, 255, 0.07);
    }}

    /* Glassmorphism Metric Cards */
    .metric-card {{
        background: rgba(22, 30, 46, 0.75);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4), 0 0 1px 1px rgba(255, 255, 255, 0.03);
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
        overflow: hidden;
    }}
    .metric-card:hover {{
        border-color: {border_rgba};
        transform: translateY(-2px);
        box-shadow: 0 8px 28px -4px rgba(0, 0, 0, 0.5), 0 0 15px {glow_rgba};
    }}
    .metric-title {{
        color: #94A3B8;
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}
    .metric-value {{
        color: #F8FAFC;
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }}
    .metric-subtitle {{
        color: #64748B;
        font-size: 0.78rem;
        margin-top: 5px;
        font-weight: 500;
    }}

    /* High-Definition Sentiment Badges */
    .badge-pos {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background-color: rgba(16, 185, 129, 0.12);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.35);
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        box-shadow: 0 0 8px rgba(16, 185, 129, 0.15);
    }}
    .badge-neg {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background-color: rgba(244, 63, 94, 0.12);
        color: #F43F5E;
        border: 1px solid rgba(244, 63, 94, 0.35);
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        box-shadow: 0 0 8px rgba(244, 63, 94, 0.15);
    }}
    .badge-neu {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background-color: rgba(148, 163, 184, 0.12);
        color: #94A3B8;
        border: 1px solid rgba(148, 163, 184, 0.35);
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }}

    /* Hero Banner with Glow & Dynamic Border */
    .hero-banner {{
        background: linear-gradient(135deg, rgba(22, 30, 46, 0.95) 0%, rgba(13, 19, 31, 0.95) 100%);
        border: 1px solid {border_rgba};
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.5), 0 0 20px {glow_rgba};
        border-radius: 14px;
        padding: 26px 30px;
        margin-bottom: 24px;
        position: relative;
    }}

    /* Glowing Pulse Status Dot */
    .status-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10B981;
        display: inline-block;
        box-shadow: 0 0 8px #10B981;
        animation: pulse 2s infinite;
    }}
    @keyframes pulse {{
        0% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }}
        70% {{ transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }}
        100% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }}
    }}

    /* Dataset Active Pill */
    .dataset-badge {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.12);
        color: #E2E8F0;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }}

    /* Action Card Hover Lift */
    .action-card {{
        background: rgba(22, 30, 46, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        height: 100%;
        transition: all 0.2s ease;
    }}
    .action-card:hover {{
        border-color: {accent_hex};
        transform: translateY(-3px);
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.4), 0 0 12px {glow_rgba};
    }}

    /* Code & Mono */
    code {{
        font-family: 'JetBrains Mono', monospace !important;
        color: #38BDF8 !important;
        background: rgba(13, 19, 31, 0.7) !important;
        padding: 2px 7px !important;
        border-radius: 5px !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
    }}

    /* Streamlit Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 4px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px 8px 0 0;
        padding: 8px 18px;
        font-weight: 600;
        font-size: 0.9rem;
        color: #94A3B8;
    }}
    .stTabs [aria-selected="true"] {{
        color: #FFFFFF !important;
        border-bottom: 2px solid {accent_hex} !important;
        background: rgba(255, 255, 255, 0.02);
    }}
    </style>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Renders persistent branding, active dataset telemetry, theme customizer, and quick actions."""
    with st.sidebar:
        st.markdown(f"""
        <div style="padding: 10px 0 18px 0;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="background: linear-gradient(135deg, #6366F1, #06B6D4); padding: 8px; border-radius: 10px; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 14px rgba(99, 102, 241, 0.4);">
                    <span style="font-size: 20px;">⚡</span>
                </div>
                <div>
                    <h2 style="margin: 0; font-size: 1.35rem; font-weight: 800; color: #F8FAFC; letter-spacing: -0.02em;">{AppConfig.APP_NAME}</h2>
                    <div style="display: flex; align-items: center; gap: 6px; margin-top: 2px;">
                        <span class="status-dot"></span>
                        <span style="font-size: 0.72rem; color: #10B981; font-weight: 700;">PROD v{AppConfig.APP_VERSION}</span>
                    </div>
                </div>
            </div>
            <p style="margin: 10px 0 0 0; font-size: 0.78rem; color: #94A3B8; line-height: 1.4;">
                Enterprise Product Review Intelligence & Dual-Lexicon NLP
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Active Dataset Telemetry
        if "dataset_metadata" in st.session_state and st.session_state["dataset_metadata"]:
            meta = st.session_state["dataset_metadata"]
            is_demo = meta.get("is_demo", False)
            source_label = meta.get("source", "Demo Dataset")
            row_count = meta.get("total_rows", 0)
            valid_rows = meta.get("valid_rows", 0)

            st.markdown(f"""
            <div style="background: rgba(22, 30, 46, 0.75); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 14px; margin-bottom: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-size: 0.7rem; color: #94A3B8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">Active Ingestion</span>
                    <span style="font-size: 0.68rem; color: #6366F1; background: rgba(99,102,241,0.15); padding: 2px 6px; border-radius: 4px; font-weight: 600;">{'DEMO' if is_demo else 'UPLOAD'}</span>
                </div>
                <div style="font-size: 0.9rem; font-weight: 700; color: #F8FAFC; margin-bottom: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="{source_label}">{source_label}</div>
                <div style="font-size: 0.76rem; color: #64748B; display: flex; justify-content: space-between;">
                    <span><b>{row_count:,}</b> total rows</span>
                    <span style="color: #10B981;">{valid_rows:,} scored</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if not is_demo:
                if st.button("↺ Restore Demo Data", use_container_width=True, help="Reset active session data back to the bundled demonstration dataset."):
                    from src.data_manager import DataManager
                    DataManager.restore_demonstration_data()
                    st.success("Demonstration data restored!")
                    st.rerun()

        # AI Copilot Quick Launcher
        if st.button("🤖 Launch AI Copilot", use_container_width=True, help="Open conversational review intelligence analyst"):
            st.switch_page("pages/8_🤖_AI_Copilot.py")

        # UI Accent Customizer
        with st.expander("🎨 Visual Accent Theme"):
            chosen_accent = st.selectbox(
                "Accent Highlight:",
                options=list(ACCENT_THEMES.keys()),
                index=list(ACCENT_THEMES.keys()).index(st.session_state.get("ui_accent_theme", "Electric Indigo (Default)")),
                key="theme_selector_dropdown"
            )
            if chosen_accent != st.session_state.get("ui_accent_theme"):
                st.session_state["ui_accent_theme"] = chosen_accent
                st.rerun()

        # System Diagnostics & Observability Telemetry
        with st.expander("⚙️ System Telemetry & Health"):
            from src.logger import get_system_telemetry
            diag = get_system_telemetry()
            st.caption(f"🐍 Python Runtime: **{diag.get('python_version')}**")
            st.caption(f"💾 Memory Footprint: **{diag.get('memory_rss_mb')} MB**")
            st.caption(f"💻 Architecture: **{diag.get('architecture')}**")
            st.caption(f"⚡ Environment: **{AppConfig.ENVIRONMENT.upper()}**")
            st.caption("🔒 Privacy: Local Zero-Exfiltration")

        st.markdown("---")
        st.markdown(
            "<div style='font-size: 0.72rem; color: #64748B; text-align: center; line-height: 1.5;'>"
            "TextBlob 0.20+ & NLTK VADER<br>"
            "Zero Cloud Data Exfiltration • Local Execution"
            "</div>",
            unsafe_allow_html=True
        )


def render_kpi_card(title: str, value: Any, subtitle: str = "", border_color: Optional[str] = None):
    """Renders an ultra-modern glassmorphism KPI metric card with animated border and subtle indicator bar."""
    accent_col = border_color or "#6366F1"
    border_style = f"border-left: 4px solid {accent_col};"
    st.markdown(f"""
    <div class="metric-card" style="{border_style}">
        <div class="metric-title">
            <span>{title}</span>
            <span style="color: {accent_col}; font-size: 14px;">●</span>
        </div>
        <div class="metric-value">{value}</div>
        {f'<div class="metric-subtitle">{subtitle}</div>' if subtitle else ''}
        <div style="height: 3px; background: rgba(255,255,255,0.06); border-radius: 2px; margin-top: 10px; overflow: hidden;">
            <div style="height: 100%; width: 70%; background: {accent_col}; border-radius: 2px;"></div>
        </div>
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
    <div style="text-align: center; padding: 48px 20px; background: rgba(22, 30, 46, 0.4); border: 1px dashed rgba(255, 255, 255, 0.12); border-radius: 14px; margin: 20px 0;">
        <span style="font-size: 40px; display: inline-block; margin-bottom: 8px;">🔍</span>
        <h4 style="margin: 8px 0 6px 0; color: #F8FAFC; font-weight: 700;">No Records Found</h4>
        <p style="color: #94A3B8; font-size: 0.88rem; max-width: 440px; margin: 0 auto; line-height: 1.5;">{message}</p>
    </div>
    """, unsafe_allow_html=True)
