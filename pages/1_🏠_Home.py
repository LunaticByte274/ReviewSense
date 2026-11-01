"""
ReviewSense - Page 1: Home Landing & Executive Summary
Displays intuitive workflow choosers, instant 1-click live sandbox,
executive KPIs with plain-English interpretations, visual sentiment charts,
and guided navigation.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path for direct page execution
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import src
src.reload_src_modules()

import streamlit as st
import pandas as pd

from src.config import AppConfig, THEME_TOKENS
from src.data_manager import DataManager
from src.analytics import AnalyticsEngine
from src.nlp_engine import default_nlp_engine
from src.aspect_engine import default_aspect_engine
from src.visualizations import (
    create_sentiment_donut_chart,
    create_sentiment_bar_chart,
    create_aspect_radar_chart
)
from src.ui_components import (
    inject_custom_css,
    render_sidebar,
    render_kpi_card,
    render_sentiment_badge
)

# Page Configuration
st.set_page_config(
    page_title=f"{AppConfig.APP_NAME} - Home",
    page_icon="⚡",
    layout="wide"
)

# Shared styling and sidebar
inject_custom_css()
render_sidebar()

# Ensure active dataset is initialized
df = DataManager.ensure_active_dataset()
meta = st.session_state.get("dataset_metadata", {})
total_records = len(df) if df is not None else 0

# ----------------- 1. WELCOME HERO SECTION -----------------
st.markdown(f"""
<div class="hero-banner">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 15px;">
        <div>
            <div class="dataset-badge">
                <span>●</span> Ready • {total_records:,} Reviews Active in Session ({meta.get('source', 'Demo Dataset')})
            </div>
            <h1 style="margin: 10px 0 6px 0; font-size: 2.3rem; font-weight: 800; color: #F8FAFC; letter-spacing: -0.02em;">
                ⚡ {AppConfig.APP_NAME}
            </h1>
            <p style="margin: 0; color: #94A3B8; font-size: 1.05rem; max-width: 720px; line-height: 1.5;">
                Turn messy customer reviews into clear, actionable product insights in seconds using dual-engine Natural Language Processing (NLP).
            </p>
        </div>
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 12px 18px; text-align: right;">
            <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">AI Status</div>
            <div style="font-size: 1.0rem; font-weight: 700; color: #10B981; margin: 2px 0;">● TextBlob + VADER Online</div>
            <div style="font-size: 0.75rem; color: #64748B;">Dual Consensus Architecture</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- 2. GUIDED WORKFLOW CHOOSER (WHERE TO START) -----------------
st.markdown("### 🎯 Choose How You Want to Start")
wf1, wf2, wf3 = st.columns(3)

with wf1:
    st.markdown("""
    <div style="background: rgba(22, 30, 46, 0.8); border: 1px solid rgba(99, 102, 241, 0.35); border-radius: 12px; padding: 20px; height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
        <div>
            <div style="font-size: 28px; margin-bottom: 8px;">⚡</div>
            <h3 style="margin: 0 0 6px 0; font-size: 1.15rem; color: #F8FAFC;">1. Test a Single Review</h3>
            <p style="color: #94A3B8; font-size: 0.85rem; line-height: 1.4; margin-bottom: 14px;">
                Type or paste any customer review to see real-time AI sentiment, emotional polarity, and auto-detected product features.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Try Single Review →", key="wf_single_btn", type="primary", use_container_width=True):
        st.switch_page("pages/2_🔍_Single_Review.py")

with wf2:
    st.markdown("""
    <div style="background: rgba(22, 30, 46, 0.8); border: 1px solid rgba(6, 182, 212, 0.35); border-radius: 12px; padding: 20px; height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
        <div>
            <div style="font-size: 28px; margin-bottom: 8px;">📊</div>
            <h3 style="margin: 0 0 6px 0; font-size: 1.15rem; color: #F8FAFC;">2. Explore Full Analytics</h3>
            <p style="color: #94A3B8; font-size: 0.85rem; line-height: 1.4; margin-bottom: 14px;">
                Browse interactive charts, aspect scorecards, keyword trends, and discover which features drive complaints.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Analytics Dashboard →", key="wf_dash_btn", use_container_width=True):
        st.switch_page("pages/4_📊_Analytics_Dashboard.py")

with wf3:
    st.markdown("""
    <div style="background: rgba(22, 30, 46, 0.8); border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 12px; padding: 20px; height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
        <div>
            <div style="font-size: 28px; margin-bottom: 8px;">🤖</div>
            <h3 style="margin: 0 0 6px 0; font-size: 1.15rem; color: #F8FAFC;">3. Ask AI Copilot</h3>
            <p style="color: #94A3B8; font-size: 0.85rem; line-height: 1.4; margin-bottom: 14px;">
                Ask questions in plain English: <i>"What are our top complaints?"</i>, <i>"How is battery feedback?"</i>, or <i>"Find rating mismatches"</i>.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Chat with AI Copilot →", key="wf_copilot_btn", use_container_width=True):
        st.switch_page("pages/8_🤖_AI_Copilot.py")

st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

# ----------------- 3. INSTANT 1-CLICK LIVE SANDBOX -----------------
st.markdown("### ⚡ Instant Live Sandbox (Click to Test)")
st.caption("Click any preset below to see how ReviewSense scores sentiment and identifies product features right on the spot:")

sb_c1, sb_c2, sb_c3 = st.columns(3)
sample_to_test = None

with sb_c1:
    if st.button("🌟 Positive: Audio & Battery Praise", use_container_width=True):
        sample_to_test = "The active noise cancellation is remarkable! Crystal clear sound and battery easily lasts 30 hours."

with sb_c2:
    if st.button("🔴 Complaint: Flimsy Hinge & Support", use_container_width=True):
        sample_to_test = "Terrible build quality. The hinge snapped after 4 days and customer service refused to refund me!"

with sb_c3:
    if st.button("🛡️ Nuance: Negation ('Not Bad')", use_container_width=True):
        sample_to_test = "This headset is not bad at all, and it definitely does not feel cheap for this price."

if sample_to_test:
    res = default_nlp_engine.analyze(sample_to_test, include_vader=True)
    aspects = default_aspect_engine.extract_aspects_from_text(sample_to_test)
    badge_bg = "rgba(16, 185, 129, 0.15)" if res.sentiment == "Positive" else "rgba(244, 63, 94, 0.15)" if res.sentiment == "Negative" else "rgba(148, 163, 184, 0.15)"
    badge_fg = "#10B981" if res.sentiment == "Positive" else "#F43F5E" if res.sentiment == "Negative" else "#94A3B8"
    
    aspect_pills = []
    for a_name, a_info in aspects.items():
        s_icon = "🟢" if a_info["sentiment"] == "Positive" else "🔴" if a_info["sentiment"] == "Negative" else "⚪"
        aspect_pills.append(f"{s_icon} **{a_name}** ({a_info['sentiment']})")
    aspect_str = " • ".join(aspect_pills) if aspect_pills else "General feedback (no specific dimension keyword)"

    st.markdown(f"""
    <div style="background: rgba(30, 41, 59, 0.95); border: 1px solid rgba(255,255,255,0.12); border-left: 4px solid {badge_fg}; border-radius: 10px; padding: 18px 22px; margin: 12px 0;">
        <div style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 6px;">Evaluated Review:</div>
        <div style="font-size: 0.95rem; font-style: italic; color: #F8FAFC; margin-bottom: 14px;">"{sample_to_test}"</div>
        <div style="display: flex; gap: 16px; align-items: center; flex-wrap: wrap;">
            <div style="background: {badge_bg}; color: {badge_fg}; padding: 4px 12px; border-radius: 6px; font-weight: 800; font-size: 0.9rem;">
                VERDICT: {res.sentiment.upper()} ({res.polarity:+.2f})
            </div>
            <div style="font-size: 0.82rem; color: #CBD5E1;">
                <b>Subjectivity:</b> {res.subjectivity*100:.0f}% Opinionated
            </div>
            <div style="font-size: 0.82rem; color: #CBD5E1;">
                <b>Detected Dimensions:</b> {aspect_str}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ----------------- 4. PLAIN-ENGLISH METRIC GUIDE (ACCORDION) -----------------
with st.expander("📘 How to Read These Metrics (1-Minute Plain English Guide)"):
    st.markdown("""
    | Metric | Plain English Meaning | How to Interpret |
    | :--- | :--- | :--- |
    | **Sentiment** | The overall emotional tone classified by AI (*Positive*, *Neutral*, *Negative*). | Quick indicator of customer satisfaction. |
    | **Polarity Score** | Precise score from **`-1.0`** (extremely angry) to **`+1.0`** (thrilled). `0.0` is neutral. | Values above `+0.05` are positive; values below `-0.05` are negative. |
    | **Net Sentiment (NSS)** | Percentage of **Positive reviews minus Negative reviews**. (Similar to NPS). | Above `+40` = World-class loyalty; `0 to +20` = Moderate; `< 0` = Critical issues. |
    | **Subjectivity** | Measures whether the review contains **personal opinions** (`1.0`) vs **factual specs** (`0.0`). | Higher subjectivity means emotional sentiment; lower means factual reporting. |
    | **Aspect Dimensions** | Granular ratings across 8 specific product features (Battery, Sound, Build, Price, etc.). | Tells your engineering and product teams **exactly what to fix or praise**. |
    """)

# Calculate real KPIs from active dataset
kpis = AnalyticsEngine.calculate_kpis(df)
nss = round(kpis["positive_pct"] - kpis["negative_pct"], 1)

# ----------------- 5. EXECUTIVE METRICS OVERVIEW -----------------
st.markdown("### 📈 Executive Performance Overview")
st.caption(f"Calculated in real time over all **{total_records:,} reviews** in the active dataset:")

c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    render_kpi_card(
        title="Total Reviews",
        value=f"{kpis['total_reviews']:,}",
        subtitle=f"{kpis['valid_reviews']:,} verified entries",
        border_color=THEME_TOKENS["accent_primary"]
    )
with c2:
    render_kpi_card(
        title="Positive Reviews",
        value=f"{kpis['positive_count']:,}",
        subtitle=f"{kpis['positive_pct']}% of customer base",
        border_color=THEME_TOKENS["positive_color"]
    )
with c3:
    render_kpi_card(
        title="Negative Reviews",
        value=f"{kpis['negative_count']:,}",
        subtitle=f"{kpis['negative_pct']}% complaints",
        border_color=THEME_TOKENS["negative_color"]
    )
with c4:
    render_kpi_card(
        title="Neutral Reviews",
        value=f"{kpis['neutral_count']:,}",
        subtitle=f"{kpis['neutral_pct']}% factual/mixed",
        border_color=THEME_TOKENS["neutral_color"]
    )
with c5:
    render_kpi_card(
        title="Avg Polarity",
        value=f"{kpis['avg_polarity']:+.3f}",
        subtitle="Scale: -1.0 to +1.0",
        border_color=THEME_TOKENS["accent_secondary"]
    )
with c6:
    render_kpi_card(
        title="Net Sentiment (NSS)",
        value=f"{nss:+0.1f}",
        subtitle="🟢 Strong Brand Health" if nss >= 40 else "🟡 Moderate",
        border_color="#10B981" if nss >= 20 else ("#F59E0B" if nss >= 0 else "#F43F5E")
    )

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# Visual Summary Row
col_chart_1, col_chart_2 = st.columns([1, 1])

with col_chart_1:
    st.plotly_chart(create_sentiment_donut_chart(df), use_container_width=True)

with col_chart_2:
    st.plotly_chart(create_sentiment_bar_chart(df), use_container_width=True)

# ----------------- 6. AUTOMATED EXECUTIVE TAKEAWAYS -----------------
text_col = meta.get("text_column", "review_text")
aspect_df = default_aspect_engine.analyze_dataset_aspects(df, text_col=text_col)
rating_col = meta.get("rating_column", "rating")
anomalies_df = AnalyticsEngine.detect_sentiment_anomalies(df, rating_col=rating_col) if rating_col in df.columns else None

insights = AnalyticsEngine.generate_executive_insights(df, kpis, aspect_df=aspect_df, anomalies_df=anomalies_df)

st.markdown("### 💡 Key Findings & Strategic Recommendations")
ins_cols = st.columns(len(insights))
for idx, ins in enumerate(insights):
    border_color = (
        "#10B981" if ins["type"] == "positive"
        else ("#F43F5E" if ins["type"] == "negative"
        else ("#F59E0B" if ins["type"] == "warning" else "#6366F1"))
    )
    with ins_cols[idx]:
        st.markdown(f"""
        <div style="background: rgba(22, 30, 46, 0.75); border: 1px solid rgba(255,255,255,0.08); border-top: 3px solid {border_color}; border-radius: 10px; padding: 16px; height: 100%; box-shadow: 0 4px 12px rgba(0,0,0,0.25);">
            <div style="font-size: 0.72rem; color: #94A3B8; font-weight: 700; text-transform: uppercase;">{ins['category']}</div>
            <div style="font-size: 0.95rem; font-weight: 700; color: #F8FAFC; margin: 4px 0 6px 0;">{ins['title']}</div>
            <div style="font-size: 0.8rem; color: #CBD5E1; line-height: 1.4;">{ins['description']}</div>
        </div>
        """, unsafe_allow_html=True)

# Aspect Intelligence Summary Callout
if not aspect_df.empty:
    best_aspect = aspect_df.sort_values(by="net_aspect_score", ascending=False).iloc[0]
    worst_aspect = aspect_df.sort_values(by="net_aspect_score", ascending=True).iloc[0]
    
    st.markdown("### 🎯 Key Product Drivers (What's Working vs What's Broken)")
    a_col1, a_col2 = st.columns(2)
    with a_col1:
        st.markdown(f"""
        <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 8px; padding: 14px 18px;">
            <div style="font-size: 0.75rem; color: #10B981; font-weight: 700; text-transform: uppercase;">Top Winning Product Feature</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #F8FAFC; margin: 4px 0;">{best_aspect['aspect']}</div>
            <div style="font-size: 0.82rem; color: #CBD5E1;">
                <b>{best_aspect['positive_rate']}% Positive</b> ({best_aspect['positive_count']} positive mentions) • Net Score: <b>{best_aspect['net_aspect_score']:+0.1f}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with a_col2:
        st.markdown(f"""
        <div style="background: rgba(244, 63, 94, 0.08); border: 1px solid rgba(244, 63, 94, 0.25); border-radius: 8px; padding: 14px 18px;">
            <div style="font-size: 0.75rem; color: #F43F5E; font-weight: 700; text-transform: uppercase;">Primary Customer Pain Point</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #F8FAFC; margin: 4px 0;">{worst_aspect['aspect']}</div>
            <div style="font-size: 0.82rem; color: #CBD5E1;">
                <b>{worst_aspect['negative_rate']}% Negative</b> ({worst_aspect['negative_count']} complaints) • Net Score: <b>{worst_aspect['net_aspect_score']:+0.1f}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ----------------- 7. COMPLETE PLATFORM DIRECTORY -----------------
st.markdown("### 🧭 Complete Navigation Directory")
st.caption("Access all analytical and educational tools in the platform:")

nav1, nav2, nav3, nav4 = st.columns(4)

with nav1:
    st.markdown("""
    <div style="background: #1E293B; border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 16px; height: 100%;">
        <div style="font-size: 22px;">🔍 <b>Single Review</b></div>
        <p style="color: #94A3B8; font-size: 0.8rem; margin: 6px 0 10px 0;">Detailed dual-engine scoring with polarity gauges and linguistic breakdown.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Single Review →", key="dir_btn_single", use_container_width=True):
        st.switch_page("pages/2_🔍_Single_Review.py")

with nav2:
    st.markdown("""
    <div style="background: #1E293B; border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 16px; height: 100%;">
        <div style="font-size: 22px;">📁 <b>Bulk Analysis</b></div>
        <p style="color: #94A3B8; font-size: 0.8rem; margin: 6px 0 10px 0;">Upload your CSV files, batch-score feedback, and generate HTML reports.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Bulk Analysis →", key="dir_btn_bulk", use_container_width=True):
        st.switch_page("pages/3_📁_Bulk_Analysis.py")

with nav3:
    st.markdown("""
    <div style="background: #1E293B; border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 16px; height: 100%;">
        <div style="font-size: 22px;">🔎 <b>Review Explorer</b></div>
        <p style="color: #94A3B8; font-size: 0.8rem; margin: 6px 0 10px 0;">Filter, search, sort, and inspect individual customer reviews with pagination.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Explorer →", key="dir_btn_explorer", use_container_width=True):
        st.switch_page("pages/5_🔎_Review_Explorer.py")

with nav4:
    st.markdown("""
    <div style="background: #1E293B; border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 16px; height: 100%;">
        <div style="font-size: 22px;">⚙️ <b>NLP Preprocessing</b></div>
        <p style="color: #94A3B8; font-size: 0.8rem; margin: 6px 0 10px 0;">See how tokenization, cleaning, and negation preservation work step-by-step.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Preprocessing Lab →", key="dir_btn_preproc", use_container_width=True):
        st.switch_page("pages/6_⚙️_Text_Preprocessing.py")
