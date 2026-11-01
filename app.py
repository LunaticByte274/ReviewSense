"""
ReviewSense - Main Application Entry Point
Initializes global application state, design system CSS, active session data,
and renders the executive home landing dashboard.
"""

import streamlit as st
import pandas as pd

from src.config import AppConfig, THEME_TOKENS
from src.data_manager import DataManager
from src.analytics import AnalyticsEngine
from src.aspect_engine import default_aspect_engine
from src.visualizations import (
    create_sentiment_donut_chart,
    create_sentiment_bar_chart,
    create_aspect_radar_chart
)
from src.ui_components import (
    inject_custom_css,
    render_sidebar,
    render_kpi_card
)

# Set Streamlit Page Configuration
st.set_page_config(
    page_title=f"{AppConfig.APP_NAME} - Enterprise Review Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom SaaS CSS & Render Standardized Sidebar
inject_custom_css()
render_sidebar()

# Ensure Active Dataset is Initialized
df = DataManager.ensure_active_dataset()
meta = st.session_state.get("dataset_metadata", {})

# Top Hero Section
st.markdown(f"""
<div class="hero-banner">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 15px;">
        <div>
            <div class="dataset-badge">
                <span>●</span> Active Source: {meta.get('source', 'Demonstration Dataset')}
            </div>
            <h1 style="margin: 12px 0 6px 0; font-size: 2.2rem; font-weight: 700; color: #F8FAFC;">
                {AppConfig.APP_NAME}
            </h1>
            <p style="margin: 0; color: #94A3B8; font-size: 1.05rem; max-width: 680px;">
                {AppConfig.APP_TAGLINE}
            </p>
        </div>
        <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 10px 16px; text-align: right;">
            <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; font-weight: 600;">Engine Status</div>
            <div style="font-size: 0.95rem; font-weight: 600; color: #10B981;">● TextBlob & VADER Online</div>
            <div style="font-size: 0.75rem; color: #64748B;">Dual-Lexicon NLP Architecture</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Calculate real KPIs from active dataset
kpis = AnalyticsEngine.calculate_kpis(df)
nss = round(kpis["positive_pct"] - kpis["negative_pct"], 1)

# KPI Metrics Bar
st.markdown("### 📈 Executive Metrics Overview")
c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    render_kpi_card(
        title="Total Reviews",
        value=f"{kpis['total_reviews']:,}",
        subtitle=f"{kpis['valid_reviews']:,} validated entries",
        border_color=THEME_TOKENS["accent_primary"]
    )
with c2:
    render_kpi_card(
        title="Positive Reviews",
        value=f"{kpis['positive_count']:,}",
        subtitle=f"{kpis['positive_pct']}% of dataset",
        border_color=THEME_TOKENS["positive_color"]
    )
with c3:
    render_kpi_card(
        title="Negative Reviews",
        value=f"{kpis['negative_count']:,}",
        subtitle=f"{kpis['negative_pct']}% of dataset",
        border_color=THEME_TOKENS["negative_color"]
    )
with c4:
    render_kpi_card(
        title="Neutral Reviews",
        value=f"{kpis['neutral_count']:,}",
        subtitle=f"{kpis['neutral_pct']}% of dataset",
        border_color=THEME_TOKENS["neutral_color"]
    )
with c5:
    render_kpi_card(
        title="Avg Polarity",
        value=f"{kpis['avg_polarity']:+.3f}",
        subtitle=f"Subj: {kpis['avg_subjectivity']:.2f}",
        border_color=THEME_TOKENS["accent_secondary"]
    )
with c6:
    render_kpi_card(
        title="Net Sentiment (NSS)",
        value=f"{nss:+0.1f}",
        subtitle="Pos% minus Neg%",
        border_color="#10B981" if nss >= 20 else ("#F59E0B" if nss >= 0 else "#F43F5E")
    )

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# Visual Summary Row
col_chart_1, col_chart_2 = st.columns([1, 1])

with col_chart_1:
    st.plotly_chart(create_sentiment_donut_chart(df), use_container_width=True)

with col_chart_2:
    st.plotly_chart(create_sentiment_bar_chart(df), use_container_width=True)

# ----------------- ENHANCEMENT: AUTOMATED EXECUTIVE TAKEAWAYS -----------------
text_col = meta.get("text_column", "review_text")
aspect_df = default_aspect_engine.analyze_dataset_aspects(df, text_col=text_col)
rating_col = meta.get("rating_column", "rating")
anomalies_df = AnalyticsEngine.detect_sentiment_anomalies(df, rating_col=rating_col) if rating_col in df.columns else None

insights = AnalyticsEngine.generate_executive_insights(df, kpis, aspect_df=aspect_df, anomalies_df=anomalies_df)

st.markdown("### 💡 Automated Executive Takeaways & CX Strategy")
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

# Aspect Radar Chart Section
if not aspect_df.empty:
    with st.expander("🕸️ View 8-Axis Aspect Dimension Radar Chart"):
        r_col1, r_col2 = st.columns([1, 1])
        with r_col1:
            st.plotly_chart(create_aspect_radar_chart(aspect_df), use_container_width=True)
        with r_col2:
            st.markdown("##### 📋 Dimension Performance Summary")
            st.markdown("The 8-axis radar chart models customer satisfaction percentages across discrete product engineering dimensions.")
            st.dataframe(aspect_df[["aspect", "total_mentions", "positive_rate", "net_aspect_score"]], use_container_width=True)

# Aspect Intelligence Summary Callout
if not aspect_df.empty:
    best_aspect = aspect_df.sort_values(by="net_aspect_score", ascending=False).iloc[0]
    worst_aspect = aspect_df.sort_values(by="net_aspect_score", ascending=True).iloc[0]
    
    st.markdown("### 🎯 Key Product Drivers (Aspect Intelligence)")
    a_col1, a_col2 = st.columns(2)
    with a_col1:
        st.markdown(f"""
        <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 8px; padding: 14px 18px;">
            <div style="font-size: 0.75rem; color: #10B981; font-weight: 700; text-transform: uppercase;">Top Customer Praise Driver</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #F8FAFC; margin: 4px 0;">{best_aspect['aspect']}</div>
            <div style="font-size: 0.82rem; color: #CBD5E1;">
                <b>{best_aspect['positive_rate']}% Positive</b> ({best_aspect['positive_count']} praises) • Net Score: <b>{best_aspect['net_aspect_score']:+0.1f}</b>
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

# Quick Navigation Actions
st.markdown("### 🚀 Platform Capabilities & Quick Launch")
act1, act2, act3, act4 = st.columns(4)

with act1:
    st.markdown("""
    <div style="background: #1E293B; border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 18px; height: 100%;">
        <div style="font-size: 24px; margin-bottom: 8px;">🔍</div>
        <h4 style="margin: 0 0 6px 0; color: #F8FAFC;">Single Review</h4>
        <p style="color: #94A3B8; font-size: 0.82rem; margin-bottom: 12px;">Instant NLP scoring, polarity gauge, subjectivity, and VADER agreement checks.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Single Review →", key="root_btn_single", use_container_width=True):
        st.switch_page("pages/2_🔍_Single_Review.py")

with act2:
    st.markdown("""
    <div style="background: #1E293B; border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 18px; height: 100%;">
        <div style="font-size: 24px; margin-bottom: 8px;">📁</div>
        <h4 style="margin: 0 0 6px 0; color: #F8FAFC;">Bulk Analysis</h4>
        <p style="color: #94A3B8; font-size: 0.82rem; margin-bottom: 12px;">Upload CSV datasets, auto-detect columns, batch process, and export enriched data.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Bulk Analysis →", key="root_btn_bulk", use_container_width=True):
        st.switch_page("pages/3_📁_Bulk_Analysis.py")

with act3:
    st.markdown("""
    <div style="background: #1E293B; border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 18px; height: 100%;">
        <div style="font-size: 24px; margin-bottom: 8px;">📊</div>
        <h4 style="margin: 0 0 6px 0; color: #F8FAFC;">Analytics Dashboard</h4>
        <p style="color: #94A3B8; font-size: 0.82rem; margin-bottom: 12px;">Interactive cross-filtering, category breakdowns, star ratings, and timeline trends.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Dashboard →", key="root_btn_dash", use_container_width=True):
        st.switch_page("pages/4_📊_Analytics_Dashboard.py")

with act4:
    st.markdown("""
    <div style="background: #1E293B; border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 18px; height: 100%;">
        <div style="font-size: 24px; margin-bottom: 8px;">🔎</div>
        <h4 style="margin: 0 0 6px 0; color: #F8FAFC;">Review Explorer</h4>
        <p style="color: #94A3B8; font-size: 0.82rem; margin-bottom: 12px;">Search, sort, paginate, and inspect individual customer sentiments in detail.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Explorer →", key="root_btn_explorer", use_container_width=True):
        st.switch_page("pages/5_🔎_Review_Explorer.py")

st.markdown("---")

# Active Dataset Sample Preview
st.markdown("### 📋 Active Dataset Preview (Top 5 Records)")
text_col = meta.get("text_column", "review_text")
display_cols = [c for c in ["product_name", "category", "rating", "tb_sentiment", "tb_polarity", text_col] if c in df.columns]
st.dataframe(df[display_cols].head(5), use_container_width=True)

# Getting Started Guide & Architecture
with st.expander("ℹ️ Getting Started & Platform Architecture Guide"):
    st.markdown("""
    - **Active Dataset Persistence**: ReviewSense maintains a shared session dataset across all pages. You can upload custom CSV data in **Bulk Analysis** or analyze the default demonstration dataset.
    - **Dual-Engine NLP**: Sentiment is scored using **TextBlob** (Pattern-based lexicon evaluating polarity and subjectivity) alongside **VADER** (rule-based valence analyzer tuned for social and review sentiment).
    - **Restoration**: If you wish to reset your view back to the bundled 50-review demo dataset, use the **↺ Restore Demo Data** button in the sidebar anytime.
    """)

