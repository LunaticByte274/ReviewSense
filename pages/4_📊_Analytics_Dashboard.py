"""
ReviewSense - Page 4: Interactive Analytics Dashboard
Provides multi-dimensional cross-filtering, statistical breakdowns,
category/product intelligence, rating correlations, and timeline trends.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd

from src.config import AppConfig, THEME_TOKENS
from src.data_manager import DataManager
from src.analytics import AnalyticsEngine
from src.visualizations import (
    create_sentiment_donut_chart,
    create_sentiment_bar_chart,
    create_polarity_histogram,
    create_subjectivity_distribution,
    create_category_sentiment_chart,
    create_rating_vs_sentiment_chart,
    create_timeline_trend_chart
)
from src.ui_components import (
    inject_custom_css,
    render_sidebar,
    render_kpi_card,
    render_sentiment_badge,
    render_empty_state
)

# Page Setup
st.set_page_config(
    page_title=f"{AppConfig.APP_NAME} - Analytics Dashboard",
    page_icon="⚡",
    layout="wide"
)

inject_custom_css()
render_sidebar()

# Ensure active dataset
df = DataManager.ensure_active_dataset()
meta = st.session_state.get("dataset_metadata", {})

st.markdown("""
<div style="margin-bottom: 20px;">
    <h2 style="margin: 0; color: #F8FAFC;">📊 Product Review Analytics Dashboard</h2>
    <p style="color: #94A3B8; font-size: 0.95rem; margin-top: 4px;">
        Interactive statistical telemetry and sentiment distributions across products, categories, and customer ratings.
    </p>
</div>
""", unsafe_allow_html=True)

# ----------------- FILTERS BAR -----------------
with st.expander("🔍 Interactive Data Filters & Dimensions", expanded=True):
    fc1, fc2, fc3, fc4 = st.columns(4)

    with fc1:
        sentiment_options = ["All", "Positive", "Negative", "Neutral"]
        selected_sentiment = st.selectbox("Filter by Sentiment:", sentiment_options, index=0)

    with fc2:
        cat_col = meta.get("category_column") or ("category" if "category" in df.columns else None)
        if cat_col and cat_col in df.columns:
            available_cats = ["All"] + sorted([str(c) for c in df[cat_col].dropna().unique()])
            selected_category = st.selectbox("Product Category:", available_cats, index=0)
        else:
            selected_category = "All"
            st.selectbox("Product Category:", ["All (No category field)"], disabled=True)

    with fc3:
        prod_col = meta.get("product_column") or ("product_name" if "product_name" in df.columns else None)
        if prod_col and prod_col in df.columns:
            available_prods = ["All"] + sorted([str(p) for p in df[prod_col].dropna().unique()])
            selected_product = st.selectbox("Specific Product:", available_prods, index=0)
        else:
            selected_product = "All"
            st.selectbox("Specific Product:", ["All (No product field)"], disabled=True)

    with fc4:
        search_term = st.text_input("Keyword Search in Review:", placeholder="Search text...")

    # Slider for polarity range
    pol_range = st.slider(
        "Polarity Range Filter:",
        min_value=-1.0,
        max_value=1.0,
        value=(-1.0, 1.0),
        step=0.05
    )

# Apply filters
filtered_df = df.copy()

if selected_sentiment != "All" and "tb_sentiment" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["tb_sentiment"] == selected_sentiment]

if selected_category != "All" and cat_col and cat_col in filtered_df.columns:
    filtered_df = filtered_df[filtered_df[cat_col].astype(str) == selected_category]

if selected_product != "All" and prod_col and prod_col in filtered_df.columns:
    filtered_df = filtered_df[filtered_df[prod_col].astype(str) == selected_product]

if "tb_polarity" in filtered_df.columns:
    filtered_df = filtered_df[
        (filtered_df["tb_polarity"] >= pol_range[0]) & (filtered_df["tb_polarity"] <= pol_range[1])
    ]

text_col = meta.get("text_column", "review_text")
if search_term.strip() and text_col in filtered_df.columns:
    filtered_df = filtered_df[
        filtered_df[text_col].astype(str).str.contains(search_term.strip(), case=False, na=False)
    ]

# ----------------- EMPTY STATE CHECK -----------------
if filtered_df.empty:
    render_empty_state("No customer reviews match your active filter criteria. Try broadening your polarity range or clearing keywords.")
    st.stop()

# ----------------- KPI CARDS -----------------
kpis = AnalyticsEngine.calculate_kpis(filtered_df)

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    render_kpi_card("Filtered Reviews", f"{kpis['total_reviews']:,}", f"From {len(df):,} total", THEME_TOKENS["accent_primary"])
with c2:
    render_kpi_card("Positive", f"{kpis['positive_count']:,}", f"{kpis['positive_pct']}% of filtered", THEME_TOKENS["positive_color"])
with c3:
    render_kpi_card("Negative", f"{kpis['negative_count']:,}", f"{kpis['negative_pct']}% of filtered", THEME_TOKENS["negative_color"])
with c4:
    render_kpi_card("Neutral", f"{kpis['neutral_count']:,}", f"{kpis['neutral_pct']}% of filtered", THEME_TOKENS["neutral_color"])
with c5:
    render_kpi_card("Mean Polarity", f"{kpis['avg_polarity']:+.3f}", f"Avg Subj: {kpis['avg_subjectivity']:.2f}", THEME_TOKENS["accent_secondary"])

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# ----------------- CHARTS ROW 1: BREAKDOWNS -----------------
col_pie, col_bar = st.columns([1, 1])

with col_pie:
    st.plotly_chart(create_sentiment_donut_chart(filtered_df), use_container_width=True)

with col_bar:
    st.plotly_chart(create_sentiment_bar_chart(filtered_df), use_container_width=True)

# ----------------- CHARTS ROW 2: DISTRIBUTIONS -----------------
col_hist1, col_hist2 = st.columns([1, 1])

with col_hist1:
    st.plotly_chart(create_polarity_histogram(filtered_df), use_container_width=True)

with col_hist2:
    st.plotly_chart(create_subjectivity_distribution(filtered_df), use_container_width=True)

# ----------------- OPTIONAL ADVANCED CHARTS -----------------
# 1. Category Breakdown
if cat_col and cat_col in filtered_df.columns:
    cat_breakdown = AnalyticsEngine.get_category_breakdown(filtered_df, category_col=cat_col)
    if cat_breakdown is not None and len(cat_breakdown) > 1:
        st.markdown("### 🏢 Category-Level Sentiment Proportions")
        st.plotly_chart(create_category_sentiment_chart(cat_breakdown), use_container_width=True)

# 2. Rating vs Sentiment & Timeline Trend
rating_col = meta.get("rating_column") or ("rating" if "rating" in df.columns else None)
date_col = meta.get("date_column") or ("date" if "date" in df.columns else None)

col_r, col_t = st.columns([1, 1])
has_rating = rating_col and rating_col in filtered_df.columns
has_date = date_col and date_col in filtered_df.columns

with col_r:
    if has_rating:
        r_df = AnalyticsEngine.get_rating_vs_sentiment(filtered_df, rating_col=rating_col)
        if r_df is not None:
            st.plotly_chart(create_rating_vs_sentiment_chart(r_df), use_container_width=True)
        else:
            st.info("Insufficient star rating data to compute cross-tabulation.")
    else:
        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.4); border: 1px dashed rgba(255,255,255,0.08); border-radius: 8px; padding: 30px; text-align: center; color: #94A3B8;">
            No user star rating column configured in current dataset.
        </div>
        """, unsafe_allow_html=True)

with col_t:
    if has_date:
        t_df = AnalyticsEngine.get_timeline_trend(filtered_df, date_col=date_col)
        if t_df is not None:
            st.plotly_chart(create_timeline_trend_chart(t_df), use_container_width=True)
        else:
            st.info("Insufficient date timestamps to compute timeline trend.")
    else:
        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.4); border: 1px dashed rgba(255,255,255,0.08); border-radius: 8px; padding: 30px; text-align: center; color: #94A3B8;">
            No temporal date column configured in current dataset.
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ----------------- EXTREMES & ANALYZER AGREEMENT -----------------
col_ext_pos, col_ext_neg = st.columns(2)
extremes = AnalyticsEngine.get_extreme_reviews(filtered_df, n=3, text_col=text_col)

with col_ext_pos:
    st.markdown("##### 🌟 Most Positive Reviews (Highest Polarity)")
    pos_top = extremes["top_positive"]
    if not pos_top.empty:
        for _, row in pos_top.iterrows():
            st.markdown(f"""
            <div style="background: rgba(16, 185, 129, 0.08); border-left: 3px solid #10B981; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px;">
                <div style="font-size: 0.8rem; color: #10B981; font-weight: 600;">Polarity: {row.get('tb_polarity', 0):+.2f} | {row.get('product_name', 'Product')}</div>
                <div style="font-size: 0.84rem; color: #F8FAFC; margin-top: 4px;">"{row.get(text_col, '')}"</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.caption("No positive reviews available.")

with col_ext_neg:
    st.markdown("##### ⚠️ Most Critical Reviews (Lowest Polarity)")
    neg_top = extremes["top_negative"]
    if not neg_top.empty:
        for _, row in neg_top.iterrows():
            st.markdown(f"""
            <div style="background: rgba(244, 63, 94, 0.08); border-left: 3px solid #F43F5E; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px;">
                <div style="font-size: 0.8rem; color: #F43F5E; font-weight: 600;">Polarity: {row.get('tb_polarity', 0):+.2f} | {row.get('product_name', 'Product')}</div>
                <div style="font-size: 0.84rem; color: #F8FAFC; margin-top: 4px;">"{row.get(text_col, '')}"</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.caption("No negative reviews available.")

# Analyzer Agreement Telemetry
agr_stats = AnalyticsEngine.get_analyzer_agreement_stats(filtered_df)
if agr_stats.get("available"):
    st.markdown("---")
    st.markdown("### 🤝 TextBlob & VADER Dual-Analyzer Agreement")
    ag1, ag2, ag3 = st.columns(3)
    with ag1:
        st.metric("Total Compared", f"{agr_stats['total_evaluated']:,}")
    with ag2:
        st.metric("Consensus Matches", f"{agr_stats['agreed_count']:,} ({agr_stats['agreement_percentage']}%)")
    with ag3:
        st.metric("Divergent Classifications", f"{agr_stats['disagreed_count']:,}")
