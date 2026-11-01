"""
ReviewSense - Page 5: Review Explorer & Detail Inspector
Interactive multi-field table with full-text search, column sorting,
sentiment & subjectivity filters, pagination, and filtered CSV export.
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
from src.data_manager import DataManager, sanitize_dataframe_for_export
from src.aspect_engine import PRODUCT_ASPECTS
from src.ui_components import (
    inject_custom_css,
    render_sidebar,
    render_empty_state
)

# Page Setup
st.set_page_config(
    page_title=f"{AppConfig.APP_NAME} - Review Explorer",
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
    <h2 style="margin: 0; color: #F8FAFC;">🔎 Review Explorer & Granular Telemetry</h2>
    <p style="color: #94A3B8; font-size: 0.95rem; margin-top: 4px;">
        Filter, search, inspect, and export individual customer feedback entries with comprehensive NLP scores.
    </p>
</div>
""", unsafe_allow_html=True)

# ----------------- FILTER CONTROLS -----------------
with st.container():
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)

    with f_col1:
        sentiment_opts = ["All", "Positive", "Negative", "Neutral"]
        selected_sent = st.selectbox("Sentiment Filter:", sentiment_opts, index=0)

    with f_col2:
        cat_col = meta.get("category_column") or ("category" if "category" in df.columns else None)
        if cat_col and cat_col in df.columns:
            cat_list = ["All"] + sorted([str(x) for x in df[cat_col].dropna().unique()])
            selected_cat = st.selectbox("Category Filter:", cat_list, index=0)
        else:
            selected_cat = "All"
            st.selectbox("Category Filter:", ["All (N/A)"], disabled=True)

    with f_col3:
        search_query = st.text_input("Search Text Content:", placeholder="Keyword (e.g., battery, sound)...")

    with f_col4:
        sort_options = {
            "Highest Polarity First": ("tb_polarity", False),
            "Lowest Polarity First": ("tb_polarity", True),
            "Highest Subjectivity First": ("tb_subjectivity", False),
            "Lowest Subjectivity First": ("tb_subjectivity", True),
        }
        if "rating" in df.columns:
            sort_options["Highest Rating First"] = ("rating", False)
            sort_options["Lowest Rating First"] = ("rating", True)

        sort_choice = st.selectbox("Sort Order:", list(sort_options.keys()), index=0)

    # Secondary filter row: Aspect and Anomalies
    f2_col1, f2_col2, f2_col3 = st.columns([1.5, 1.5, 1])
    with f2_col1:
        aspect_filter_opts = ["All Aspects"] + list(PRODUCT_ASPECTS.keys())
        selected_aspect = st.selectbox("Filter by Mentioned Aspect:", aspect_filter_opts, index=0)
    with f2_col2:
        pol_bounds = st.slider("Polarity Range:", -1.0, 1.0, (-1.0, 1.0), 0.05)
    with f2_col3:
        sub_bounds = st.slider("Subjectivity Range:", 0.0, 1.0, (0.0, 1.0), 0.05)
        only_anomalies = st.checkbox("Show Rating Mismatches Only", value=False, help="Filters reviews where star rating contradicts textual polarity (e.g. 5-star negative or 1-star positive).")

# Apply Filters
exp_df = df.copy()

if selected_sent != "All" and "tb_sentiment" in exp_df.columns:
    exp_df = exp_df[exp_df["tb_sentiment"] == selected_sent]

if selected_cat != "All" and cat_col and cat_col in exp_df.columns:
    exp_df = exp_df[exp_df[cat_col].astype(str) == selected_cat]

if selected_aspect != "All Aspects":
    keywords = PRODUCT_ASPECTS[selected_aspect]
    pattern = r"\b(?:" + "|".join(keywords) + r")\b"
    text_col = meta.get("text_column", "review_text")
    if text_col in exp_df.columns:
        exp_df = exp_df[exp_df[text_col].astype(str).str.contains(pattern, case=False, na=False)]

if only_anomalies and "rating" in exp_df.columns and "tb_polarity" in exp_df.columns:
    num_r = pd.to_numeric(exp_df["rating"], errors="coerce")
    anom_mask = ((num_r >= 4) & (exp_df["tb_polarity"] < -0.05)) | ((num_r <= 2) & (exp_df["tb_polarity"] > 0.05))
    exp_df = exp_df[anom_mask]

if "tb_polarity" in exp_df.columns:
    exp_df = exp_df[(exp_df["tb_polarity"] >= pol_bounds[0]) & (exp_df["tb_polarity"] <= pol_bounds[1])]

if "tb_subjectivity" in exp_df.columns:
    exp_df = exp_df[(exp_df["tb_subjectivity"] >= sub_bounds[0]) & (exp_df["tb_subjectivity"] <= sub_bounds[1])]

text_col = meta.get("text_column", "review_text")
if search_query.strip() and text_col in exp_df.columns:
    exp_df = exp_df[exp_df[text_col].astype(str).str.contains(search_query.strip(), case=False, na=False)]

# Sorting
sort_col, ascending_order = sort_options[sort_choice]
if sort_col in exp_df.columns:
    exp_df = exp_df.sort_values(by=sort_col, ascending=ascending_order)

st.markdown("---")

# Empty State Check
if exp_df.empty:
    render_empty_state("No customer reviews match your active filter specifications.")
    st.stop()

# ----------------- PAGINATION CONTROLS -----------------
total_matching = len(exp_df)
p_col1, p_col2, p_col3 = st.columns([2, 1, 1])

with p_col1:
    st.markdown(f"**Showing {total_matching:,} of {len(df):,} total reviews** in active dataset.")

with p_col2:
    page_size = st.selectbox("Reviews per page:", [10, 25, 50, 100], index=0)

total_pages = max(1, (total_matching + page_size - 1) // page_size)

with p_col3:
    current_page = st.number_input("Page:", min_value=1, max_value=total_pages, value=1, step=1)

# Slice for current page
start_idx = (current_page - 1) * page_size
end_idx = min(start_idx + page_size, total_matching)
page_df = exp_df.iloc[start_idx:end_idx]

# Preferred column display order
preferred_cols = [
    "review_id", "product_name", "category", "rating", "tb_sentiment",
    "tb_polarity", "tb_subjectivity", "vader_sentiment", "vader_compound", text_col
]
available_disp_cols = [c for c in preferred_cols if c in page_df.columns]
# If any extra original columns exist, append them
remaining_cols = [c for c in page_df.columns if c not in available_disp_cols and c not in ("clean_text", "review_status")]
all_disp_cols = available_disp_cols + remaining_cols

# Display Data Table
st.dataframe(
    page_df[all_disp_cols],
    use_container_width=True,
    height=420
)

# Export Filtered Dataset
st.markdown("---")
export_col1, export_col2 = st.columns([2, 1])
with export_col1:
    st.caption("All exported datasets are sanitized to prevent CSV spreadsheet injection attacks.")

with export_col2:
    safe_filtered = sanitize_dataframe_for_export(exp_df[all_disp_cols])
    csv_bytes = safe_filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"💾 Export Filtered Reviews ({total_matching:,} Rows) as CSV",
        data=csv_bytes,
        file_name="ReviewSense_Filtered_Explorer_Export.csv",
        mime="text/csv",
        use_container_width=True
    )
