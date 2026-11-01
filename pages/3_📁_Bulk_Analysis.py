"""
ReviewSense - Page 3: Bulk CSV Dataset Analysis & Ingestion
Provides multi-encoding CSV upload, automatic schema detection, batch NLP scoring,
atomic session state activation, and injection-safe export generation.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import datetime
import streamlit as st
import pandas as pd

from src.config import AppConfig, THEME_TOKENS
from src.data_manager import (
    DataManager,
    sanitize_dataframe_for_export
)
from src.analytics import AnalyticsEngine
from src.nlp_engine import default_nlp_engine
from src.visualizations import create_sentiment_donut_chart, create_sentiment_bar_chart
from src.ui_components import (
    inject_custom_css,
    render_sidebar,
    render_kpi_card,
    render_sentiment_badge
)

# Page Setup
st.set_page_config(
    page_title=f"{AppConfig.APP_NAME} - Bulk CSV Analysis",
    page_icon="⚡",
    layout="wide"
)

inject_custom_css()
render_sidebar()

st.markdown("""
<div style="margin-bottom: 20px;">
    <h2 style="margin: 0; color: #F8FAFC;">📁 Bulk CSV Analysis & Dataset Ingestion</h2>
    <p style="color: #94A3B8; font-size: 0.95rem; margin-top: 4px;">
        Upload customer feedback in CSV format to execute dual-lexicon NLP batch scoring and populate the shared platform dataset.
    </p>
</div>
""", unsafe_allow_html=True)

# Step 1: File Upload
st.markdown("### 📤 Step 1: Upload Dataset")
uploaded_file = st.file_uploader(
    label="Choose a CSV file (UTF-8, UTF-8 with BOM, Latin1 supported):",
    type=["csv"],
    help=f"Maximum recommended file size: {AppConfig.MAX_FILE_SIZE_MB}MB. The CSV must contain at least one column with customer review text."
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name

    # Validate file size
    file_size_mb = len(file_bytes) / (1024 * 1024)
    if file_size_mb > AppConfig.MAX_FILE_SIZE_MB:
        st.error(f"❌ File size exceeds {AppConfig.MAX_FILE_SIZE_MB}MB limit. Uploaded size: {file_size_mb:.1f}MB.")
        st.stop()

    # Parse CSV with multi-encoding fallback
    with st.spinner("Decoding and parsing CSV..."):
        raw_df, error_msg = DataManager.load_csv_bytes(file_bytes, filename=filename)

    if error_msg or raw_df is None:
        st.error(f"❌ CSV Validation Error: {error_msg}")
        st.info("ℹ️ Your previously active dataset remains intact.")
    else:
        st.success(f"✓ Successfully parsed **{filename}** ({len(raw_df):,} rows, {len(raw_df.columns)} columns).")

        # Step 2: Column Selection & Schema Mapping
        st.markdown("### ⚙️ Step 2: Configure Schema Mapping")
        detected = DataManager.detect_candidate_columns(raw_df)
        cols = list(raw_df.columns)

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            # Review Text Column (Mandatory)
            default_text_idx = cols.index(detected["text_column"]) if detected["text_column"] in cols else 0
            text_column = st.selectbox(
                "Review Text Column (Required):",
                options=cols,
                index=default_text_idx,
                help="Select the column containing the review text body to analyze."
            )

            # Product Category Column (Optional)
            cat_options = ["None"] + cols
            default_cat_idx = cat_options.index(detected["category_column"]) if detected["category_column"] in cat_options else 0
            category_column = st.selectbox(
                "Product Category Column (Optional):",
                options=cat_options,
                index=default_cat_idx,
                help="Categorical dimension for segmented analytics."
            )

        with col_c2:
            # Product Name / ID Column (Optional)
            prod_options = ["None"] + cols
            default_prod_idx = prod_options.index(detected["product_column"]) if detected["product_column"] in prod_options else 0
            product_column = st.selectbox(
                "Product Name / Identifier (Optional):",
                options=prod_options,
                index=default_prod_idx,
                help="Item title or ASIN for product-level comparisons."
            )

            # Rating Column (Optional)
            rating_options = ["None"] + cols
            default_rating_idx = rating_options.index(detected["rating_column"]) if detected["rating_column"] in rating_options else 0
            rating_column = st.selectbox(
                "User Star Rating Column (Optional):",
                options=rating_options,
                index=default_rating_idx,
                help="1-5 star ratings to evaluate rating vs. sentiment correlation."
            )

        # Raw Data Preview
        with st.expander("👀 Inspect Raw Upload Preview (First 5 Rows)"):
            st.dataframe(raw_df.head(5), use_container_width=True)

        st.markdown("---")

        # Step 3: Run Batch Analysis
        st.markdown("### ⚡ Step 3: Execute Batch NLP Intelligence")
        start_analysis = st.button("🚀 Analyze & Set as Active Dataset", type="primary")

        if start_analysis:
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def update_progress(ratio: float):
                progress_bar.progress(ratio)
                status_text.text(f"Processing NLP evaluations: {int(ratio * 100)}% complete...")

            t0 = datetime.datetime.now()
            analyzed_df = DataManager.process_dataframe(
                df=raw_df,
                text_column=text_column,
                engine=default_nlp_engine,
                progress_callback=update_progress
            )
            t1 = datetime.datetime.now()
            duration_sec = (t1 - t0).total_seconds()

            progress_bar.progress(1.0)
            status_text.text("Batch analysis completed successfully!")

            # Build metadata
            metadata = {
                "dataset_id": f"upload_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "source": f"User Upload: {filename}",
                "filename": filename,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_rows": len(analyzed_df),
                "valid_rows": int((analyzed_df["review_status"] == "Valid").sum()),
                "invalid_rows": int((analyzed_df["review_status"] != "Valid").sum()),
                "text_column": text_column,
                "category_column": category_column if category_column != "None" else None,
                "product_column": product_column if product_column != "None" else None,
                "rating_column": rating_column if rating_column != "None" else None,
                "date_column": detected.get("date_column"),
                "is_demo": False
            }

            # Atomically update active session state
            DataManager.set_active_dataset(analyzed_df, metadata)
            st.session_state["last_upload_success"] = True

            st.success(
                f"🎉 Processed **{len(analyzed_df):,}** reviews in **{duration_sec:.2f}s**! "
                f"Active dataset updated across all pages."
            )

# Check if active dataset is present to display results summary & export controls
active_df = st.session_state.get("active_data")
active_meta = st.session_state.get("dataset_metadata", {})

if active_df is not None:
    st.markdown("---")
    st.markdown("### 📊 Active Batch Results & Data Export")
    
    # Calculate KPIs
    kpis = AnalyticsEngine.calculate_kpis(active_df)

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        render_kpi_card("Total Rows", f"{kpis['total_reviews']:,}", f"{kpis['valid_reviews']:,} valid", THEME_TOKENS["accent_primary"])
    with k2:
        render_kpi_card("Positive", f"{kpis['positive_count']:,}", f"{kpis['positive_pct']}%", THEME_TOKENS["positive_color"])
    with k3:
        render_kpi_card("Negative", f"{kpis['negative_count']:,}", f"{kpis['negative_pct']}%", THEME_TOKENS["negative_color"])
    with k4:
        render_kpi_card("Neutral", f"{kpis['neutral_count']:,}", f"{kpis['neutral_pct']}%", THEME_TOKENS["neutral_color"])
    with k5:
        render_kpi_card("Avg Polarity", f"{kpis['avg_polarity']:+.3f}", f"Subj: {kpis['avg_subjectivity']:.2f}", THEME_TOKENS["accent_secondary"])

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Preview Processed Dataset
    st.markdown("##### 🔍 Enriched Dataset Preview (Top 10 Rows)")
    st.dataframe(active_df.head(10), use_container_width=True)

    # Export options with Formula Injection protection
    st.markdown("##### 📥 Export Intelligence")
    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        # Full Export
        safe_export_df = sanitize_dataframe_for_export(active_df)
        csv_buffer = safe_export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Download Full Processed CSV (With NLP Columns)",
            data=csv_buffer,
            file_name=f"ReviewSense_Processed_{active_meta.get('filename', 'dataset')}.csv",
            mime="text/csv",
            use_container_width=True,
            help="Exports all original fields plus tb_polarity, tb_subjectivity, tb_sentiment, and VADER scores, protected against formula injection."
        )

    with col_dl2:
        # Filtered Negative Reviews Export for QA/Support triage
        neg_df = active_df[active_df["tb_sentiment"] == "Negative"]
        if not neg_df.empty:
            safe_neg_df = sanitize_dataframe_for_export(neg_df)
            neg_csv_buffer = safe_neg_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"🚨 Export Negative Reviews ({len(neg_df):,} records for Triage)",
                data=neg_csv_buffer,
                file_name=f"ReviewSense_Negative_Triage_{active_meta.get('filename', 'dataset')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("No negative reviews detected in current dataset.")
