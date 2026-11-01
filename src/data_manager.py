"""
ReviewSense - Data Management Layer
Handles CSV loading, robust schema validation, multi-encoding parsing,
batch NLP execution, session-state synchronization, and injection-safe exports.
"""

import io
import gc
import time
import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import streamlit as st

from src.config import (
    AppConfig,
    SAMPLE_REVIEWS_CSV,
    EVAL_LABELED_CSV
)
from src.nlp_engine import default_nlp_engine, NLPEngine
from src.preprocessing import clean_html, normalize_whitespace
from src.logger import logger, log_execution_time


def sanitize_for_csv_injection(val: Any) -> Any:
    """
    Guards against CSV formula injection (OWASP CSV Injection).
    Prepends a single apostrophe to strings beginning with dangerous formula characters:
    '=', '+', '-', '@', '\\t', '\\r'.
    Preserves legitimate numeric types (int, float).
    """
    if isinstance(val, str) and len(val) > 0:
        if val[0] in ("=", "+", "-", "@", "\t", "\r"):
            return f"'{val}"
    return val


def sanitize_dataframe_for_export(df: pd.DataFrame) -> pd.DataFrame:
    """Applies CSV formula injection sanitization to all object/string columns."""
    export_df = df.copy()
    for col in export_df.columns:
        if (
            export_df[col].dtype in ("object", "str", "string")
            or pd.api.types.is_string_dtype(export_df[col])
        ):
            export_df[col] = export_df[col].apply(sanitize_for_csv_injection)
    return export_df


class DataManager:
    """Central dataset ingestion, validation, batch scoring, and session manager."""

    @staticmethod
    def load_csv_bytes(file_bytes: bytes, filename: str = "upload.csv") -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Loads CSV content from raw bytes trying multiple standard encodings.
        Handles UTF-8, UTF-8-sig (with BOM), latin1, and cp1252.
        """
        if not file_bytes or len(file_bytes) == 0:
            return None, "The uploaded file is empty (0 bytes)."

        encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
        last_error = None

        for enc in encodings:
            try:
                buffer = io.BytesIO(file_bytes)
                df = pd.read_csv(buffer, encoding=enc)
                if df.empty:
                    return None, "The uploaded CSV contains no data rows."
                # Strip leading/trailing whitespaces from column headers
                df.columns = [str(c).strip() for c in df.columns]
                return df, None
            except Exception as e:
                last_error = str(e)
                continue

        return None, f"Failed to parse CSV with standard encodings: {last_error}"

    @staticmethod
    def detect_candidate_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
        """
        Inspects dataframe columns and recommends matching fields for:
        - text_column (review body)
        - category_column
        - product_column
        - rating_column
        - date_column
        """
        cols_lower = {str(c).lower(): c for c in df.columns}
        candidates = {
            "text_column": None,
            "category_column": None,
            "product_column": None,
            "rating_column": None,
            "date_column": None,
        }

        # Match text column (ignoring explicit ID columns like 'review_id')
        for cand in AppConfig.TEXT_COLUMN_CANDIDATES:
            for col_l, orig_col in cols_lower.items():
                if col_l.endswith("_id") or col_l == "id" or col_l.endswith("id"):
                    continue
                if cand in col_l:
                    candidates["text_column"] = orig_col
                    break
            if candidates["text_column"]:
                break

        # Match category column
        for cand in AppConfig.CATEGORY_COLUMN_CANDIDATES:
            for col_l, orig_col in cols_lower.items():
                if cand in col_l:
                    candidates["category_column"] = orig_col
                    break
            if candidates["category_column"]:
                break

        # Match product column (excluding already assigned category column)
        for cand in AppConfig.PRODUCT_COLUMN_CANDIDATES:
            for col_l, orig_col in cols_lower.items():
                if orig_col == candidates["category_column"]:
                    continue
                if cand in col_l:
                    candidates["product_column"] = orig_col
                    break
            if candidates["product_column"]:
                break

        # Match rating column
        for cand in AppConfig.RATING_COLUMN_CANDIDATES:
            for col_l, orig_col in cols_lower.items():
                if cand in col_l:
                    candidates["rating_column"] = orig_col
                    break
            if candidates["rating_column"]:
                break

        # Match date column
        for cand in AppConfig.DATE_COLUMN_CANDIDATES:
            for col_l, orig_col in cols_lower.items():
                if cand in col_l:
                    candidates["date_column"] = orig_col
                    break
            if candidates["date_column"]:
                break

        return candidates

    @staticmethod
    def process_dataframe(
        df: pd.DataFrame,
        text_column: str,
        engine: Optional[NLPEngine] = None,
        progress_callback=None
    ) -> pd.DataFrame:
        """
        Performs non-destructive batch NLP evaluation over the selected review column.
        Preserves original rows, detects empty or non-string inputs,
        and appends structured NLP scores.
        """
        if df.empty:
            return df

        nlp = engine or default_nlp_engine
        processed_df = df.copy()

        # Track row processing status
        statuses: List[str] = []
        clean_texts: List[str] = []
        polarities: List[float] = []
        subjectivities: List[float] = []
        tb_sentiments: List[str] = []
        vader_compounds: List[Optional[float]] = []
        vader_sentiments: List[Optional[str]] = []
        agreements: List[Optional[bool]] = []

        total_rows = len(processed_df)
        vader_avail = nlp.is_vader_available

        for idx, val in enumerate(processed_df[text_column]):
            if progress_callback and idx % 25 == 0:
                progress_callback(min(1.0, (idx + 1) / total_rows))

            if pd.isna(val) or not str(val).strip():
                statuses.append("Empty/Invalid")
                clean_texts.append("")
                polarities.append(0.0)
                subjectivities.append(0.0)
                tb_sentiments.append("Neutral")
                vader_compounds.append(0.0 if vader_avail else None)
                vader_sentiments.append("Neutral" if vader_avail else None)
                agreements.append(True if vader_avail else None)
            else:
                raw_str = str(val)
                # Clean html and whitespace for robust evaluation
                norm_str = normalize_whitespace(clean_html(raw_str))
                eval_text = norm_str if norm_str else raw_str

                res = nlp.analyze(eval_text, include_vader=vader_avail)
                statuses.append("Valid")
                clean_texts.append(norm_str)
                polarities.append(res.polarity)
                subjectivities.append(res.subjectivity)
                tb_sentiments.append(res.sentiment)
                vader_compounds.append(res.vader_compound)
                vader_sentiments.append(res.vader_sentiment)
                agreements.append(res.agreement)

        if progress_callback:
            progress_callback(1.0)

        processed_df["review_status"] = statuses
        processed_df["clean_text"] = clean_texts
        processed_df["tb_polarity"] = polarities
        processed_df["tb_subjectivity"] = subjectivities
        processed_df["tb_sentiment"] = tb_sentiments
        if vader_avail:
            processed_df["vader_compound"] = vader_compounds
            processed_df["vader_sentiment"] = vader_sentiments
            processed_df["analyzer_agreement"] = agreements

        logger.info(f"Batch processed {total_rows} reviews successfully.")
        gc.collect()

        return processed_df

    @classmethod
    def load_demonstration_data(cls) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Loads the bundled sample dataset from disk, executes NLP analysis,
        and returns the active dataframe and metadata with memory caching.
        """
        csv_path = SAMPLE_REVIEWS_CSV
        if not csv_path.exists():
            raise FileNotFoundError(f"Bundled demonstration dataset not found at: {csv_path}")

        try:
            # Check if streamlit cache is available in current execution context
            @st.cache_data(show_spinner=False)
            def _get_cached_demo_df():
                raw_df = pd.read_csv(csv_path)
                return cls.process_dataframe(raw_df, text_column="review_text")
            
            analyzed_df = _get_cached_demo_df().copy()
        except Exception:
            # Fallback for headless CLI or tests where Streamlit context is absent
            raw_df = pd.read_csv(csv_path)
            analyzed_df = cls.process_dataframe(raw_df, text_column="review_text")

        metadata = {
            "dataset_id": "demo_sample_dataset",
            "source": "Bundled Demonstration Dataset",
            "filename": "sample_reviews.csv",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_rows": len(analyzed_df),
            "valid_rows": int((analyzed_df["review_status"] == "Valid").sum()),
            "invalid_rows": int((analyzed_df["review_status"] != "Valid").sum()),
            "text_column": "review_text",
            "category_column": "category" if "category" in analyzed_df.columns else None,
            "product_column": "product_name" if "product_name" in analyzed_df.columns else None,
            "rating_column": "rating" if "rating" in analyzed_df.columns else None,
            "date_column": "date" if "date" in analyzed_df.columns else None,
            "is_demo": True
        }

        return analyzed_df, metadata

    @classmethod
    def ensure_active_dataset(cls) -> pd.DataFrame:
        """
        Ensures an active dataset is present in st.session_state.
        If absent, automatically initializes the demonstration dataset.
        Guarantees that Home, Dashboard, and Explorer never encounter empty state on launch.
        """
        if "active_data" not in st.session_state or st.session_state["active_data"] is None:
            df, meta = cls.load_demonstration_data()
            st.session_state["active_data"] = df
            st.session_state["dataset_metadata"] = meta

        return st.session_state["active_data"].copy()

    @classmethod
    def set_active_dataset(
        cls,
        df: pd.DataFrame,
        metadata: Dict[str, Any]
    ) -> None:
        """Atomically replaces the active dataset in session state."""
        st.session_state["active_data"] = df.copy()
        st.session_state["dataset_metadata"] = metadata.copy()

    @classmethod
    def restore_demonstration_data(cls) -> None:
        """Restores the initial demonstration dataset in session state."""
        df, meta = cls.load_demonstration_data()
        cls.set_active_dataset(df, meta)

    @classmethod
    def load_evaluation_dataset(cls) -> Optional[pd.DataFrame]:
        """Loads the ground-truth labeled evaluation dataset for academic benchmarking."""
        if EVAL_LABELED_CSV.exists():
            return pd.read_csv(EVAL_LABELED_CSV)
        return None
