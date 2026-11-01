"""
Unit tests for ReviewSense Data Management Layer.
Tests CSV parsing, multi-encoding support, schema detection,
batch processing, formula injection protection, and demonstration data loading.
"""

import pytest
import pandas as pd
from src.data_manager import (
    DataManager,
    sanitize_for_csv_injection,
    sanitize_dataframe_for_export
)
from src.config import SAMPLE_REVIEWS_CSV


def test_csv_formula_injection_sanitization():
    # Dangerous formula characters in strings
    assert sanitize_for_csv_injection("=cmd|'/C calc'!A0") == "'=cmd|'/C calc'!A0"
    assert sanitize_for_csv_injection("+123456") == "'+123456"
    assert sanitize_for_csv_injection("-SUM(A1:A10)") == "'-SUM(A1:A10)"
    assert sanitize_for_csv_injection("@HYPERLINK()") == "'@HYPERLINK()"

    # Legitimate numbers should NOT be modified
    assert sanitize_for_csv_injection(42) == 42
    assert sanitize_for_csv_injection(-3.14) == -3.14
    assert sanitize_for_csv_injection(0.05) == 0.05

    # Safe strings
    assert sanitize_for_csv_injection("Great product!") == "Great product!"


def test_dataframe_export_sanitization():
    df = pd.DataFrame({
        "review_id": ["REV-1", "REV-2"],
        "review_text": ["=1+1", "Normal review"],
        "rating": [5, 1],
        "polarity": [0.8, -0.5]
    })
    safe_df = sanitize_dataframe_for_export(df)
    assert safe_df.loc[0, "review_text"] == "'=1+1"
    assert safe_df.loc[1, "review_text"] == "Normal review"
    # Ensure numeric columns remain unchanged
    assert safe_df.loc[0, "rating"] == 5
    assert safe_df.loc[1, "polarity"] == -0.5


def test_load_csv_bytes_valid_utf8():
    csv_content = b"review_id,review_text,rating\nR1,Excellent product!,5\nR2,Terrible quality,1"
    df, err = DataManager.load_csv_bytes(csv_content, filename="test.csv")
    assert err is None
    assert df is not None
    assert len(df) == 2
    assert "review_text" in df.columns


def test_load_csv_bytes_utf8_bom():
    # UTF-8 with BOM prefix b'\xef\xbb\xbf'
    csv_content = b"\xef\xbb\xbfreview_id,review_text\nR1,Testing with BOM"
    df, err = DataManager.load_csv_bytes(csv_content, filename="bom.csv")
    assert err is None
    assert df is not None
    assert len(df) == 1
    assert "review_id" in df.columns


def test_load_csv_bytes_empty():
    df, err = DataManager.load_csv_bytes(b"", filename="empty.csv")
    assert df is None
    assert "empty" in err.lower()


def test_column_auto_detection():
    df = pd.DataFrame({
        "customer_review": ["text"],
        "product_category": ["audio"],
        "item_name": ["headphones"],
        "user_rating": [5],
        "created_at": ["2025-01-01"]
    })
    detected = DataManager.detect_candidate_columns(df)
    assert detected["text_column"] == "customer_review"
    assert detected["category_column"] == "product_category"
    assert detected["product_column"] == "item_name"
    assert detected["rating_column"] == "user_rating"
    assert detected["date_column"] == "created_at"


def test_demonstration_dataset_loads_and_scores():
    assert SAMPLE_REVIEWS_CSV.exists()
    df, meta = DataManager.load_demonstration_data()
    assert df is not None
    assert not df.empty
    assert len(df) >= 50
    assert "tb_polarity" in df.columns
    assert "tb_sentiment" in df.columns
    assert meta["source"] == "Bundled Demonstration Dataset"
    assert meta["valid_rows"] > 0
