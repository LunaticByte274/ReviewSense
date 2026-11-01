"""
Unit tests for ReviewSense Executive Intelligence Report Generator.
Validates HTML report structure, Net Sentiment Score, and aspect integration.
"""

import pytest
import pandas as pd
from src.report_generator import ReportGenerator


def test_html_report_generation():
    df = pd.DataFrame({
        "product_name": ["ZenBook Pro", "ZenBook Pro"],
        "review_text": [
            "Outstanding performance and fast M-series chip!",
            "Terrible keyboard and awful customer support."
        ],
        "tb_sentiment": ["Positive", "Negative"],
        "tb_polarity": [0.8, -0.6],
        "tb_subjectivity": [0.6, 0.7],
        "review_status": ["Valid", "Valid"]
    })
    meta = {
        "source": "Test Ingestion",
        "filename": "test.csv",
        "total_rows": 2
    }

    html = ReportGenerator.generate_html_report(df, meta, text_col="review_text")
    assert "<!DOCTYPE html>" in html
    assert "ReviewSense Executive Report" in html
    assert "Net Sentiment Score" in html
    assert "ZenBook Pro" in html
    assert "Performance & Speed" in html or "Customer Support" in html
