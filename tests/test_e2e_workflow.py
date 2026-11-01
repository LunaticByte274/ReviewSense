"""
Integration and End-to-End Workflow Tests for ReviewSense.
Simulates end-to-end user journeys:
1. Initialize demonstration data.
2. Run single review inference.
3. Process bulk CSV dataset with custom columns.
4. Execute cross-filtering and metric calculations.
5. Generate injection-safe CSV export.
"""

import io
import pytest
import pandas as pd
from src.data_manager import DataManager, sanitize_dataframe_for_export
from src.nlp_engine import default_nlp_engine
from src.analytics import AnalyticsEngine


def test_e2e_full_pipeline_flow():
    # 1. Load demonstration dataset
    demo_df, demo_meta = DataManager.load_demonstration_data()
    assert demo_df is not None
    assert len(demo_df) >= 50
    assert demo_meta["is_demo"] is True

    # 2. Verify single review inference
    test_review = "The camera has extraordinary low light sensitivity, but battery drains quickly."
    res = default_nlp_engine.analyze(test_review, include_vader=True)
    assert res.sentiment in ("Positive", "Negative", "Neutral")
    assert res.polarity != 0.0
    assert res.subjectivity > 0.0

    # 3. Simulate user CSV upload
    custom_csv = (
        "id,feedback,dept,user_score\n"
        "101,Unbelievable performance and great battery life!,Hardware,5\n"
        "102,Awful experience broke after one hour of use,Hardware,1\n"
        "103,Package received with instruction sheet,Shipping,3\n"
    ).encode("utf-8")

    user_df, err = DataManager.load_csv_bytes(custom_csv, filename="user_reviews.csv")
    assert err is None
    assert len(user_df) == 3

    # Detect candidate columns
    candidates = DataManager.detect_candidate_columns(user_df)
    assert candidates["text_column"] == "feedback"

    # Batch process
    analyzed_user_df = DataManager.process_dataframe(user_df, text_column="feedback")
    assert "tb_sentiment" in analyzed_user_df.columns
    assert "tb_polarity" in analyzed_user_df.columns

    # 4. Analytics on processed user data
    kpis = AnalyticsEngine.calculate_kpis(analyzed_user_df)
    assert kpis["total_reviews"] == 3
    assert kpis["positive_count"] == 1
    assert kpis["negative_count"] == 1
    assert kpis["neutral_count"] == 1

    # 5. Export sanitization
    safe_export = sanitize_dataframe_for_export(analyzed_user_df)
    csv_out = safe_export.to_csv(index=False)
    assert "feedback" in csv_out
    assert "tb_sentiment" in csv_out

