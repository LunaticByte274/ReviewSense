"""
Unit and integration tests for ReviewSense AI Copilot Engine (src.copilot_engine).
Validates natural language intent routing, direct review scoring, aspect analytics,
KPI summaries, anomaly detection, and defensive empty/invalid input handling.
"""

import pandas as pd
import pytest

from src.copilot_engine import ReviewCopilotEngine, default_copilot_engine
from src.data_manager import DataManager


@pytest.fixture
def copilot_fixture_df():
    """Returns a realistic scored sample DataFrame."""
    df, _ = DataManager.load_demonstration_data()
    return df


def test_copilot_empty_query():
    copilot = ReviewCopilotEngine()
    resp = copilot.process_query("")
    assert "Please enter a question" in resp


def test_copilot_direct_review_scoring():
    copilot = ReviewCopilotEngine()
    resp = copilot.process_query("Analyze: The sound quality is mind-blowing, but battery life is mediocre.")
    assert "Real-Time Review Scoring Result" in resp
    assert "Polarity Score" in resp
    assert "Sound & Acoustics" in resp or "Audio & Acoustics" in resp or "Battery & Power" in resp


def test_copilot_kpi_overview_query(copilot_fixture_df):
    copilot = ReviewCopilotEngine()
    resp = copilot.process_query("What is our Net Sentiment Score and KPI health?", df=copilot_fixture_df)
    assert "Executive Sentiment Health" in resp
    assert "Net Sentiment Score (NSS)" in resp
    assert "Total Ingested Reviews" in resp


def test_copilot_aspect_query(copilot_fixture_df):
    copilot = ReviewCopilotEngine()
    resp = copilot.process_query("What are users saying about battery life?", df=copilot_fixture_df)
    assert "Aspect Intelligence Deep-Dive" in resp
    assert "Battery & Power" in resp
    assert "Net Aspect Score" in resp


def test_copilot_complaint_drivers_query(copilot_fixture_df):
    copilot = ReviewCopilotEngine()
    resp = copilot.process_query("What are the biggest customer complaints and what to fix?", df=copilot_fixture_df)
    assert "Top Negative Drivers" in resp


def test_copilot_strengths_query(copilot_fixture_df):
    copilot = ReviewCopilotEngine()
    resp = copilot.process_query("What are our top strengths and what do users love?", df=copilot_fixture_df)
    assert "Top Product Strengths" in resp


def test_copilot_anomalies_query(copilot_fixture_df):
    copilot = ReviewCopilotEngine()
    resp = copilot.process_query("Find any rating anomalies or suspicious discrepancies", df=copilot_fixture_df)
    assert "Anomaly Report" in resp or "No Rating-Sentiment Anomalies" in resp


def test_copilot_category_comparison(copilot_fixture_df):
    copilot = ReviewCopilotEngine()
    resp = copilot.process_query("Compare category performance", df=copilot_fixture_df)
    assert "Category Sentiment Comparison" in resp


def test_copilot_empty_dataframe_guard():
    copilot = ReviewCopilotEngine()
    resp = copilot.process_query("What is our Net Sentiment Score?", df=pd.DataFrame())
    assert "No Active Dataset Found" in resp


def test_copilot_tuple_input_handling(copilot_fixture_df):
    copilot = ReviewCopilotEngine()
    # Handle tuple if caller passed (df, metadata)
    resp = copilot.process_query("What is our Net Sentiment Score?", df=(copilot_fixture_df, {}))
    assert "Executive Sentiment Health" in resp
