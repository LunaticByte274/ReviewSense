"""
Unit tests for ReviewSense Analytics Layer.
Tests KPI calculations, denominators, category aggregations,
extremes extraction, and academic model evaluation metrics.
"""

import pytest
import pandas as pd
from src.analytics import AnalyticsEngine


def test_kpi_calculations_exact():
    # Construct controlled test dataset with known manual expected values
    test_df = pd.DataFrame({
        "review_id": [1, 2, 3, 4],
        "tb_sentiment": ["Positive", "Positive", "Negative", "Neutral"],
        "tb_polarity": [0.8, 0.4, -0.6, 0.0],
        "tb_subjectivity": [0.6, 0.4, 0.8, 0.2],
        "review_status": ["Valid", "Valid", "Valid", "Valid"]
    })

    kpis = AnalyticsEngine.calculate_kpis(test_df)

    assert kpis["total_reviews"] == 4
    assert kpis["valid_reviews"] == 4
    assert kpis["positive_count"] == 2
    assert kpis["negative_count"] == 1
    assert kpis["neutral_count"] == 1
    assert kpis["positive_pct"] == 50.0
    assert kpis["negative_pct"] == 25.0
    assert kpis["neutral_pct"] == 25.0
    # Mean polarity: (0.8 + 0.4 - 0.6 + 0.0) / 4 = 0.6 / 4 = 0.15
    assert abs(kpis["avg_polarity"] - 0.15) < 1e-4
    # Mean subjectivity: (0.6 + 0.4 + 0.8 + 0.2) / 4 = 2.0 / 4 = 0.5
    assert abs(kpis["avg_subjectivity"] - 0.5) < 1e-4


def test_kpi_empty_dataframe():
    empty_df = pd.DataFrame()
    kpis = AnalyticsEngine.calculate_kpis(empty_df)
    assert kpis["total_reviews"] == 0
    assert kpis["valid_reviews"] == 0
    assert kpis["positive_count"] == 0
    assert kpis["avg_polarity"] == 0.0


def test_category_breakdown():
    df = pd.DataFrame({
        "category": ["Audio", "Audio", "Computing"],
        "tb_sentiment": ["Positive", "Negative", "Positive"],
        "tb_polarity": [0.8, -0.4, 0.5],
        "review_status": ["Valid", "Valid", "Valid"]
    })
    cat_df = AnalyticsEngine.get_category_breakdown(df, category_col="category")
    assert cat_df is not None
    assert len(cat_df) == 2
    audio_row = cat_df[cat_df["category"] == "Audio"].iloc[0]
    assert audio_row["total_reviews"] == 2
    assert audio_row["positive"] == 1
    assert audio_row["negative"] == 1


def test_ground_truth_model_evaluation():
    eval_df = pd.DataFrame({
        "true_sentiment": ["Positive", "Positive", "Negative", "Neutral", "Negative"],
        "pred_sentiment": ["Positive", "Neutral", "Negative", "Neutral", "Negative"]
    })
    results = AnalyticsEngine.evaluate_model_on_ground_truth(
        eval_df,
        ground_truth_col="true_sentiment",
        pred_col="pred_sentiment"
    )
    assert results["status"] == "success"
    assert results["total_samples"] == 5
    # Correct predictions: 4 out of 5 = 0.80
    assert results["accuracy"] == 0.8
    assert results["confusion_matrix"] is not None
    assert results["macro_f1"] > 0.0
