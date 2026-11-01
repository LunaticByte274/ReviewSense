"""
Unit tests for ReviewSense Aspect-Based Sentiment Analysis (ABSA) Engine.
Validates multi-aspect extraction, clause scoping, and dataset aggregations.
"""

import pytest
import pandas as pd
from src.aspect_engine import AspectEngine, default_aspect_engine


def test_mixed_aspect_clause_extraction():
    engine = default_aspect_engine
    text = "The audio sound is crystal clear and magnificent. However, the battery life is completely terrible and drains fast."
    extracted = engine.extract_aspects_from_text(text)

    assert "Audio & Acoustics" in extracted
    assert "Battery & Power" in extracted

    audio_res = extracted["Audio & Acoustics"]
    assert audio_res["sentiment"] == "Positive"
    assert audio_res["polarity"] > 0.05

    battery_res = extracted["Battery & Power"]
    assert battery_res["sentiment"] == "Negative"
    assert battery_res["polarity"] < -0.05


def test_aspect_dataset_aggregation():
    df = pd.DataFrame({
        "review_text": [
            "Superb screen and OLED display!",
            "Terrible display and awful screen flicker.",
            "Great battery lasting all week."
        ],
        "review_status": ["Valid", "Valid", "Valid"]
    })
    aspect_df = AspectEngine.analyze_dataset_aspects(df, text_col="review_text")
    assert not aspect_df.empty
    assert "Display & Visuals" in aspect_df["aspect"].values
    disp_row = aspect_df[aspect_df["aspect"] == "Display & Visuals"].iloc[0]
    assert disp_row["total_mentions"] == 2
    assert disp_row["positive_count"] == 1
    assert disp_row["negative_count"] == 1

