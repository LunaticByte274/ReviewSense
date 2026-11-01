"""
Unit tests for the Data Visualization layer (src.visualizations).
Guarantees that all Plotly figures render without error, handle missing or
empty datasets gracefully, and do not raise multiple keyword argument exceptions.
"""

import pandas as pd
import pytest
import plotly.graph_objects as go

from src.visualizations import (
    create_sentiment_donut_chart,
    create_sentiment_bar_chart,
    create_polarity_histogram,
    create_subjectivity_distribution,
    create_category_sentiment_chart,
    create_rating_vs_sentiment_chart,
    create_timeline_trend_chart,
    create_confusion_matrix_heatmap,
    create_polarity_gauge,
    create_aspect_sentiment_chart,
    create_ngram_bar_chart,
    create_rating_polarity_scatter,
    create_aspect_radar_chart,
    create_aspect_treemap,
    create_word_cloud_bubble_chart,
    CHART_LAYOUT_DEFAULTS,
)


@pytest.fixture
def sample_viz_dataframe():
    return pd.DataFrame({
        "tb_sentiment": ["Positive", "Negative", "Neutral", "Positive", "Negative"],
        "tb_polarity": [0.85, -0.72, 0.02, 0.45, -0.15],
        "tb_subjectivity": [0.60, 0.40, 0.10, 0.80, 0.35],
        "rating": [5, 1, 3, 4, 2],
        "category": ["Audio", "Audio", "Wearables", "Audio", "Wearables"],
        "date_str": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"],
        "review_count": [12, 18, 5, 22, 9],
        "avg_polarity": [0.35, -0.20, 0.05, 0.40, -0.10],
        "aspect": ["battery", "sound", "price", "build", "comfort"],
        "positive_count": [10, 15, 4, 8, 12],
        "negative_count": [2, 3, 1, 14, 2],
        "positive": [10, 15, 4, 8, 12],
        "neutral": [2, 1, 1, 2, 1],
        "negative": [2, 3, 1, 14, 2],
        "positive_rate": [83.3, 83.3, 80.0, 36.4, 85.7],
        "total_mentions": [14, 19, 6, 24, 15],
        "net_aspect_score": [57.1, 63.2, 50.0, -25.0, 66.7],
    })


def test_polarity_gauge_rendering():
    """Verify create_polarity_gauge renders correctly without duplicate keyword argument errors."""
    for val in [-1.0, -0.5, 0.0, 0.05, 0.5, 1.0]:
        fig = create_polarity_gauge(val)
        assert isinstance(fig, go.Figure)
        assert fig.data
        assert fig.data[0].type == "indicator"


def test_donut_chart(sample_viz_dataframe):
    fig = create_sentiment_donut_chart(sample_viz_dataframe)
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "pie"

    # Empty df fallback
    fig_empty = create_sentiment_donut_chart(pd.DataFrame())
    assert isinstance(fig_empty, go.Figure)


def test_bar_chart(sample_viz_dataframe):
    fig = create_sentiment_bar_chart(sample_viz_dataframe)
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "bar"

    fig_empty = create_sentiment_bar_chart(None)
    assert isinstance(fig_empty, go.Figure)


def test_polarity_histogram(sample_viz_dataframe):
    fig = create_polarity_histogram(sample_viz_dataframe)
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "histogram"

    fig_empty = create_polarity_histogram(pd.DataFrame())
    assert isinstance(fig_empty, go.Figure)


def test_subjectivity_distribution(sample_viz_dataframe):
    fig = create_subjectivity_distribution(sample_viz_dataframe)
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "histogram"

    fig_empty = create_subjectivity_distribution(pd.DataFrame())
    assert isinstance(fig_empty, go.Figure)


def test_category_sentiment_chart(sample_viz_dataframe):
    cat_df = sample_viz_dataframe[["category", "positive", "neutral", "negative"]]
    fig = create_category_sentiment_chart(cat_df)
    assert isinstance(fig, go.Figure)

    fig_empty = create_category_sentiment_chart(pd.DataFrame())
    assert isinstance(fig_empty, go.Figure)


def test_rating_vs_sentiment_chart(sample_viz_dataframe):
    rating_df = sample_viz_dataframe[["rating", "positive", "neutral", "negative"]]
    fig = create_rating_vs_sentiment_chart(rating_df)
    assert isinstance(fig, go.Figure)

    fig_empty = create_rating_vs_sentiment_chart(pd.DataFrame())
    assert isinstance(fig_empty, go.Figure)


def test_timeline_trend_chart(sample_viz_dataframe):
    trend_df = sample_viz_dataframe[["date_str", "review_count", "avg_polarity"]]
    fig = create_timeline_trend_chart(trend_df)
    assert isinstance(fig, go.Figure)

    fig_empty = create_timeline_trend_chart(pd.DataFrame())
    assert isinstance(fig_empty, go.Figure)


def test_confusion_matrix_heatmap():
    cm_df = pd.DataFrame([[10, 2, 1], [1, 15, 0], [0, 1, 12]], index=["Pos", "Neu", "Neg"], columns=["Pos", "Neu", "Neg"])
    fig = create_confusion_matrix_heatmap(cm_df, ["Pos", "Neu", "Neg"])
    assert isinstance(fig, go.Figure)

    fig_empty = create_confusion_matrix_heatmap(pd.DataFrame(), [])
    assert isinstance(fig_empty, go.Figure)


def test_aspect_sentiment_chart(sample_viz_dataframe):
    fig = create_aspect_sentiment_chart(sample_viz_dataframe)
    assert isinstance(fig, go.Figure)

    fig_empty = create_aspect_sentiment_chart(pd.DataFrame())
    assert isinstance(fig_empty, go.Figure)


def test_ngram_bar_chart():
    ngrams = [("battery life", 45), ("sound quality", 38), ("noise cancellation", 29)]
    fig = create_ngram_bar_chart(ngrams, "Top Bigrams", "#6366F1")
    assert isinstance(fig, go.Figure)

    fig_empty = create_ngram_bar_chart([], "Empty", "#6366F1")
    assert isinstance(fig_empty, go.Figure)


def test_rating_polarity_scatter(sample_viz_dataframe):
    fig = create_rating_polarity_scatter(sample_viz_dataframe)
    assert isinstance(fig, go.Figure)

    fig_empty = create_rating_polarity_scatter(pd.DataFrame())
    assert isinstance(fig_empty, go.Figure)


def test_aspect_radar_chart(sample_viz_dataframe):
    fig = create_aspect_radar_chart(sample_viz_dataframe)
    assert isinstance(fig, go.Figure)

    fig_empty = create_aspect_radar_chart(pd.DataFrame())
    assert isinstance(fig_empty, go.Figure)


def test_aspect_treemap(sample_viz_dataframe):
    fig = create_aspect_treemap(sample_viz_dataframe)
    assert isinstance(fig, go.Figure)

    fig_empty = create_aspect_treemap(pd.DataFrame())
    assert isinstance(fig_empty, go.Figure)


def test_word_cloud_bubble_chart():
    words = [("crystal", 25), ("clear", 20), ("comfortable", 18), ("durable", 15)]
    fig = create_word_cloud_bubble_chart(words)
    assert isinstance(fig, go.Figure)

    fig_empty = create_word_cloud_bubble_chart([])
    assert isinstance(fig_empty, go.Figure)
