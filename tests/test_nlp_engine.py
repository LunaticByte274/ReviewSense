"""
Unit tests for ReviewSense NLP Engine.
Tests TextBlob, VADER, score ranges, boundary thresholds, and edge-case inputs.
"""

import pytest
from src.nlp_engine import NLPEngine, default_nlp_engine
from src.config import SentimentThresholds


def test_positive_sentiment_detection():
    engine = default_nlp_engine
    res = engine.analyze("This product is absolutely amazing, wonderful, and works flawlessly!")
    assert res.sentiment == "Positive"
    assert res.polarity > 0.05
    assert -1.0 <= res.polarity <= 1.0
    assert 0.0 <= res.subjectivity <= 1.0


def test_negative_sentiment_detection():
    engine = default_nlp_engine
    res = engine.analyze("Terrible, broke on the first day. Horrible quality and awful support.")
    assert res.sentiment == "Negative"
    assert res.polarity < -0.05
    assert -1.0 <= res.polarity <= 1.0


def test_neutral_sentiment_detection():
    engine = default_nlp_engine
    res = engine.analyze("The package arrived on Tuesday and weighs 450 grams.")
    assert res.sentiment == "Neutral"
    assert -0.05 <= res.polarity <= 0.05


def test_empty_and_whitespace_inputs():
    engine = default_nlp_engine
    for empty_input in ["", "   ", "\n\t  ", None]:
        res = engine.analyze(empty_input)
        assert res.sentiment == "Neutral"
        assert res.polarity == 0.0
        assert res.subjectivity == 0.0


def test_exact_threshold_boundaries():
    engine = NLPEngine()
    # Exactly on positive boundary +0.05 -> Neutral
    assert engine.classify_tb_polarity(0.05) == "Neutral"
    # Just above positive boundary -> Positive
    assert engine.classify_tb_polarity(0.050001) == "Positive"
    
    # Exactly on negative boundary -0.05 -> Neutral
    assert engine.classify_tb_polarity(-0.05) == "Neutral"
    # Just below negative boundary -> Negative
    assert engine.classify_tb_polarity(-0.050001) == "Negative"

    # Middle zero
    assert engine.classify_tb_polarity(0.0) == "Neutral"


def test_vader_integration_and_agreement():
    engine = default_nlp_engine
    if engine.is_vader_available:
        res = engine.analyze("Great device, highly recommended!")
        assert res.vader_compound is not None
        assert res.vader_sentiment in ("Positive", "Negative", "Neutral")
        assert res.agreement is not None
        assert isinstance(res.agreement, bool)


def test_subjectivity_bounds():
    engine = default_nlp_engine
    res1 = engine.analyze("The table is made of oak wood.")
    res2 = engine.analyze("I truly believe this is the most gorgeous, breathtaking table in the world!")
    assert 0.0 <= res1.subjectivity <= 1.0
    assert 0.0 <= res2.subjectivity <= 1.0
    assert res2.subjectivity > res1.subjectivity

