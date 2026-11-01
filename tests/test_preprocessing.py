"""
Unit tests for ReviewSense NLP Preprocessing Pipeline.
Validates HTML decoding, whitespace normalization, tokenization,
and strict negation preservation during stopword filtering.
"""

import pytest
from src.preprocessing import (
    clean_html,
    normalize_whitespace,
    remove_punctuation,
    tokenize_text,
    remove_stopwords_from_tokens,
    preprocess_review,
    get_step_by_step_transformations
)
from src.config import NEGATION_WORDS


def test_html_tag_and_entity_removal():
    raw = "<div><p>Great sound &amp; awesome battery! Price was &gt; $200.</p></div>"
    cleaned = clean_html(raw)
    assert "<p>" not in cleaned
    assert "<div>" not in cleaned
    assert "&amp;" not in cleaned
    assert "&" in cleaned
    assert ">" in cleaned


def test_whitespace_normalization():
    raw = "  This   review   has \n\n multiple \t tabs   and spaces.  "
    normalized = normalize_whitespace(raw)
    assert normalized == "This review has multiple tabs and spaces."


def test_punctuation_removal_preserves_contractions():
    raw = "It isn't working, and I can't fix it! Why???"
    cleaned = remove_punctuation(raw)
    assert "!" not in cleaned
    assert "?" not in cleaned
    assert "isn't" in cleaned or "isnt" in cleaned
    assert "can't" in cleaned or "cant" in cleaned


def test_tokenization():
    raw = "The quick brown fox jumps over the lazy dog."
    tokens = tokenize_text(raw)
    # NLTK word_tokenize splits punctuation as a token (9 words + 1 period = 10)
    assert len(tokens) == 10
    assert tokens[0] == "The"
    assert tokens[-2] == "dog"
    assert tokens[-1] == "."


def test_negation_preservation_in_stopwords():
    tokens = ["the", "battery", "is", "not", "working", "and", "never", "charges", "without", "heat"]
    
    # 1. With negation preservation (Default)
    preserved = remove_stopwords_from_tokens(tokens, preserve_negations=True)
    assert "not" in preserved
    assert "never" in preserved
    assert "without" in preserved
    assert "the" not in preserved
    assert "is" not in preserved
    assert "and" not in preserved

    # 2. Without negation preservation (Naive baseline)
    naive = remove_stopwords_from_tokens(tokens, preserve_negations=False)
    assert "not" not in naive


def test_step_by_step_transformations():
    raw = "<p>It is not good!</p>"
    steps = get_step_by_step_transformations(raw, preserve_negations=True)
    assert len(steps) == 7
    step_names = [s["name"] for s in steps]
    assert "Original Raw Text" in step_names
    assert "HTML & Entity Normalization" in step_names
    assert "Negation-Aware Stopword Removal" in step_names
