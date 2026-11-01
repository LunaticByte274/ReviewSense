"""
ReviewSense - Natural Language Processing Preprocessing Module
Implements robust, non-destructive text cleaning and tokenization
with strict negation preservation and step-by-step transformation tracking.
"""

import re
import html
from typing import List, Dict, Any, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from src.config import NEGATION_WORDS

# Lazy-loaded stopword cache
_STOPWORDS_CACHE = None


def get_negation_aware_stopwords() -> set:
    """
    Returns standard NLTK English stopwords minus critical negation words.
    This prevents sentiment inversion during stopword filtering.
    """
    global _STOPWORDS_CACHE
    if _STOPWORDS_CACHE is None:
        try:
            base_stopwords = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            base_stopwords = set(stopwords.words("english"))
        # Exclude negation words from stopword list
        _STOPWORDS_CACHE = base_stopwords - NEGATION_WORDS
    return _STOPWORDS_CACHE


def clean_html(text: str) -> str:
    """Removes HTML tags and unescapes HTML entities."""
    if not isinstance(text, str):
        return ""
    # Unescape HTML entities (e.g. &amp; -> &, &quot; -> ")
    decoded = html.unescape(text)
    # Strip HTML tags
    cleaned = re.sub(r"<[^>]+>", " ", decoded)
    return cleaned


def normalize_whitespace(text: str) -> str:
    """Normalizes consecutive whitespace, tabs, and newlines into single spaces."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def remove_punctuation(text: str, keep_emojis: bool = True) -> str:
    """
    Normalizes punctuation while optionally preserving expressive characters or emojis.
    Keeps apostrophes within contractions (e.g., don't, can't).
    """
    if not isinstance(text, str):
        return ""
    # Remove standard punctuation except internal apostrophes
    # Replaces standalone punctuation with space
    pattern = r"[^\w\s\']|(?<=\s)\'|\'(?=\s)"
    cleaned = re.sub(pattern, " ", text)
    return normalize_whitespace(cleaned)


def tokenize_text(text: str) -> List[str]:
    """Tokenizes text into individual words using NLTK word_tokenize with fallback."""
    if not isinstance(text, str) or not text.strip():
        return []
    try:
        tokens = word_tokenize(text)
    except Exception:
        # Fallback regex tokenization if punkt tokenizer resource is missing
        tokens = re.findall(r"\b\w+(?:'\w+)?\b", text)
    return tokens


def remove_stopwords_from_tokens(
    tokens: List[str],
    preserve_negations: bool = True
) -> List[str]:
    """
    Filters stopwords from tokens.
    If preserve_negations is True, negation words (not, never, no, etc.) are kept.
    """
    if not tokens:
        return []
    
    if preserve_negations:
        active_stopwords = get_negation_aware_stopwords()
    else:
        try:
            active_stopwords = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            active_stopwords = set(stopwords.words("english"))

    return [t for t in tokens if t.lower() not in active_stopwords]


def preprocess_review(
    text: str,
    strip_html: bool = True,
    normalize_spaces: bool = True,
    to_lower: bool = False,
    strip_punct: bool = False,
    filter_stopwords: bool = False,
    preserve_negations: bool = True
) -> str:
    """
    Comprehensive text preprocessing pipeline for product reviews.
    Maintains semantic integrity while allowing selective transformations.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    result = text
    if strip_html:
        result = clean_html(result)

    if strip_punct:
        result = remove_punctuation(result)

    if to_lower:
        result = result.lower()

    if filter_stopwords:
        tokens = tokenize_text(result)
        filtered_tokens = remove_stopwords_from_tokens(
            tokens, preserve_negations=preserve_negations
        )
        result = " ".join(filtered_tokens)

    if normalize_spaces:
        result = normalize_whitespace(result)

    return result


def get_step_by_step_transformations(
    text: str,
    preserve_negations: bool = True
) -> List[Dict[str, Any]]:
    """
    Generates step-by-step transformations of the input text for educational
    and inspection purposes in the Text Preprocessing page.
    """
    steps = []
    
    current = text if isinstance(text, str) else ""
    steps.append({
        "step_num": 1,
        "name": "Original Raw Text",
        "description": "Input text as submitted by user or loaded from dataset.",
        "text": current,
        "token_count": len(current.split()) if current else 0
    })

    # Step 2: HTML Cleaning & Entity Decoding
    after_html = clean_html(current)
    steps.append({
        "step_num": 2,
        "name": "HTML & Entity Normalization",
        "description": "Strips HTML tags (<p>, <br>) and unescapes entities (&amp; -> &, &quot; -> \").",
        "text": after_html,
        "token_count": len(after_html.split()) if after_html else 0
    })

    # Step 3: Whitespace Normalization
    after_ws = normalize_whitespace(after_html)
    steps.append({
        "step_num": 3,
        "name": "Whitespace Normalization",
        "description": "Collapses redundant tabs, line breaks, and repeated spaces into uniform single spaces.",
        "text": after_ws,
        "token_count": len(after_ws.split()) if after_ws else 0
    })

    # Step 4: Lowercase Conversion
    after_lower = after_ws.lower()
    steps.append({
        "step_num": 4,
        "name": "Lowercasing",
        "description": "Converts all characters to lowercase to standardize lexical matching.",
        "text": after_lower,
        "token_count": len(after_lower.split()) if after_lower else 0
    })

    # Step 5: Punctuation Handling
    after_punct = remove_punctuation(after_lower)
    steps.append({
        "step_num": 5,
        "name": "Punctuation Normalization",
        "description": "Removes symbols while preserving internal contraction apostrophes (don't, isn't).",
        "text": after_punct,
        "token_count": len(after_punct.split()) if after_punct else 0
    })

    # Step 6: Tokenization
    tokens = tokenize_text(after_punct)
    steps.append({
        "step_num": 6,
        "name": "Word Tokenization",
        "description": "Decomposes the character stream into discrete word tokens.",
        "text": " | ".join(tokens),
        "token_count": len(tokens),
        "is_tokens": True,
        "tokens": tokens
    })

    # Step 7: Stopword Removal (with Negation Preservation)
    neg_filtered_tokens = remove_stopwords_from_tokens(tokens, preserve_negations=preserve_negations)
    preserved_negs = [t for t in neg_filtered_tokens if t in NEGATION_WORDS]
    steps.append({
        "step_num": 7,
        "name": "Negation-Aware Stopword Removal",
        "description": (
            "Eliminates uninformative words (the, is, at) while strictly protecting sentiment-critical negations "
            f"(e.g., {', '.join(sorted(list(NEGATION_WORDS)[:6]))}...)."
        ),
        "text": " ".join(neg_filtered_tokens),
        "token_count": len(neg_filtered_tokens),
        "is_tokens": True,
        "tokens": neg_filtered_tokens,
        "preserved_negations": preserved_negs
    })

    return steps
