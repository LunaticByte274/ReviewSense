"""
ReviewSense - Central Application Configuration
Defines default thresholds, file limits, design tokens, and project paths
with full 12-factor environment variable override support for production.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any

# Base Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"
SAMPLE_REVIEWS_CSV = SAMPLE_DATA_DIR / "sample_reviews.csv"
EVAL_LABELED_CSV = SAMPLE_DATA_DIR / "eval_labeled_reviews.csv"


@dataclass(frozen=True)
class SentimentThresholds:
    """Configurable sentiment polarity and compound score thresholds."""
    # TextBlob polarity thresholds: [-1.0, 1.0]
    TB_POSITIVE_MIN: float = float(os.getenv("REVIEWSENSE_TB_POS_MIN", "0.05"))
    TB_NEGATIVE_MAX: float = float(os.getenv("REVIEWSENSE_TB_NEG_MAX", "-0.05"))

    # VADER compound thresholds: [-1.0, 1.0]
    VADER_POSITIVE_MIN: float = float(os.getenv("REVIEWSENSE_VADER_POS_MIN", "0.05"))
    VADER_NEGATIVE_MAX: float = float(os.getenv("REVIEWSENSE_VADER_NEG_MAX", "-0.05"))

    # Subjectivity thresholds: [0.0, 1.0]
    SUBJECTIVITY_OBJECTIVE_MAX: float = float(os.getenv("REVIEWSENSE_SUBJ_OBJ_MAX", "0.35"))
    SUBJECTIVITY_SUBJECTIVE_MIN: float = float(os.getenv("REVIEWSENSE_SUBJ_SUB_MIN", "0.65"))


@dataclass(frozen=True)
class AppConfig:
    """Application-wide settings, environment variables, and resource boundaries."""
    APP_NAME: str = os.getenv("REVIEWSENSE_APP_NAME", "ReviewSense")
    APP_TAGLINE: str = "Enterprise-Grade Product Review Intelligence & NLP Analytics"
    APP_VERSION: str = "2.1.0"
    ENVIRONMENT: str = os.getenv("REVIEWSENSE_ENV", "production")  # 'development', 'staging', 'production'
    
    # File upload limits
    MAX_FILE_SIZE_MB: int = int(os.getenv("REVIEWSENSE_MAX_FILE_SIZE_MB", "25"))
    MAX_ROWS_RECOMMENDED: int = int(os.getenv("REVIEWSENSE_MAX_ROWS_RECOMMENDED", "15000"))
    SUPPORTED_EXTENSIONS: tuple = (".csv",)
    
    # Common column candidate names for auto-detection (ordered from most specific to general)
    TEXT_COLUMN_CANDIDATES: tuple = (
        "review_text", "reviewtext", "customer_review", "feedback", "comment",
        "comments", "body", "content", "summary", "text", "review"
    )
    CATEGORY_COLUMN_CANDIDATES: tuple = (
        "category", "product_category", "department", "type", "segment"
    )
    PRODUCT_COLUMN_CANDIDATES: tuple = (
        "product", "product_name", "item", "item_name", "title", "asin"
    )
    RATING_COLUMN_CANDIDATES: tuple = (
        "rating", "stars", "score", "overall", "user_rating"
    )
    DATE_COLUMN_CANDIDATES: tuple = (
        "date", "review_date", "timestamp", "created_at", "time"
    )


# Color Palette and Design System Tokens
THEME_TOKENS: Dict[str, Any] = {
    "positive_color": "#10B981",    # Emerald 500
    "positive_light": "rgba(16, 185, 129, 0.15)",
    "negative_color": "#F43F5E",    # Rose 500
    "negative_light": "rgba(244, 63, 94, 0.15)",
    "neutral_color": "#94A3B8",     # Slate 400
    "neutral_light": "rgba(148, 163, 184, 0.15)",
    "accent_primary": "#6366F1",    # Indigo 500
    "accent_secondary": "#06B6D4",  # Cyan 500
    "card_bg": "#1E293B",           # Slate 800
    "border_color": "rgba(255, 255, 255, 0.08)",
    "text_primary": "#F8FAFC",      # Slate 50
    "text_secondary": "#94A3B8",    # Slate 400
}

# Negation words that MUST be preserved during NLP preprocessing
NEGATION_WORDS: frozenset = frozenset([
    "not", "no", "never", "none", "neither", "nor", "nowhere", "hardly",
    "scarcely", "barely", "cannot", "can't", "don't", "doesn't", "didn't",
    "won't", "wouldn't", "shouldn't", "couldn't", "isn't", "aren't",
    "wasn't", "weren't", "hasn't", "haven't", "hadn't", "without"
])
