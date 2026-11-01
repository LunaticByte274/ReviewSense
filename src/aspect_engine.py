"""
ReviewSense - Aspect-Based Sentiment Analysis (ABSA) Engine
Extracts granular product dimensions (Battery, Audio, Display, Build, Performance,
Price, Software, Customer Support) and calculates sentence-level aspect sentiment.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from textblob import TextBlob

from src.nlp_engine import default_nlp_engine

# Curated domain aspect definitions with keyword seeds
PRODUCT_ASPECTS = {
    "Battery & Power": [
        "battery", "charge", "charging", "charger", "power", "drain", "draining",
        "endurance", "mah", "watt", "adapter", "cord", "runtime"
    ],
    "Audio & Acoustics": [
        "sound", "audio", "bass", "treble", "volume", "speaker", "speakers",
        "headphones", "noise cancellation", "anc", "mic", "microphone", "clarity",
        "acoustic", "earbuds"
    ],
    "Display & Visuals": [
        "screen", "display", "oled", "panel", "brightness", "bezel", "flicker",
        "resolution", "4k", "hdr", "color", "colors", "viewing angle"
    ],
    "Build & Ergonomics": [
        "build", "quality", "durability", "durable", "hinge", "material", "materials",
        "plastic", "aluminum", "metal", "chassis", "craftsmanship", "sturdy", "weight",
        "heavy", "lightweight", "comfortable", "comfort"
    ],
    "Performance & Speed": [
        "performance", "speed", "fast", "slow", "lag", "laggy", "snappy", "processor",
        "cpu", "gpu", "ram", "thermal", "thermals", "heat", "hot", "fan", "cooling",
        "smooth"
    ],
    "Price & Value": [
        "price", "cost", "value", "worth", "expensive", "cheap", "overpriced",
        "bargain", "affordable", "deal", "money"
    ],
    "Software & App": [
        "software", "app", "ui", "interface", "firmware", "update", "updates",
        "bug", "bugs", "glitch", "crash", "crashes", "setup", "sync", "bluetooth"
    ],
    "Customer Support": [
        "support", "service", "customer service", "warranty", "refund", "replacement",
        "shipping", "delivery", "return", "returns", "rep", "agent"
    ]
}


class AspectEngine:
    """Performs clause-level aspect sentiment extraction and dataset aggregations."""

    @staticmethod
    def extract_aspects_from_text(text: str) -> Dict[str, Dict[str, Any]]:
        """
        Splits review into sentences/clauses, matches keywords to defined aspects,
        and computes sentiment specifically for the clause mentioning that aspect.
        """
        if not isinstance(text, str) or not text.strip():
            return {}

        results: Dict[str, Dict[str, Any]] = {}
        # Segment into sentences using regex boundary detection
        sentences = re.split(r"(?<=[.!?])\s+", text)

        for aspect_name, keywords in PRODUCT_ASPECTS.items():
            matched_sentences = []
            for s in sentences:
                s_lower = s.lower()
                for kw in keywords:
                    # Match whole word or key phrase
                    if re.search(r"\b" + re.escape(kw) + r"\b", s_lower):
                        matched_sentences.append(s.strip())
                        break

            if matched_sentences:
                combined_clause = " ".join(matched_sentences)
                nlp_res = default_nlp_engine.analyze(combined_clause)
                results[aspect_name] = {
                    "aspect": aspect_name,
                    "sentiment": nlp_res.sentiment,
                    "polarity": nlp_res.polarity,
                    "subjectivity": nlp_res.subjectivity,
                    "snippet": matched_sentences[0] if len(matched_sentences) == 1 else combined_clause,
                    "mention_count": len(matched_sentences)
                }

        return results

    @classmethod
    def analyze_dataset_aspects(cls, df: pd.DataFrame, text_col: str = "review_text") -> pd.DataFrame:
        """
        Scans all reviews in dataframe, aggregates aspect mentions,
        and calculates aspect frequency, positive/negative count, and mean polarity.
        """
        if df is None or df.empty or text_col not in df.columns:
            return pd.DataFrame()

        valid_df = df[df["review_status"] == "Valid"] if "review_status" in df.columns else df
        if valid_df.empty:
            return pd.DataFrame()

        aspect_stats = {
            name: {
                "aspect": name,
                "mentions": 0,
                "positive": 0,
                "negative": 0,
                "neutral": 0,
                "polarity_sum": 0.0
            }
            for name in PRODUCT_ASPECTS.keys()
        }

        for text in valid_df[text_col]:
            if isinstance(text, str) and text.strip():
                extracted = cls.extract_aspects_from_text(text)
                for aspect_name, data in extracted.items():
                    stats = aspect_stats[aspect_name]
                    stats["mentions"] += 1
                    stats["polarity_sum"] += data["polarity"]
                    if data["sentiment"] == "Positive":
                        stats["positive"] += 1
                    elif data["sentiment"] == "Negative":
                        stats["negative"] += 1
                    else:
                        stats["neutral"] += 1

        rows = []
        for name, stats in aspect_stats.items():
            if stats["mentions"] > 0:
                avg_pol = round(stats["polarity_sum"] / stats["mentions"], 3)
                pos_rate = round((stats["positive"] / stats["mentions"]) * 100, 1)
                neg_rate = round((stats["negative"] / stats["mentions"]) * 100, 1)
                rows.append({
                    "aspect": name,
                    "total_mentions": stats["mentions"],
                    "positive_count": stats["positive"],
                    "negative_count": stats["negative"],
                    "neutral_count": stats["neutral"],
                    "avg_polarity": avg_pol,
                    "positive_rate": pos_rate,
                    "negative_rate": neg_rate,
                    # Net Aspect Score = Pos Rate - Neg Rate
                    "net_aspect_score": round(pos_rate - neg_rate, 1)
                })

        res_df = pd.DataFrame(rows)
        if not res_df.empty:
            return res_df.sort_values(by="total_mentions", ascending=False)
        return res_df


# Global instance
default_aspect_engine = AspectEngine()

