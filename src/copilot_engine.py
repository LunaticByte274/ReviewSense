"""
ReviewSense - AI Review Intelligence Copilot Engine
Provides interactive natural language conversational intelligence over customer reviews,
ad-hoc single review NLP scoring, aspect-level diagnosis, anomaly hunting, and executive KPI queries.
Strictly adheres to zero-fabrication rules: all insights are computed dynamically from real dataset execution.
"""

from typing import Dict, Any, List, Optional, Tuple
import re
import pandas as pd
import numpy as np

from src.nlp_engine import default_nlp_engine
from src.aspect_engine import default_aspect_engine, PRODUCT_ASPECTS
from src.analytics import AnalyticsEngine
from src.config import THEME_TOKENS, SentimentThresholds


class ReviewCopilotEngine:
    """Conversational intelligence engine that parses user queries, identifies intents,
    executes real analytical computations against active review data, and formats
    production-grade structured responses.
    """

    def __init__(self):
        self.thresholds = SentimentThresholds()

    def process_query(self, query: str, df: Optional[pd.DataFrame] = None) -> str:
        """Main dispatch method. Parses query, determines intent, and executes real computation.

        Args:
            query: User's natural language question or command.
            df: Active scored DataFrame. If None or empty, copilot instructs user to load data.

        Returns:
            Rich markdown formatted response.
        """
        if not query or not query.strip():
            return "💬 Please enter a question, topic, or review text to analyze."

        clean_q = query.strip()
        lower_q = clean_q.lower()

        # Intent 1: Direct Review Evaluation
        # Matches commands like: "Analyze: The battery is great", "Score this: ...", or quoted text
        direct_eval_match = re.search(r"^(?:analyze|score|evaluate|test|review|check)(?:\s+this)?(?:\s+review)?\s*[:\-]\s*(.+)", clean_q, re.IGNORECASE)
        if direct_eval_match:
            review_to_analyze = direct_eval_match.group(1).strip()
            return self._handle_direct_review_analysis(review_to_analyze)

        if (lower_q.startswith('"') and lower_q.endswith('"') and len(clean_q) > 15) or \
           ("what is the sentiment of" in lower_q or "sentiment of this review" in lower_q):
            # Extract review text from quotes or question
            extracted = clean_q.strip('"').strip("'")
            if "sentiment of" in lower_q:
                parts = re.split(r"sentiment of\s*(?:this review)?(?:\s*[:\-]?\s*)?", clean_q, flags=re.IGNORECASE)
                if len(parts) > 1 and len(parts[1].strip()) > 5:
                    extracted = parts[1].strip().strip('"').strip("'")
            return self._handle_direct_review_analysis(extracted)

        # Normalize tuple input if caller passed (df, metadata)
        if isinstance(df, tuple) and len(df) > 0 and isinstance(df[0], pd.DataFrame):
            df = df[0]

        # Ensure active dataset exists for data-dependent queries
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return (
                "⚠️ **No Active Dataset Found in Session**\n\n"
                "I need an active dataset to perform dataset analytics. You can:\n"
                "- Upload a CSV file in **Bulk Analysis** (`pages/3_📁_Bulk_Analysis.py`).\n"
                "- Restore the bundled demonstration dataset from the sidebar.\n"
                "- Or ask me to analyze a specific review text directly (e.g., `Analyze: Great sound but poor battery`)."
            )

        # Intent 2: Executive Overview / Health / KPIs
        if any(w in lower_q for w in ["kpi", "overview", "summary", "nss", "net sentiment", "health", "how are we doing", "status"]):
            return self._handle_kpi_overview(df)

        # Intent 3: Anomaly / Suspicious / Fake Reviews Hunter
        if any(w in lower_q for w in ["anomaly", "anomalies", "fake", "mismatch", "suspicious", "discrepancy", "sarcasm", "divergence"]):
            return self._handle_anomalies_query(df)

        # Intent 4: Top Complaints / Negative Drivers / Fix Recommendations
        if any(w in lower_q for w in ["complaint", "negative driver", "biggest issue", "what to fix", "what needs improvement", "problems", "worst aspect", "why are customers unhappy"]):
            return self._handle_negative_drivers_query(df)

        # Intent 5: Top Strengths / Positive Drivers / What Users Love
        if any(w in lower_q for w in ["strength", "positive driver", "what do users love", "best feature", "highlights", "top praise"]):
            return self._handle_positive_drivers_query(df)

        # Intent 6: Category Comparison
        if any(w in lower_q for w in ["category", "categories", "compare category", "department", "wearables vs audio"]):
            return self._handle_category_comparison(df)

        # Intent 7: Aspect Deep-Dive
        matched_aspect = self._detect_aspect(lower_q)
        if matched_aspect:
            return self._handle_aspect_query(df, matched_aspect)

        # Intent 8: Keyword Search over dataset
        # Extract potential keyword from queries like "show me reviews about cord", "what about comfort"
        keyword_match = re.search(r"(?:about|mentioning|containing|with|for)\s+([a-zA-Z0-9_\-]+)", clean_q, re.IGNORECASE)
        if keyword_match:
            kw = keyword_match.group(1).lower()
            if len(kw) > 2 and kw not in ["the", "this", "that", "what", "show", "some"]:
                return self._handle_keyword_search(df, kw)

        # Fallback / Conversational Guide
        return self._handle_fallback_guide(clean_q, df)

    def _handle_direct_review_analysis(self, review_text: str) -> str:
        """Executes real-time dual-engine NLP inference and aspect extraction on single text."""
        if not review_text or len(review_text) < 3:
            return "⚠️ Please provide a non-empty review text to analyze (e.g., `Analyze: Outstanding battery life!`)."

        result = default_nlp_engine.analyze(review_text, include_vader=True)
        extracted_aspects = default_aspect_engine.extract_aspects_from_text(review_text)

        badge_color = {
            "Positive": "#10B981",
            "Negative": "#F43F5E",
            "Neutral": "#94A3B8"
        }.get(result.sentiment, "#94A3B8")

        # Consensus check
        vader_sent = result.vader_sentiment or result.sentiment
        consensus = (result.sentiment == vader_sent)
        consensus_str = "✓ **100% Dual-Engine Consensus** (TextBlob & VADER agree)" if consensus else f"⚠️ **Lexical Divergence** (TextBlob: *{result.sentiment}*, VADER: *{vader_sent}*)"

        aspect_lines = []
        if extracted_aspects:
            for asp_name, item in extracted_aspects.items():
                asp_sent = item.get("sentiment", "Neutral")
                s_icon = "🟢" if asp_sent == "Positive" else "🔴" if asp_sent == "Negative" else "⚪"
                snippet = item.get("snippet", "")
                aspect_lines.append(f"- {s_icon} **{asp_name}** ({asp_sent}): *\"{snippet}\"*")
            aspects_block = "\n".join(aspect_lines)
        else:
            aspects_block = "_No explicit dimension keywords (battery, sound, etc.) detected in this sentence._"

        sub_label = "Objective / Factual" if result.subjectivity < 0.35 else "Balanced" if result.subjectivity < 0.65 else "Emotional / Opinionated"

        return f"""### ⚡ Real-Time Review Scoring Result

> *\"{review_text}\"*

| Metric | Result | Interpretation |
| :--- | :---: | :--- |
| **Final Sentiment** | <span style="color:{badge_color}; font-weight:800;">{result.sentiment.upper()}</span> | Classified sentiment |
| **Polarity Score** | **`{result.polarity:+.3f}`** | Scale: `-1.0` (Extremely Negative) to `+1.0` (Extremely Positive) |
| **Subjectivity** | **`{result.subjectivity:.3f}`** | `{result.subjectivity*100:.1f}%` ({sub_label}) |
| **Engine Consensus** | {consensus_str} | Dual-lexicon validation |

#### 🏷️ Detected Aspect Mentions
{aspects_block}
"""

    def _handle_kpi_overview(self, df: pd.DataFrame) -> str:
        """Calculates and displays real executive KPI metrics from active data."""
        kpis = AnalyticsEngine.calculate_kpis(df)
        total = kpis.get("total_reviews", 0)
        pos_pct = kpis.get("positive_pct", 0.0)
        neg_pct = kpis.get("negative_pct", 0.0)
        neu_pct = kpis.get("neutral_pct", 0.0)
        nss = round(pos_pct - neg_pct, 1)
        avg_pol = kpis.get("avg_polarity", 0.0)
        avg_sub = kpis.get("avg_subjectivity", 0.0)

        # Calculate average rating if rating column present
        rating_col = "rating" if "rating" in df.columns else None
        if rating_col:
            num_ratings = pd.to_numeric(df[rating_col], errors="coerce").dropna()
            avg_rating = float(num_ratings.mean()) if not num_ratings.empty else None
        else:
            avg_rating = None

        nss_badge = "🟢 Excellent" if nss >= 40 else "🟡 Moderate" if nss >= 0 else "🔴 Critical"
        rating_str = f"★ {avg_rating:.2f} / 5.0" if avg_rating is not None else "N/A"

        # Auto executive takeaways
        takeaways = AnalyticsEngine.generate_executive_insights(df, kpis)
        takeaways_bullets = "\n".join([f"- **{t.get('verdict')}**: {t.get('description')}" for t in takeaways[:2]]) if takeaways else "- Feedback volume is within expected operational thresholds."

        return f"""### 📊 Executive Sentiment Health & Platform Overview

Here is the live breakdown of your active dataset (**{total:,} reviews**):

| Key Performance Indicator | Value | Benchmark / Health |
| :--- | :---: | :--- |
| **Total Ingested Reviews** | **`{total:,}`** | 100% analyzed & scored |
| **Net Sentiment Score (NSS)** | **`{nss:+.1f}`** | {nss_badge} (% Pos − % Neg) |
| **Positive Reviews** | **`{pos_pct:.1f}%`** | {kpis.get('positive_count', 0):,} records |
| **Neutral Reviews** | **`{neu_pct:.1f}%`** | {kpis.get('neutral_count', 0):,} records |
| **Negative Reviews** | **`{neg_pct:.1f}%`** | {kpis.get('negative_count', 0):,} records |
| **Mean Polarity** | **`{avg_pol:+.3f}`** | Range: `[-1.0, +1.0]` |
| **Average Star Rating** | **`{rating_str}`** | User-reported scale `[1, 5]` |
| **Mean Subjectivity** | **`{avg_sub:.3f}`** | `{avg_sub*100:.1f}%` Opinion density |

#### 💡 Automated Executive Takeaways
{takeaways_bullets}
"""

    def _handle_anomalies_query(self, df: pd.DataFrame) -> str:
        """Finds and surfaces genuine rating-polarity discrepancy anomalies."""
        anomalies_df = AnalyticsEngine.detect_sentiment_anomalies(df)
        total_anomalies = len(anomalies_df)

        if total_anomalies == 0:
            return (
                "✅ **No Rating-Sentiment Anomalies Detected!**\n\n"
                "All user star ratings in your dataset align coherently with NLP polarity scores "
                "(no 5-star ratings with severe negative text or 1-star ratings with glowing praise)."
            )

        high_rating_neg = anomalies_df[anomalies_df["anomaly_type"] == "High Rating / Negative Sentiment"]
        low_rating_pos = anomalies_df[anomalies_df["anomaly_type"] == "Low Rating / Positive Sentiment"]

        examples = []
        text_col = "review_text" if "review_text" in df.columns else df.columns[0]
        pol_col = "tb_polarity" if "tb_polarity" in df.columns else "polarity"

        for idx, row in anomalies_df.head(4).iterrows():
            r = row.get("rating", "N/A")
            pol = row.get(pol_col, 0.0)
            txt = str(row.get(text_col, ""))[:110]
            a_type = row.get("anomaly_type", "Anomaly")
            examples.append(f"- **★ {r} Rating vs `{pol:+.2f}` Polarity** ({a_type}):\n  *\"{txt}...\"*")

        examples_str = "\n".join(examples)

        return f"""### ⚠️ Rating & Sentiment Anomaly Report

Detected **{total_anomalies} anomalous reviews** ({total_anomalies/len(df)*100:.1f}% of active dataset):

- **{len(high_rating_neg)} High-Rating Discrepancies** (★ 4-5 stars with negative text — potential customer sarcasm or misclicks).
- **{len(low_rating_pos)} Low-Rating Discrepancies** (★ 1-2 stars with positive text — potential delivery complaints or inverted scales).

#### 🔍 Notable Anomaly Samples:
{examples_str}

*Recommendation: Review these records in the **Review Explorer** (`pages/5_🔎_Review_Explorer.py`) using the Anomaly Filter.*
"""

    def _handle_negative_drivers_query(self, df: pd.DataFrame) -> str:
        """Analyzes negative sentiment drivers, top negative N-grams, and lowest aspect scores."""
        text_col = "clean_text" if "clean_text" in df.columns else "review_text" if "review_text" in df.columns else df.columns[0]
        sent_col = "tb_sentiment" if "tb_sentiment" in df.columns else "sentiment"

        neg_df = df[df[sent_col] == "Negative"]
        if neg_df.empty:
            return "🎉 **Zero Negative Reviews!** Your active dataset contains no negative reviews."

        # Extract top negative bigrams
        top_bigrams = AnalyticsEngine.extract_ngrams(df, text_col=text_col, n=2, top_k=5, sentiment_filter="Negative")
        bigram_str = "\n".join([f"- **`{phrase}`**: mentioned `{count}` times in complaints" for phrase, count in top_bigrams]) if top_bigrams else "No recurring negative N-grams found."

        # Aspect analysis
        aspect_df = default_aspect_engine.analyze_dataset_aspects(df, text_col=text_col)
        aspect_recs = []
        if not aspect_df.empty:
            lowest_aspects = aspect_df.sort_values(by="net_aspect_score").head(3)
            for _, row in lowest_aspects.iterrows():
                aspect_recs.append(f"- 🔴 **{row['aspect']}**: Net Aspect Score `{row['net_aspect_score']:+.1f}` ({row['negative_count']} negative vs {row['positive_count']} positive mentions)")
        aspects_str = "\n".join(aspect_recs) if aspect_recs else "No aspect dimensions detected in complaints."

        return f"""### 🛑 Top Negative Drivers & Customer Pain Points

Based on **{len(neg_df)} negative reviews** in your dataset:

#### 1. Recurring Complaint Phrases (Top Bigrams)
{bigram_str}

#### 2. Most Vulnerable Product Dimensions (Lowest Net Aspect Scores)
{aspects_str}

#### 🎯 Strategic Engineering Actions:
1. Prioritize engineering sprint on the lowest-scoring aspect above.
2. Filter for these exact phrases in **Review Explorer** to review detailed customer quotes.
"""

    def _handle_positive_drivers_query(self, df: pd.DataFrame) -> str:
        """Analyzes top positive drivers, praise N-grams, and winning product aspects."""
        text_col = "clean_text" if "clean_text" in df.columns else "review_text" if "review_text" in df.columns else df.columns[0]
        sent_col = "tb_sentiment" if "tb_sentiment" in df.columns else "sentiment"

        pos_df = df[df[sent_col] == "Positive"]
        if pos_df.empty:
            return "⚠️ No positive reviews found in the current dataset."

        top_bigrams = AnalyticsEngine.extract_ngrams(df, text_col=text_col, n=2, top_k=5, sentiment_filter="Positive")
        bigram_str = "\n".join([f"- **`{phrase}`**: mentioned `{count}` times in positive feedback" for phrase, count in top_bigrams]) if top_bigrams else "No recurring positive N-grams found."

        aspect_df = default_aspect_engine.analyze_dataset_aspects(df, text_col=text_col)
        aspect_recs = []
        if not aspect_df.empty:
            highest_aspects = aspect_df.sort_values(by="net_aspect_score", ascending=False).head(3)
            for _, row in highest_aspects.iterrows():
                aspect_recs.append(f"- 🟢 **{row['aspect']}**: Net Aspect Score `{row['net_aspect_score']:+.1f}` ({row['positive_count']} positive mentions, `{row['positive_rate']:.1f}%` positive)")
        aspects_str = "\n".join(aspect_recs) if aspect_recs else "No aspect dimensions detected."

        return f"""### 🌟 Top Product Strengths & Customer Praise

Based on **{len(pos_df)} positive reviews** in your dataset:

#### 1. Core Positive Lexical Drivers
{bigram_str}

#### 2. Top-Performing Product Dimensions (Highest Net Aspect Scores)
{aspects_str}

#### 💡 Marketing & Product Takeaway:
Highlight these winning dimensions in product marketing and landing pages to maximize conversion.
"""

    def _handle_aspect_query(self, df: pd.DataFrame, aspect_name: str) -> str:
        """Deep-dive analytics for a specific product aspect dimension."""
        text_col = "review_text" if "review_text" in df.columns else df.columns[0]
        sent_col = "tb_sentiment" if "tb_sentiment" in df.columns else "sentiment"
        pol_col = "tb_polarity" if "tb_polarity" in df.columns else "polarity"

        keywords = PRODUCT_ASPECTS.get(aspect_name, [])
        regex_pattern = r"\b(?:" + "|".join([re.escape(k) for k in keywords]) + r")\b"

        matched = df[df[text_col].astype(str).str.contains(regex_pattern, case=False, regex=True, na=False)]
        total_mentions = len(matched)

        if total_mentions == 0:
            return f"ℹ️ **No Mentions of '{aspect_name}' Found** in the current dataset (checked keywords: {', '.join(keywords[:5])})."

        pos_count = len(matched[matched[sent_col] == "Positive"])
        neg_count = len(matched[matched[sent_col] == "Negative"])
        neu_count = len(matched[matched[sent_col] == "Neutral"])
        pos_rate = (pos_count / total_mentions * 100) if total_mentions > 0 else 0.0
        net_score = ((pos_count - neg_count) / total_mentions * 100) if total_mentions > 0 else 0.0
        avg_aspect_pol = matched[pol_col].mean() if pol_col in matched.columns else 0.0

        # Sample positive quote
        pos_sample = matched[matched[sent_col] == "Positive"][text_col].dropna().head(1).tolist()
        pos_quote = f"\"{pos_sample[0][:110]}...\"" if pos_sample else "_None available_"

        # Sample negative quote
        neg_sample = matched[matched[sent_col] == "Negative"][text_col].dropna().head(1).tolist()
        neg_quote = f"\"{neg_sample[0][:110]}...\"" if neg_sample else "_None available_"

        status_emoji = "🟢" if net_score >= 30 else "🟡" if net_score >= 0 else "🔴"

        return f"""### 🏷️ Aspect Intelligence Deep-Dive: **{aspect_name}**

{status_emoji} **Net Aspect Score**: **`{net_score:+.1f}`** | **Total Mentions**: **`{total_mentions}`** reviews

| Aspect Metric | Value | Proportion |
| :--- | :---: | :---: |
| **Positive Mentions** | **`{pos_count}`** | `{pos_rate:.1f}%` |
| **Negative Mentions** | **`{neg_count}`** | `{(neg_count/total_mentions*100):.1f}%` |
| **Neutral Mentions** | **`{neu_count}`** | `{(neu_count/total_mentions*100):.1f}%` |
| **Average Aspect Polarity** | **`{avg_aspect_pol:+.3f}`** | `[-1.0, +1.0]` scale |

#### 🗣️ Customer Voice Samples
- **Customer Praise (Positive)**:
  > *{pos_quote}*
- **Customer Complaint (Negative)**:
  > *{neg_quote}*
"""

    def _handle_category_comparison(self, df: pd.DataFrame) -> str:
        """Compares categories across sentiment and volume."""
        cat_col = "category" if "category" in df.columns else df.columns[0]
        cat_df = AnalyticsEngine.get_category_breakdown(df, category_col=cat_col)
        if cat_df is None or cat_df.empty:
            return "ℹ️ No product category column identified in the active dataset."

        rows = []
        for _, r in cat_df.iterrows():
            c_name = r[cat_col]
            tot = r["total_reviews"]
            pos = r["positive"]
            neg = r["negative"]
            pos_rate = r["positive_rate"]
            rows.append(f"| **{c_name}** | **`{tot}`** | `{pos}` ({pos_rate:.1f}%) | `{neg}` |")

        table_body = "\n".join(rows)

        return f"""### 📦 Product Category Sentiment Comparison

| Category | Review Volume | Positive Mentions | Negative Mentions |
| :--- | :---: | :---: | :---: |
{table_body}

*Data derived dynamically from column `{cat_col}` in the active dataset.*
"""

    def _handle_keyword_search(self, df: pd.DataFrame, keyword: str) -> str:
        """Performs lexical search across all reviews for a user-specified keyword."""
        text_col = "review_text" if "review_text" in df.columns else df.columns[0]
        sent_col = "tb_sentiment" if "tb_sentiment" in df.columns else "sentiment"
        pol_col = "tb_polarity" if "tb_polarity" in df.columns else "polarity"

        matched = df[df[text_col].astype(str).str.contains(r"\b" + re.escape(keyword) + r"\b", case=False, regex=True, na=False)]
        total = len(matched)

        if total == 0:
            return f"🔍 No reviews found containing the keyword **'{keyword}'** in the current dataset."

        pos_count = len(matched[matched[sent_col] == "Positive"])
        neg_count = len(matched[matched[sent_col] == "Negative"])
        avg_pol = matched[pol_col].mean() if pol_col in matched.columns else 0.0

        sample_reviews = []
        for idx, row in matched.head(3).iterrows():
            s = row.get(sent_col, "Neutral")
            txt = str(row.get(text_col, ""))[:110]
            s_icon = "🟢" if s == "Positive" else "🔴" if s == "Negative" else "⚪"
            sample_reviews.append(f"- {s_icon} [{s}]: *\"{txt}...\"*")
        sample_str = "\n".join(sample_reviews)

        return f"""### 🔍 Search Intelligence for Keyword: **\"{keyword}\"**

Found **{total} matching reviews** across your dataset:
- **Positive**: `{pos_count}` (`{pos_count/total*100:.1f}%`)
- **Negative**: `{neg_count}` (`{neg_count/total*100:.1f}%`)
- **Mean Polarity**: `{avg_pol:+.3f}`

#### 💬 Representative Quotes:
{sample_str}
"""

    def _detect_aspect(self, query: str) -> Optional[str]:
        """Maps query keywords to recognized product aspect dimensions."""
        aspect_aliases = {
            "Battery & Power": ["battery", "charge", "charging", "charger", "power", "battery life"],
            "Audio & Acoustics": ["sound", "audio", "bass", "treble", "volume", "music", "noise", "microphone", "mic", "anc"],
            "Display & Visuals": ["screen", "display", "resolution", "brightness", "panel", "visuals"],
            "Price & Value": ["price", "cost", "expensive", "cheap", "value", "worth", "money", "affordable"],
            "Build & Ergonomics": ["build", "durability", "material", "quality", "sturdy", "break", "broken", "comfort", "comfortable", "fit", "ergonomics"],
            "Performance & Speed": ["performance", "speed", "fast", "slow", "lag", "cpu", "thermal", "heat"],
            "Software & App": ["software", "app", "ui", "interface", "firmware", "bug", "crash", "bluetooth", "connectivity"],
            "Customer Support": ["service", "support", "warranty", "customer service", "delivery", "shipping", "return"]
        }
        for aspect, terms in aspect_aliases.items():
            for t in terms:
                if re.search(r"\b" + re.escape(t) + r"\b", query):
                    return aspect
        return None

    def _handle_fallback_guide(self, query: str, df: pd.DataFrame) -> str:
        """Friendly conversational fallback offering actionable query suggestions."""
        total = len(df)
        return f"""### 🤖 ReviewSense AI Copilot

I can answer questions, synthesize insights, and evaluate reviews across your active dataset (**{total} reviews**).

Here are prompt templates you can try:

- 📊 **Executive Overview**: `"What is our Net Sentiment Score and overall health?"`
- 🏷️ **Aspect Diagnostics**: `"How is battery life?"` or `"What do users say about sound quality?"`
- 🛑 **Top Complaints**: `"What are the biggest negative drivers and customer complaints?"`
- 🌟 **Top Strengths**: `"What do customers love most?"`
- ⚠️ **Anomaly Detection**: `"Are there any rating or sentiment anomalies?"`
- 📦 **Category Comparison**: `"Compare category performance"`
- ⚡ **Direct Review Scoring**: `Analyze: Fantastic design, but the battery dies too quickly.`
"""


# Global singleton instance
default_copilot_engine = ReviewCopilotEngine()
