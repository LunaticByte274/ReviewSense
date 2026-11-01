"""
ReviewSense - Analytics Layer
Provides statistical aggregations, KPI computations, distributions,
cross-tabulations, trend analysis, and academic model evaluation.
"""

from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.preprocessing import tokenize_text, remove_stopwords_from_tokens


class AnalyticsEngine:
    """Calculates deterministic metrics, summaries, and breakdowns for product reviews."""

    @staticmethod
    def calculate_kpis(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates core summary KPIs with strict denominator handling.
        Returns accurate values and safe defaults for empty dataframes.
        """
        if df is None or df.empty:
            return {
                "total_reviews": 0,
                "valid_reviews": 0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "positive_pct": 0.0,
                "negative_pct": 0.0,
                "neutral_pct": 0.0,
                "avg_polarity": 0.0,
                "avg_subjectivity": 0.0,
            }

        # Filter valid reviews for scoring if review_status is present
        valid_df = df[df["review_status"] == "Valid"] if "review_status" in df.columns else df
        total_valid = len(valid_df)

        if total_valid == 0:
            return {
                "total_reviews": len(df),
                "valid_reviews": 0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "positive_pct": 0.0,
                "negative_pct": 0.0,
                "neutral_pct": 0.0,
                "avg_polarity": 0.0,
                "avg_subjectivity": 0.0,
            }

        # Prefer TextBlob sentiment as baseline
        sent_col = "tb_sentiment" if "tb_sentiment" in valid_df.columns else "sentiment"
        pol_col = "tb_polarity" if "tb_polarity" in valid_df.columns else "polarity"
        sub_col = "tb_subjectivity" if "tb_subjectivity" in valid_df.columns else "subjectivity"

        pos_count = int((valid_df[sent_col] == "Positive").sum())
        neg_count = int((valid_df[sent_col] == "Negative").sum())
        neu_count = int((valid_df[sent_col] == "Neutral").sum())

        pos_pct = round((pos_count / total_valid) * 100, 1)
        neg_pct = round((neg_count / total_valid) * 100, 1)
        neu_pct = round((neu_count / total_valid) * 100, 1)

        avg_pol = round(float(valid_df[pol_col].mean()), 4) if pol_col in valid_df.columns else 0.0
        avg_sub = round(float(valid_df[sub_col].mean()), 4) if sub_col in valid_df.columns else 0.0

        return {
            "total_reviews": len(df),
            "valid_reviews": total_valid,
            "positive_count": pos_count,
            "negative_count": neg_count,
            "neutral_count": neu_count,
            "positive_pct": pos_pct,
            "negative_pct": neg_pct,
            "neutral_pct": neu_pct,
            "avg_polarity": avg_pol,
            "avg_subjectivity": avg_sub,
        }

    @staticmethod
    def get_category_breakdown(df: pd.DataFrame, category_col: str = "category") -> Optional[pd.DataFrame]:
        """Calculates review count, sentiment counts, and average polarity per category."""
        if df is None or df.empty or category_col not in df.columns:
            return None

        valid_df = df[df["review_status"] == "Valid"] if "review_status" in df.columns else df
        if valid_df.empty:
            return None

        sent_col = "tb_sentiment" if "tb_sentiment" in valid_df.columns else "sentiment"
        pol_col = "tb_polarity" if "tb_polarity" in valid_df.columns else "polarity"

        grouped = valid_df.groupby(category_col).agg(
            total_reviews=(category_col, "count"),
            avg_polarity=(pol_col, "mean"),
            positive=(sent_col, lambda s: (s == "Positive").sum()),
            negative=(sent_col, lambda s: (s == "Negative").sum()),
            neutral=(sent_col, lambda s: (s == "Neutral").sum())
        ).reset_index()

        grouped["avg_polarity"] = grouped["avg_polarity"].round(3)
        grouped["positive_rate"] = ((grouped["positive"] / grouped["total_reviews"]) * 100).round(1)
        grouped["negative_rate"] = ((grouped["negative"] / grouped["total_reviews"]) * 100).round(1)

        return grouped.sort_values(by="total_reviews", ascending=False)

    @staticmethod
    def get_product_breakdown(df: pd.DataFrame, product_col: str = "product_name") -> Optional[pd.DataFrame]:
        """Calculates performance statistics grouped by specific product."""
        if df is None or df.empty or product_col not in df.columns:
            return None

        valid_df = df[df["review_status"] == "Valid"] if "review_status" in df.columns else df
        if valid_df.empty:
            return None

        sent_col = "tb_sentiment" if "tb_sentiment" in valid_df.columns else "sentiment"
        pol_col = "tb_polarity" if "tb_polarity" in valid_df.columns else "polarity"

        grouped = valid_df.groupby(product_col).agg(
            total_reviews=(product_col, "count"),
            avg_polarity=(pol_col, "mean"),
            positive=(sent_col, lambda s: (s == "Positive").sum()),
            negative=(sent_col, lambda s: (s == "Negative").sum()),
            neutral=(sent_col, lambda s: (s == "Neutral").sum())
        ).reset_index()

        grouped["avg_polarity"] = grouped["avg_polarity"].round(3)
        grouped["positive_rate"] = ((grouped["positive"] / grouped["total_reviews"]) * 100).round(1)
        return grouped.sort_values(by="total_reviews", ascending=False)

    @staticmethod
    def get_rating_vs_sentiment(df: pd.DataFrame, rating_col: str = "rating") -> Optional[pd.DataFrame]:
        """Generates cross-tabulation of user star rating against predicted sentiment."""
        if df is None or df.empty or rating_col not in df.columns:
            return None

        valid_df = df[df["review_status"] == "Valid"] if "review_status" in df.columns else df
        if valid_df.empty:
            return None

        sent_col = "tb_sentiment" if "tb_sentiment" in valid_df.columns else "sentiment"
        pol_col = "tb_polarity" if "tb_polarity" in valid_df.columns else "polarity"

        # Coerce rating to numeric
        valid_df = valid_df.copy()
        valid_df[rating_col] = pd.to_numeric(valid_df[rating_col], errors="coerce")
        rated_df = valid_df.dropna(subset=[rating_col])

        if rated_df.empty:
            return None

        grouped = rated_df.groupby(rating_col).agg(
            review_count=(rating_col, "count"),
            avg_polarity=(pol_col, "mean"),
            positive=(sent_col, lambda s: (s == "Positive").sum()),
            negative=(sent_col, lambda s: (s == "Negative").sum()),
            neutral=(sent_col, lambda s: (s == "Neutral").sum())
        ).reset_index()

        grouped["avg_polarity"] = grouped["avg_polarity"].round(3)
        return grouped.sort_values(by=rating_col)

    @staticmethod
    def get_timeline_trend(df: pd.DataFrame, date_col: str = "date") -> Optional[pd.DataFrame]:
        """Aggregates review volume and average polarity over time."""
        if df is None or df.empty or date_col not in df.columns:
            return None

        valid_df = df[df["review_status"] == "Valid"] if "review_status" in df.columns else df
        if valid_df.empty:
            return None

        valid_df = valid_df.copy()
        valid_df["parsed_date"] = pd.to_datetime(valid_df[date_col], errors="coerce")
        dated_df = valid_df.dropna(subset=["parsed_date"])

        if dated_df.empty:
            return None

        pol_col = "tb_polarity" if "tb_polarity" in dated_df.columns else "polarity"
        sent_col = "tb_sentiment" if "tb_sentiment" in dated_df.columns else "sentiment"

        # Group by date formatted as YYYY-MM-DD
        dated_df["date_str"] = dated_df["parsed_date"].dt.strftime("%Y-%m-%d")
        trend = dated_df.groupby("date_str").agg(
            review_count=("date_str", "count"),
            avg_polarity=(pol_col, "mean"),
            positive=(sent_col, lambda s: (s == "Positive").sum()),
            negative=(sent_col, lambda s: (s == "Negative").sum()),
            neutral=(sent_col, lambda s: (s == "Neutral").sum())
        ).reset_index()

        trend["avg_polarity"] = trend["avg_polarity"].round(3)
        return trend.sort_values(by="date_str")

    @staticmethod
    def get_analyzer_agreement_stats(df: pd.DataFrame) -> Dict[str, Any]:
        """Computes agreement metrics between TextBlob and VADER."""
        if df is None or df.empty or "analyzer_agreement" not in df.columns:
            return {"available": False}

        valid_agreements = df.dropna(subset=["analyzer_agreement"])
        if valid_agreements.empty:
            return {"available": False}

        total = len(valid_agreements)
        agree_count = int(valid_agreements["analyzer_agreement"].sum())
        disagree_count = total - agree_count
        agreement_pct = round((agree_count / total) * 100, 1)

        return {
            "available": True,
            "total_evaluated": total,
            "agreed_count": agree_count,
            "disagreed_count": disagree_count,
            "agreement_percentage": agreement_pct
        }

    @staticmethod
    def get_extreme_reviews(df: pd.DataFrame, n: int = 3, text_col: str = "review_text") -> Dict[str, pd.DataFrame]:
        """Retrieves top N most positive and top N most negative reviews based on polarity."""
        if df is None or df.empty:
            return {"top_positive": pd.DataFrame(), "top_negative": pd.DataFrame()}

        valid_df = df[df["review_status"] == "Valid"] if "review_status" in df.columns else df
        if valid_df.empty:
            return {"top_positive": pd.DataFrame(), "top_negative": pd.DataFrame()}

        pol_col = "tb_polarity" if "tb_polarity" in valid_df.columns else "polarity"
        top_pos = valid_df.sort_values(by=pol_col, ascending=False).head(n)
        top_neg = valid_df.sort_values(by=pol_col, ascending=True).head(n)

        return {
            "top_positive": top_pos,
            "top_negative": top_neg
        }

    @staticmethod
    def extract_top_keywords(
        df: pd.DataFrame,
        text_col: str = "review_text",
        top_n: int = 15,
        sentiment_filter: Optional[str] = None
    ) -> List[Tuple[str, int]]:
        """Extracts most frequent informative words excluding stopwords."""
        if df is None or df.empty or text_col not in df.columns:
            return []

        subset = df[df["review_status"] == "Valid"] if "review_status" in df.columns else df
        if sentiment_filter and "tb_sentiment" in subset.columns:
            subset = subset[subset["tb_sentiment"] == sentiment_filter]

        from collections import Counter
        word_counts = Counter()

        for text in subset[text_col]:
            if isinstance(text, str):
                tokens = tokenize_text(text.lower())
                clean_tokens = remove_stopwords_from_tokens(tokens, preserve_negations=True)
                # Keep words with length > 2
                meaningful = [w for w in clean_tokens if len(w) > 2 and w.isalpha()]
                word_counts.update(meaningful)

        return word_counts.most_common(top_n)

    @staticmethod
    def evaluate_model_on_ground_truth(
        eval_df: pd.DataFrame,
        ground_truth_col: str = "true_sentiment",
        pred_col: str = "pred_sentiment"
    ) -> Dict[str, Any]:
        """
        Calculates academic classification performance metrics against human-labeled ground truth.
        Computes Accuracy, Macro Precision, Recall, F1-Score, and Confusion Matrix.
        """
        if eval_df is None or eval_df.empty:
            return {"status": "error", "message": "Evaluation dataset is empty."}

        labels = ["Positive", "Negative", "Neutral"]
        
        # Filter to only valid ground truth labels
        valid_eval = eval_df[
            eval_df[ground_truth_col].isin(labels) & eval_df[pred_col].isin(labels)
        ]

        if valid_eval.empty:
            return {"status": "error", "message": "No matching label pairs found in evaluation set."}

        y_true = valid_eval[ground_truth_col]
        y_pred = valid_eval[pred_col]

        acc = round(accuracy_score(y_true, y_pred), 4)
        prec = round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4)
        rec = round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4)
        f1 = round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4)

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=[f"Actual {l}" for l in labels], columns=[f"Pred {l}" for l in labels])

        rep_dict = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

        return {
            "status": "success",
            "total_samples": len(valid_eval),
            "accuracy": acc,
            "macro_precision": prec,
            "macro_recall": rec,
            "macro_f1": f1,
            "confusion_matrix": cm_df,
            "classification_report": rep_dict,
            "labels": labels
        }
