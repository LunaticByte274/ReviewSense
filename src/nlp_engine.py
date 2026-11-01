"""
ReviewSense - Natural Language Processing Sentiment Engine
Provides TextBlob analysis, VADER sentiment scoring, analyzer comparison,
and comprehensive interpretability helpers with exact boundary checks.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from textblob import TextBlob

from src.config import SentimentThresholds

# Optional VADER import with fallback
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    _VADER_AVAILABLE = True
except (ImportError, LookupError):
    _VADER_AVAILABLE = False


@dataclass
class SentimentResult:
    """Standardized representation of a single text sentiment analysis."""
    text: str
    polarity: float
    subjectivity: float
    sentiment: str  # 'Positive', 'Negative', 'Neutral'
    analyzer: str  # 'TextBlob'
    # Optional VADER scores
    vader_pos: Optional[float] = None
    vader_neg: Optional[float] = None
    vader_neu: Optional[float] = None
    vader_compound: Optional[float] = None
    vader_sentiment: Optional[str] = None
    agreement: Optional[bool] = None
    explanation: str = ""


class NLPEngine:
    """
    Main NLP Sentiment Analyzer orchestration class.
    Executes TextBlob and optional VADER evaluations under deterministic rules.
    """

    def __init__(self, thresholds: Optional[SentimentThresholds] = None):
        self.thresholds = thresholds or SentimentThresholds()
        self._vader_analyzer = None
        self._init_vader()

    def _init_vader(self):
        """Initializes VADER analyzer safely if installed and data is present."""
        global _VADER_AVAILABLE
        if not _VADER_AVAILABLE:
            return
        try:
            self._vader_analyzer = SentimentIntensityAnalyzer()
        except LookupError:
            try:
                nltk.download("vader_lexicon", quiet=True)
                self._vader_analyzer = SentimentIntensityAnalyzer()
            except Exception:
                self._vader_analyzer = None
                _VADER_AVAILABLE = False
        except Exception:
            self._vader_analyzer = None
            _VADER_AVAILABLE = False

    @property
    def is_vader_available(self) -> bool:
        """Returns True if VADER Sentiment Intensity Analyzer is ready."""
        return self._vader_analyzer is not None

    def classify_tb_polarity(self, polarity: float) -> str:
        """
        Classifies TextBlob polarity into Positive, Negative, or Neutral.
        Exact boundary handling:
        - polarity > TB_POSITIVE_MIN (+0.05) -> Positive
        - polarity < TB_NEGATIVE_MAX (-0.05) -> Negative
        - otherwise (including [-0.05, 0.05]) -> Neutral
        """
        # Rounding to 6 decimal places to prevent floating-point representation jitter
        pol = round(float(polarity), 6)
        if pol > self.thresholds.TB_POSITIVE_MIN:
            return "Positive"
        elif pol < self.thresholds.TB_NEGATIVE_MAX:
            return "Negative"
        else:
            return "Neutral"

    def classify_vader_compound(self, compound: float) -> str:
        """
        Classifies VADER compound score into Positive, Negative, or Neutral.
        Exact boundary handling:
        - compound >= VADER_POSITIVE_MIN (0.05) -> Positive
        - compound <= VADER_NEGATIVE_MAX (-0.05) -> Negative
        - otherwise -> Neutral
        """
        comp = round(float(compound), 6)
        if comp >= self.thresholds.VADER_POSITIVE_MIN:
            return "Positive"
        elif comp <= self.thresholds.VADER_NEGATIVE_MAX:
            return "Negative"
        else:
            return "Neutral"

    def analyze_textblob(self, text: str) -> Dict[str, Any]:
        """Runs TextBlob sentiment extraction on input text."""
        if not isinstance(text, str) or not text.strip():
            return {
                "polarity": 0.0,
                "subjectivity": 0.0,
                "sentiment": "Neutral"
            }

        blob = TextBlob(text)
        polarity = round(float(blob.sentiment.polarity), 4)
        subjectivity = round(float(blob.sentiment.subjectivity), 4)
        sentiment = self.classify_tb_polarity(polarity)

        return {
            "polarity": polarity,
            "subjectivity": subjectivity,
            "sentiment": sentiment
        }

    def analyze_vader(self, text: str) -> Optional[Dict[str, Any]]:
        """Runs VADER sentiment extraction if available."""
        if not self.is_vader_available or not isinstance(text, str) or not text.strip():
            if not self.is_vader_available:
                return None
            return {
                "pos": 0.0,
                "neg": 0.0,
                "neu": 1.0,
                "compound": 0.0,
                "sentiment": "Neutral"
            }

        scores = self._vader_analyzer.polarity_scores(text)
        pos = round(scores.get("pos", 0.0), 4)
        neg = round(scores.get("neg", 0.0), 4)
        neu = round(scores.get("neu", 0.0), 4)
        compound = round(scores.get("compound", 0.0), 4)
        sentiment = self.classify_vader_compound(compound)

        return {
            "pos": pos,
            "neg": neg,
            "neu": neu,
            "compound": compound,
            "sentiment": sentiment
        }

    def analyze(self, text: str, include_vader: bool = True) -> SentimentResult:
        """
        Executes unified sentiment analysis on single review text.
        Calculates TextBlob, optional VADER, agreement metrics, and explanations.
        """
        if not isinstance(text, str) or not text.strip():
            return SentimentResult(
                text=str(text) if text is not None else "",
                polarity=0.0,
                subjectivity=0.0,
                sentiment="Neutral",
                analyzer="TextBlob",
                explanation="Empty or non-textual input provided. Evaluated as Neutral with 0.0 scores."
            )

        tb_res = self.analyze_textblob(text)
        result = SentimentResult(
            text=text,
            polarity=tb_res["polarity"],
            subjectivity=tb_res["subjectivity"],
            sentiment=tb_res["sentiment"],
            analyzer="TextBlob"
        )

        if include_vader and self.is_vader_available:
            vader_res = self.analyze_vader(text)
            if vader_res:
                result.vader_pos = vader_res["pos"]
                result.vader_neg = vader_res["neg"]
                result.vader_neu = vader_res["neu"]
                result.vader_compound = vader_res["compound"]
                result.vader_sentiment = vader_res["sentiment"]
                result.agreement = (result.sentiment == result.vader_sentiment)

        result.explanation = self.generate_explanation(result)
        return result

    def generate_explanation(self, result: SentimentResult) -> str:
        """Provides an academic, human-readable breakdown of the NLP scores."""
        # Polarity explanation
        if result.polarity > self.thresholds.TB_POSITIVE_MIN:
            pol_desc = f"strongly positive sentiment (polarity = {result.polarity:+.2f})"
        elif result.polarity < self.thresholds.TB_NEGATIVE_MAX:
            pol_desc = f"unfavorable or negative sentiment (polarity = {result.polarity:+.2f})"
        else:
            pol_desc = f"balanced, neutral, or mixed tone (polarity = {result.polarity:+.2f})"

        # Subjectivity explanation
        if result.subjectivity >= self.thresholds.SUBJECTIVITY_SUBJECTIVE_MIN:
            sub_desc = f"highly subjective language containing personal opinions or emotions ({result.subjectivity:.2f})"
        elif result.subjectivity <= self.thresholds.SUBJECTIVITY_OBJECTIVE_MAX:
            sub_desc = f"factual, objective statements with minimal emotional bias ({result.subjectivity:.2f})"
        else:
            sub_desc = f"moderate subjectivity mixing facts and personal impressions ({result.subjectivity:.2f})"

        exp = f"TextBlob detected {pol_desc} characterized by {sub_desc}."

        if result.vader_compound is not None:
            v_desc = f"VADER produced a compound score of {result.vader_compound:+.2f} ({result.vader_sentiment})."
            if result.agreement:
                exp += f" {v_desc} Both analyzers agree on the '{result.sentiment}' classification."
            else:
                exp += (
                    f" {v_desc} Analyzer divergence detected: TextBlob classified as '{result.sentiment}' "
                    f"while VADER classified as '{result.vader_sentiment}'. This reflects differing lexicon weighting "
                    f"and punctuation/capitalization sensitivity."
                )

        return exp


# Global default instance
default_nlp_engine = NLPEngine()

