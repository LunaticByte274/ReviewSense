"""
ReviewSense - Page 7: Academic Documentation, Architecture & Benchmark Evaluation
Comprehensive technical overview, system architecture diagram, algorithm explanations,
real ground-truth benchmark evaluation, and Viva Voce preparation guide.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd

from src.config import AppConfig, THEME_TOKENS
from src.data_manager import DataManager
from src.nlp_engine import default_nlp_engine
from src.analytics import AnalyticsEngine
from src.visualizations import create_confusion_matrix_heatmap
from src.ui_components import (
    inject_custom_css,
    render_sidebar,
    render_kpi_card
)

# Page Setup
st.set_page_config(
    page_title=f"{AppConfig.APP_NAME} - About & Academic Specs",
    page_icon="⚡",
    layout="wide"
)

inject_custom_css()
render_sidebar()

st.markdown(f"""
<div style="margin-bottom: 24px;">
    <h2 style="margin: 0; color: #F8FAFC;">📖 About ReviewSense & Academic Specifications</h2>
    <p style="color: #94A3B8; font-size: 0.95rem; margin-top: 4px;">
        Engineering specifications, architecture diagrams, algorithmic methodology, real benchmark evaluation, and Viva Voce defense resources.
    </p>
</div>
""", unsafe_allow_html=True)

# Tabs for structured academic presentation
tab_overview, tab_arch, tab_eval, tab_viva = st.tabs([
    "📋 Project Overview",
    "🏗️ System Architecture",
    "🧪 Model Benchmark Evaluation",
    "🎓 Viva Voce Defense Guide"
])

# ----------------- TAB 1: PROJECT OVERVIEW -----------------
with tab_overview:
    st.markdown("""
    ### 🎯 Problem Statement & Motivation
    In contemporary e-commerce and consumer technology ecosystems, customer feedback is voluminous, unstructured, and distributed across thousands of reviews. 
    Manual review inspection is infeasible, while traditional rating aggregations (simple star averages) mask granular customer dissatisfaction regarding specific product features, durability, or customer support quality.

    **ReviewSense** solves this by providing an enterprise-grade Natural Language Processing (NLP) intelligence platform that converts raw customer reviews into actionable semantic telemetry using deterministic dual-lexicon inference (**TextBlob** and **VADER**).

    ### 📌 Core Objectives
    1. **Automated Sentiment Extraction**: Accurately infer polarity and subjectivity scores for single reviews or bulk CSV datasets.
    2. **Dual-Lexicon Cross-Validation**: Compare Pattern-based semantic orientation (TextBlob) with heuristic valence scoring (VADER) to identify linguistic nuances and analyzer divergence.
    3. **Non-Destructive Data Integrity**: Ensure all original review text and user metadata columns remain intact throughout preprocessing and analysis pipelines.
    4. **SaaS-Grade Visual Analytics**: Provide interactive cross-filtering by sentiment, category, rating, and polarity with Plotly telemetry.
    5. **Injection-Safe Ingestion & Export**: Guard spreadsheet exports against formula injection attacks (OWASP CSV Injection).

    ### 🛠️ Technology Stack
    - **Language & Runtime**: Python 3.12 (CPython)
    - **User Interface**: Streamlit 1.66+
    - **Natural Language Processing**: TextBlob 0.20+, NLTK 3.10+ (VADER SentimentIntensityAnalyzer, Punkt tokenizer, Stopwords corpus)
    - **Data Processing**: Pandas 3.1+, NumPy 2.5+
    - **Visualizations**: Plotly Express & Plotly Graph Objects
    - **Machine Learning & Evaluation**: Scikit-Learn 1.9+
    - **Verification & QA**: Pytest 9.2+
    """)

# ----------------- TAB 2: SYSTEM ARCHITECTURE -----------------
with tab_arch:
    st.markdown("### 🏛️ Modular System Architecture Diagram")
    
    # Render Mermaid diagram via markdown
    st.markdown("""
    ```mermaid
    flowchart TD
        User([Customer / Analyst])
        
        subgraph Presentation_Layer [Presentation Layer: Streamlit UI]
            P1[1_Home.py]
            P2[2_Single_Review.py]
            P3[3_Bulk_Analysis.py]
            P4[4_Analytics_Dashboard.py]
            P5[5_Review_Explorer.py]
            P6[6_Text_Preprocessing.py]
            P7[7_About_Project.py]
        end

        subgraph Session_Manager [Shared State & Data Lifecycle]
            SS[(st.session_state)]
            ActiveData[Active Dataset: DataFrame]
            Meta[Dataset Metadata & Schema]
            SS --- ActiveData
            SS --- Meta
        end

        subgraph Core_Modules [Core Application Logic]
            DM[data_manager.py\nCSV Parsing, Validation, Export Sanitization]
            NLP[nlp_engine.py\nTextBlob & VADER Dual-Engine]
            PRE[preprocessing.py\nHTML Stripping, Negation-Aware Stopwords]
            ANA[analytics.py\nKPIs, Distributions, Scikit-Learn Evaluation]
            VIZ[visualizations.py\nPlotly Design System Charts]
        end

        subgraph Storage_Layer [Disk & Sample Assets]
            CSV1[(sample_reviews.csv\n50 Demo Records)]
            CSV2[(eval_labeled_reviews.csv\n30 Ground Truth Records)]
        end

        User --> Presentation_Layer
        Presentation_Layer <--> Session_Manager
        Presentation_Layer --> Core_Modules
        Core_Modules <--> Storage_Layer
    ```
    """)

    st.markdown(r"""
    ### 📐 Algorithmic Methodology & Mathematical Formulations
    
    #### 1. TextBlob Sentiment Analysis
    TextBlob calculates sentiment using a pattern dictionary that assigns semantic orientation to adjectives:
    - **Polarity ($P$)**: Real number in range $[-1.0, +1.0]$.
      - $P > +0.05 \implies$ **Positive**
      - $P < -0.05 \implies$ **Negative**
      - $-0.05 \le P \le +0.05 \implies$ **Neutral**
    - **Subjectivity ($S$)**: Real number in range $[0.0, 1.0]$.
      - $S \approx 0.0$: Objective factual statements.
      - $S \approx 1.0$: Highly opinionated or emotional prose.

    #### 2. VADER (Valence Aware Dictionary and sEntiment Reasoner)
    VADER applies heuristic rules incorporating capitalization, exclamation points, and negation modifiers:
    - Normalized compound score:
      $$c = \frac{\sum x}{\sqrt{(\sum x)^2 + \alpha}}$$
      where $x$ represents the valence of each lexicon match and $\alpha = 15$.
      - $c \ge +0.05 \implies$ **Positive**
      - $c \le -0.05 \implies$ **Negative**
      - $-0.05 < c < +0.05 \implies$ **Neutral**
    """)

# ----------------- TAB 3: BENCHMARK EVALUATION -----------------
with tab_eval:
    st.markdown("### 🧪 Empirical Evaluation on Ground-Truth Benchmark")
    st.markdown("""
    To demonstrate authentic scientific rigor without fabricating benchmark metrics, ReviewSense evaluates 
    its default **TextBlob** engine against a curated ground-truth test suite of human-labeled product reviews (`sample_data/eval_labeled_reviews.csv`).
    """)

    eval_df = DataManager.load_evaluation_dataset()
    if eval_df is not None and not eval_df.empty:
        # Run live inference on evaluation dataset
        predictions = []
        for text in eval_df["review_text"]:
            pred = default_nlp_engine.classify_tb_polarity(
                default_nlp_engine.analyze_textblob(text)["polarity"]
            )
            predictions.append(pred)

        eval_df_scored = eval_df.copy()
        eval_df_scored["pred_sentiment"] = predictions

        metrics = AnalyticsEngine.evaluate_model_on_ground_truth(
            eval_df_scored,
            ground_truth_col="true_sentiment",
            pred_col="pred_sentiment"
        )

        if metrics["status"] == "success":
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                render_kpi_card("Overall Accuracy", f"{metrics['accuracy'] * 100:.1f}%", f"{metrics['total_samples']} test cases", THEME_TOKENS["accent_primary"])
            with m2:
                render_kpi_card("Macro Precision", f"{metrics['macro_precision'] * 100:.1f}%", "Unweighted average", THEME_TOKENS["positive_color"])
            with m3:
                render_kpi_card("Macro Recall", f"{metrics['macro_recall'] * 100:.1f}%", "Sensitivity score", THEME_TOKENS["accent_secondary"])
            with m4:
                render_kpi_card("Macro F1-Score", f"{metrics['macro_f1'] * 100:.1f}%", "Harmonic mean", "#8B5CF6")

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

            # Confusion Matrix
            cm_col, table_col = st.columns([1, 1])
            with cm_col:
                st.plotly_chart(
                    create_confusion_matrix_heatmap(metrics["confusion_matrix"], metrics["labels"]),
                    use_container_width=True
                )

            with table_col:
                st.markdown("##### 📊 Detailed Classification Report")
                rep = metrics["classification_report"]
                rep_rows = []
                for label in metrics["labels"]:
                    if label in rep:
                        rep_rows.append({
                            "Class": label,
                            "Precision": round(rep[label]["precision"], 3),
                            "Recall": round(rep[label]["recall"], 3),
                            "F1-Score": round(rep[label]["f1-score"], 3),
                            "Support": int(rep[label]["support"])
                        })
                st.dataframe(pd.DataFrame(rep_rows), use_container_width=True)

            with st.expander("🔍 View Benchmark Evaluation Test Cases"):
                st.dataframe(eval_df_scored[["review_text", "true_sentiment", "pred_sentiment"]], use_container_width=True)
    else:
        st.warning("Evaluation dataset file `sample_data/eval_labeled_reviews.csv` could not be loaded.")

# ----------------- TAB 4: VIVA VOCE DEFENSE GUIDE -----------------
with tab_viva:
    st.markdown("### 🎓 Comprehensive Viva Voce Defense Guide")
    st.markdown("Essential technical interview questions and rigorous academic answers for project evaluation.")

    viva_qas = [
        (
            "1. What is the fundamental difference between Polarity and Subjectivity?",
            "Polarity measures the emotional orientation of a text on a scale from -1.0 (strongly negative) to +1.0 (strongly positive). Subjectivity measures the degree to which text expresses personal opinions, judgments, or beliefs (0.0 to 1.0) versus objective, factual assertions. An objective specification ('The battery is 5000mAh') has low subjectivity, whereas an emotional critique ('I despise this awful battery') has high subjectivity."
        ),
        (
            "2. Why did you implement dual analyzers (TextBlob & VADER)?",
            "TextBlob and VADER operate under different algorithmic paradigms. TextBlob uses a pattern dictionary averaging sentiment orientation of grammatical parts of speech, making it well-suited for general prose. VADER (Valence Aware Dictionary and sEntiment Reasoner) incorporates heuristic rules specifically tuned for social and review text, recognizing punctuation emphasis ('great!!!'), capitalization ('SUPERB'), and emoji valence. Comparing both provides deeper analytical insight and highlights edge cases."
        ),
        (
            "3. Why is standard stopword removal dangerous in Sentiment Analysis?",
            "Standard stopword lists (such as the default NLTK list) include negation words such as 'not', 'no', 'never', 'neither', and 'without'. If these are removed indiscriminately, a negative sentence like 'The camera is not good' is reduced to 'camera good', inverting the predicted sentiment to Positive. ReviewSense implements negation-aware stopword filtering to strictly protect 27 negation particles."
        ),
        (
            "4. How did you choose the threshold boundaries (-0.05 and +0.05)?",
            "A zero-width boundary (polarity > 0 is positive, < 0 is negative) suffers from extreme sensitivity to minor lexical noise. Reviews containing mild or neutral vocabulary can generate small non-zero floats (e.g. +0.02). Establishing a neutral threshold corridor [-0.05, +0.05] aligns with standard academic benchmarks (including VADER's published literature) and significantly reduces false-positive/negative misclassifications."
        ),
        (
            "5. What are the known limitations of lexicon-based sentiment analysis?",
            "Lexicon approaches struggle with: (1) Sarcasm and irony ('Oh great, another broken cord'), (2) Complex cross-sentence discourse where earlier negative statements are refuted later, (3) Domain-specific vocabulary (e.g. 'unpredictable plot' is positive in movies but negative in automotive brakes), and (4) Mixed aspect sentiments within a single sentence."
        ),
        (
            "6. What is CSV Formula Injection and how does ReviewSense mitigate it?",
            "When users export analysis results to open in spreadsheet software (Excel, LibreOffice, Google Sheets), malicious text starting with '=', '+', '-', or '@' can execute arbitrary spreadsheet commands or exfiltrate data (OWASP CSV Injection). ReviewSense sanitizes all exported string columns by prepending a single apostrophe to neutralize formula triggers while preserving legitimate numeric scores."
        ),
        (
            "7. How does the application preserve shared state across navigation?",
            "ReviewSense orchestrates a central data lifecycle in Streamlit's session state (`st.session_state['active_data']`). On first boot, the bundled demonstration dataset is verified and loaded. When users upload a new CSV, the active dataset is updated atomically only after successful schema validation and scoring, preventing session corruption if a malformed file is uploaded."
        )
    ]

    for q, a in viva_qas:
        with st.expander(f"📌 {q}"):
            st.markdown(a)
