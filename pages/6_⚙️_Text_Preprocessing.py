"""
ReviewSense - Page 6: Interactive NLP Preprocessing Laboratory
Visualizes step-by-step lexical transformations, tokenization,
and the critical importance of negation preservation in sentiment analysis.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from src.config import AppConfig, NEGATION_WORDS, THEME_TOKENS
from src.data_manager import DataManager
from src.preprocessing import (
    preprocess_review,
    get_step_by_step_transformations,
    tokenize_text,
    clean_html,
    normalize_whitespace,
    remove_punctuation
)
from src.nlp_engine import default_nlp_engine
from src.ui_components import (
    inject_custom_css,
    render_sidebar,
    render_kpi_card
)

# Page Setup
st.set_page_config(
    page_title=f"{AppConfig.APP_NAME} - Text Preprocessing Laboratory",
    page_icon="⚡",
    layout="wide"
)

inject_custom_css()
render_sidebar()

st.markdown("""
<div style="margin-bottom: 20px;">
    <h2 style="margin: 0; color: #F8FAFC;">⚙️ Text Preprocessing & NLP Pipeline Lab</h2>
    <p style="color: #94A3B8; font-size: 0.95rem; margin-top: 4px;">
        Explore lexical normalization, tokenization, stopword filtering, and observe how preprocessing transforms text and affects downstream sentiment.
    </p>
</div>
""", unsafe_allow_html=True)

# Curated Negation-Sensitive Presets
SAMPLE_LAB_INPUTS = {
    "Negation Inversion Case (Critical)": "The camera is not good at all, and the battery does not last long.",
    "HTML Entities & Formatting": "<p>Superb performance &amp; fast shipping! Price was &gt; $500 but worth every penny.</p>",
    "Informal Customer Feedback": "OMG!! Absolutely LOVED the noise cancellation... but charging was so slow :-(",
    "Factual Specification": "The laptop weighs 1.2 kilograms, featuring 16GB LPDDR5 RAM and dual Thunderbolt 4 ports."
}

col_preset, col_toggle = st.columns([2, 1])

with col_preset:
    preset_choice = st.selectbox("Choose a sample text to test:", list(SAMPLE_LAB_INPUTS.keys()), index=0)

with col_toggle:
    preserve_negs = st.checkbox("Protect Negation Words (Recommended)", value=True, help="Prevents words like 'not', 'no', 'never' from being stripped as stopwords.")

input_text = st.text_area(
    "Edit or input text to preprocess:",
    value=SAMPLE_LAB_INPUTS[preset_choice],
    height=90
)

st.markdown("---")

# ----------------- SECTION 1: STEP-BY-STEP PIPELINE -----------------
st.markdown("### 🧬 Step-by-Step Pipeline Transformations")

steps = get_step_by_step_transformations(input_text, preserve_negations=preserve_negs)

for s in steps:
    with st.container():
        st.markdown(f"""
        <div style="background: #1E293B; border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 14px 18px; margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <span style="font-weight: 700; color: #6366F1; font-size: 0.88rem;">Step {s['step_num']}: {s['name']}</span>
                <span style="font-size: 0.75rem; color: #94A3B8; background: rgba(255,255,255,0.05); padding: 2px 8px; border-radius: 4px;">Tokens: {s['token_count']}</span>
            </div>
            <p style="font-size: 0.8rem; color: #94A3B8; margin: 0 0 8px 0;">{s['description']}</p>
            <div style="background: #0F172A; border: 1px solid rgba(255,255,255,0.05); border-radius: 6px; padding: 10px 14px; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #E2E8F0; overflow-x: auto;">
                {s['text']}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ----------------- SECTION 2: SENTIMENT IMPACT ANALYSIS -----------------
st.markdown("---")
st.markdown("### ⚖️ The Sentiment Impact: Original vs. Naive vs. Negation-Aware")

# 1. Original text scoring
orig_res = default_nlp_engine.analyze(input_text)

# 2. Naive preprocessing (removes ALL stopwords including negations)
naive_clean = preprocess_review(
    input_text,
    strip_html=True,
    normalize_spaces=True,
    to_lower=True,
    strip_punct=True,
    filter_stopwords=True,
    preserve_negations=False
)
naive_res = default_nlp_engine.analyze(naive_clean)

# 3. Negation-aware preprocessing (protects 'not', 'never', 'no', etc.)
smart_clean = preprocess_review(
    input_text,
    strip_html=True,
    normalize_spaces=True,
    to_lower=True,
    strip_punct=True,
    filter_stopwords=True,
    preserve_negations=True
)
smart_res = default_nlp_engine.analyze(smart_clean)

cmp1, cmp2, cmp3 = st.columns(3)

with cmp1:
    st.markdown("##### 1. Original Raw Text")
    st.markdown(f"""
    <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 14px; min-height: 140px;">
        <div style="font-size: 0.82rem; color: #94A3B8;">Raw Text:</div>
        <div style="font-size: 0.84rem; color: #F8FAFC; margin-bottom: 10px;">"{input_text}"</div>
        <div style="font-size: 0.88rem; font-weight: 700; color: #38BDF8;">Polarity: {orig_res.polarity:+.3f}</div>
        <div style="font-size: 0.82rem; color: #CBD5E1;">Class: <b>{orig_res.sentiment}</b></div>
    </div>
    """, unsafe_allow_html=True)

with cmp2:
    st.markdown("##### 2. Naive Stopwords (Flawed)")
    # Highlight risk if polarity flipped
    flipped = (orig_res.sentiment != naive_res.sentiment)
    border_col = "#F43F5E" if flipped else "rgba(255,255,255,0.08)"
    st.markdown(f"""
    <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid {border_col}; border-radius: 8px; padding: 14px; min-height: 140px;">
        <div style="font-size: 0.82rem; color: #94A3B8;">Negations Dropped:</div>
        <div style="font-size: 0.84rem; color: #F8FAFC; margin-bottom: 10px;">"{naive_clean}"</div>
        <div style="font-size: 0.88rem; font-weight: 700; color: {'#F43F5E' if flipped else '#38BDF8'};">Polarity: {naive_res.polarity:+.3f}</div>
        <div style="font-size: 0.82rem; color: {'#F43F5E' if flipped else '#CBD5E1'};">Class: <b>{naive_res.sentiment}</b> {'⚠️ INVERTED!' if flipped else ''}</div>
    </div>
    """, unsafe_allow_html=True)

with cmp3:
    st.markdown("##### 3. Negation-Aware Cleaned")
    st.markdown(f"""
    <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid #10B981; border-radius: 8px; padding: 14px; min-height: 140px;">
        <div style="font-size: 0.82rem; color: #94A3B8;">Negations Preserved:</div>
        <div style="font-size: 0.84rem; color: #F8FAFC; margin-bottom: 10px;">"{smart_clean}"</div>
        <div style="font-size: 0.88rem; font-weight: 700; color: #10B981;">Polarity: {smart_res.polarity:+.3f}</div>
        <div style="font-size: 0.82rem; color: #CBD5E1;">Class: <b>{smart_res.sentiment}</b> ✓ Protected</div>
    </div>
    """, unsafe_allow_html=True)

# Academic Explanatory Note
st.markdown("---")
st.info("""
**Academic Takeaway for Sentiment Analysis**:
In generic text classification (e.g. topic categorization), removing common stopwords like *not, neither, no* reduces noise and feature dimensionality. 
However, in **sentiment analysis**, stopwords like *not* invert valence (*"not good"* -> *"good"*). ReviewSense enforces **negation-aware filtering**, protecting 27 sentiment-critical negation particles from accidental erasure.
""")

# ----------------- ENHANCEMENT: PART-OF-SPEECH (POS) TAGGING -----------------
st.markdown("---")
st.markdown("### 🏷️ Lexical Part-of-Speech (POS) Tagging & Modifier Extraction")
st.markdown("TextBlob extracts grammatical tags to identify adjectives and adverbs that carry emotional valence:")

from textblob import TextBlob

try:
    blob = TextBlob(input_text)
    pos_tags = blob.tags
except Exception:
    # Graceful fallback if POS tagger resource is not loaded
    tokens = input_text.split()
    pos_tags = [(t, "WORD") for t in tokens]

pos_pills_html = ""
for word, tag in pos_tags:
    if tag.startswith("JJ"):  # Adjective
        badge_style = "background: rgba(99, 102, 241, 0.2); border: 1px solid #6366F1; color: #A5B4FC;"
        desc = "Adjective (Sentiment Carrier)"
    elif tag.startswith("RB"):  # Adverb
        badge_style = "background: rgba(139, 92, 246, 0.2); border: 1px solid #8B5CF6; color: #C4B5FD;"
        desc = "Adverb (Modifier)"
    elif tag.startswith("NN"):  # Noun
        badge_style = "background: rgba(16, 185, 129, 0.2); border: 1px solid #10B981; color: #6EE7B7;"
        desc = "Noun (Aspect / Entity)"
    elif tag.startswith("VB"):  # Verb
        badge_style = "background: rgba(245, 158, 11, 0.2); border: 1px solid #F59E0B; color: #FCD34D;"
        desc = "Verb (Action)"
    else:
        badge_style = "background: rgba(148, 163, 184, 0.1); border: 1px solid rgba(255,255,255,0.08); color: #94A3B8;"
        desc = tag

    pos_pills_html += f"""
    <div style="display: inline-flex; flex-direction: column; align-items: center; {badge_style} border-radius: 6px; padding: 4px 8px; margin: 4px; font-size: 0.8rem;">
        <span style="font-weight: 700; font-family: 'JetBrains Mono', monospace;">{word}</span>
        <span style="font-size: 0.65rem; text-transform: uppercase;">{tag}</span>
    </div>
    """

st.markdown(f"<div style='background: #1E293B; border-radius: 8px; padding: 16px; border: 1px solid rgba(255,255,255,0.08);'>{pos_pills_html}</div>", unsafe_allow_html=True)

