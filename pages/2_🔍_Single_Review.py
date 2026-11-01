"""
ReviewSense - Page 2: Single Review Real-Time Intelligence
Performs instant NLP sentiment inference using TextBlob and VADER,
featuring polarity gauges, subjectivity spectrum, and score interpretation.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import src
src.reload_src_modules()

import streamlit as st
from src.config import AppConfig, THEME_TOKENS
from src.data_manager import DataManager
from src.nlp_engine import default_nlp_engine
from src.aspect_engine import default_aspect_engine
from src.visualizations import create_polarity_gauge
from src.ui_components import (
    inject_custom_css,
    render_sidebar,
    render_sentiment_badge,
    render_kpi_card
)

# Page Setup
st.set_page_config(
    page_title=f"{AppConfig.APP_NAME} - Single Review Analysis",
    page_icon="⚡",
    layout="wide"
)

inject_custom_css()
render_sidebar()

# Initialize session history for single reviews if not present
if "single_review_history" not in st.session_state:
    st.session_state["single_review_history"] = []

st.markdown("""
<div style="margin-bottom: 20px;">
    <h2 style="margin: 0; color: #F8FAFC;">🔍 Single Review Sentiment Analysis</h2>
    <p style="color: #94A3B8; font-size: 0.95rem; margin-top: 4px;">
        Evaluate individual customer feedback in real-time with dual-lexicon NLP, polarity gauges, and score breakdown.
    </p>
</div>
""", unsafe_allow_html=True)

# Curated Preset Examples
SAMPLE_PRESETS = {
    "Select an example review...": "",
    "Strongly Positive Review (Audio)": (
        "The active noise cancellation is absolutely remarkable! Sound clarity is crystal clear and "
        "extremely comfortable for long flights. Five stars!"
    ),
    "Strongly Negative Review (Electronics)": (
        "Terrible build quality. The unit stopped working after three days. Customer support was completely unhelpful "
        "and refused a refund. Worst purchase ever!"
    ),
    "Objective Neutral Review (Specifications)": (
        "The package arrived on Tuesday and includes an instruction manual with an AC power cable. Dimensions are 10 by 4 inches."
    ),
    "Complex Mixed Opinion (Computing)": (
        "The screen and keyboard tactile feedback are sublime, but the battery life is terribly disappointing and "
        "the fans get quite loud under load."
    ),
    "Negation Nuance (Kitchenware)": (
        "This chef knife is not bad at all, and it certainly does not feel cheap despite the modest price tag."
    )
}

# ----------------- ENHANCEMENT: ONE-CLICK QUICK INJECT PILLS -----------------
st.markdown("##### ⚡ Instant One-Click Test Scenarios")
pill_c1, pill_c2, pill_c3, pill_c4, pill_c5 = st.columns(5)
quick_injected = None

with pill_c1:
    if st.button("🎧 Noise Cancelling", use_container_width=True, help="Load strongly positive audio review"):
        quick_injected = SAMPLE_PRESETS["Strongly Positive Review (Audio)"]
with pill_c2:
    if st.button("⚡ Battery Drain", use_container_width=True, help="Load negative battery complaint"):
        quick_injected = "The battery drains 50% in two hours. Overheats constantly and shuts down without warning!"
with pill_c3:
    if st.button("⚖️ Mixed Opinion", use_container_width=True, help="Load mixed laptop review"):
        quick_injected = SAMPLE_PRESETS["Complex Mixed Opinion (Computing)"]
with pill_c4:
    if st.button("📦 Neutral Specs", use_container_width=True, help="Load objective specs statement"):
        quick_injected = SAMPLE_PRESETS["Objective Neutral Review (Specifications)"]
with pill_c5:
    if st.button("🛡️ Negation Test", use_container_width=True, help="Load negation phrase"):
        quick_injected = SAMPLE_PRESETS["Negation Nuance (Kitchenware)"]

col_input_left, col_input_right = st.columns([3, 1])

with col_input_right:
    st.markdown("##### 💡 Preset Selector")
    selected_preset = st.selectbox(
        "Choose an illustrative example:",
        options=list(SAMPLE_PRESETS.keys()),
        index=0,
        label_visibility="collapsed"
    )

if quick_injected:
    default_text = quick_injected
elif selected_preset != "Select an example review...":
    default_text = SAMPLE_PRESETS.get(selected_preset, "")
else:
    default_text = ""

with col_input_left:
    user_review = st.text_area(
        label="Enter product review text:",
        value=default_text,
        height=140,
        placeholder="Type or paste customer review text here (e.g., 'Fantastic sound and great battery, but a bit pricey...')"
    )

    # Word and character count display
    char_len = len(user_review)
    word_len = len(user_review.split()) if user_review.strip() else 0
    st.caption(f"📊 Character Count: **{char_len}** | Word Count: **{word_len}**")

analyze_btn = st.button("⚡ Run NLP Sentiment Analysis", type="primary", use_container_width=False)

if analyze_btn or quick_injected or (user_review and selected_preset != "Select an example review..."):
    if not user_review.strip():
        st.warning("⚠️ Please enter or select a non-empty review text to evaluate.")
    else:
        with st.spinner("Executing TextBlob and VADER inference..."):
            result = default_nlp_engine.analyze(user_review, include_vader=True)

            # Store in session history (latest first, max 10 entries)
            history_entry = {
                "text": user_review[:80] + ("..." if len(user_review) > 80 else ""),
                "polarity": result.polarity,
                "subjectivity": result.subjectivity,
                "sentiment": result.sentiment,
                "vader_sentiment": result.vader_sentiment
            }
            if not any(h["text"] == history_entry["text"] for h in st.session_state["single_review_history"]):
                st.session_state["single_review_history"].insert(0, history_entry)
                if len(st.session_state["single_review_history"]) > 10:
                    st.session_state["single_review_history"].pop()

        st.markdown("---")
        st.markdown("### 🎯 Inference Results & Detailed Breakdown")

        # Top Summary Card
        res_col1, res_col2, res_col3, res_col4 = st.columns(4)

        with res_col1:
            render_kpi_card(
                title="Primary Sentiment",
                value=result.sentiment,
                subtitle=f"Classified by TextBlob",
                border_color=THEME_TOKENS["positive_color"] if result.sentiment == "Positive" else (
                    THEME_TOKENS["negative_color"] if result.sentiment == "Negative" else THEME_TOKENS["neutral_color"]
                )
            )
        with res_col2:
            render_kpi_card(
                title="Polarity Score",
                value=f"{result.polarity:+.4f}",
                subtitle="Range: -1.0 (Neg) to +1.0 (Pos)",
                border_color=THEME_TOKENS["accent_primary"]
            )
        with res_col3:
            render_kpi_card(
                title="Subjectivity Score",
                value=f"{result.subjectivity:.4f}",
                subtitle="Range: 0.0 (Obj) to 1.0 (Subj)",
                border_color=THEME_TOKENS["accent_secondary"]
            )
        with res_col4:
            v_val = result.vader_sentiment if result.vader_sentiment else "N/A"
            v_sub = f"Compound: {result.vader_compound:+.4f}" if result.vader_compound is not None else "VADER Offline"
            render_kpi_card(
                title="VADER Sentiment",
                value=v_val,
                subtitle=v_sub,
                border_color="#8B5CF6"
            )

        # ----------------- DUAL-ENGINE CONSENSUS ARENA -----------------
        agree_status = result.agreement
        if agree_status is not None:
            if agree_status:
                seal_html = '<span style="color: #10B981; font-weight: 800; font-size: 1.1rem;">✓ Dual Consensus Verified (100% Agreement)</span>'
                seal_desc = "Both TextBlob and VADER independently arrived at the same sentiment classification."
                seal_border = "border: 1px solid rgba(16, 185, 129, 0.4); box-shadow: 0 0 16px rgba(16, 185, 129, 0.15);"
            else:
                seal_html = '<span style="color: #F59E0B; font-weight: 800; font-size: 1.1rem;">⚠️ Lexical Divergence Detected</span>'
                seal_desc = f"TextBlob classified as '{result.sentiment}', while VADER classified as '{result.vader_sentiment}'. This highlights nuanced language or capitalization emphasis."
                seal_border = "border: 1px solid rgba(245, 158, 11, 0.4); box-shadow: 0 0 16px rgba(245, 158, 11, 0.15);"

            st.markdown(f"""
            <div style="background: rgba(22, 30, 46, 0.9); border-radius: 10px; padding: 14px 20px; margin: 12px 0 16px 0; display: flex; justify-content: space-between; align-items: center; {seal_border}">
                <div>
                    <div>{seal_html}</div>
                    <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 2px;">{seal_desc}</div>
                </div>
                <div style="font-size: 0.75rem; background: rgba(255,255,255,0.06); padding: 4px 10px; border-radius: 6px; color: #CBD5E1; font-weight: 600;">
                    TB: {result.sentiment} | VADER: {result.vader_sentiment}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Visual Indicators Row
        g_col1, g_col2 = st.columns([1, 1])

        with g_col1:
            st.markdown("##### 🧭 Polarity Precision Gauge")
            st.plotly_chart(create_polarity_gauge(result.polarity), use_container_width=True)

        with g_col2:
            st.markdown("##### 📐 Subjectivity Spectrum")
            sub_pct = int(result.subjectivity * 100)
            st.markdown(f"""
            <div style="background: #1E293B; border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 20px; height: 180px; display: flex; flex-direction: column; justify-content: center;">
                <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #94A3B8; margin-bottom: 8px;">
                    <span>Factual / Objective (0.0)</span>
                    <span style="color: #F8FAFC; font-weight: 700;">{sub_pct}%</span>
                    <span>Emotional / Opinionated (1.0)</span>
                </div>
                <div style="background: rgba(255,255,255,0.08); border-radius: 6px; height: 12px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #06B6D4, #6366F1); width: {sub_pct}%; height: 100%; border-radius: 6px;"></div>
                </div>
                <div style="margin-top: 14px; font-size: 0.82rem; color: #CBD5E1;">
                    {'Higher subjectivity indicates the presence of emotional modifiers, personal judgments, or subjective adjectives.' if result.subjectivity >= 0.5 else 'Lower subjectivity indicates factual specifications or restrained reporting.'}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Academic Explanation & Lexicon Comparison
        st.markdown("##### 📝 Score Interpretation & Analyzer Agreement")
        st.info(result.explanation)

        # ----------------- ENHANCEMENT: LIVE ASPECT EXTRACTION -----------------
        extracted_aspects = default_aspect_engine.extract_aspects_from_text(user_review)
        if extracted_aspects:
            st.markdown("##### 🎯 Granular Aspect Sentiment Extraction")
            aspect_cols = st.columns(min(4, len(extracted_aspects)))
            for idx, (asp_name, asp_data) in enumerate(extracted_aspects.items()):
                target_col = aspect_cols[idx % len(aspect_cols)]
                with target_col:
                    s_color = (
                        THEME_TOKENS["positive_color"] if asp_data["sentiment"] == "Positive"
                        else (THEME_TOKENS["negative_color"] if asp_data["sentiment"] == "Negative" else THEME_TOKENS["neutral_color"])
                    )
                    st.markdown(f"""
                    <div style="background: rgba(30, 41, 59, 0.7); border-left: 3px solid {s_color}; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px;">
                        <div style="font-size: 0.75rem; color: #94A3B8; font-weight: 600; text-transform: uppercase;">{asp_name}</div>
                        <div style="font-size: 0.92rem; font-weight: 700; color: {s_color}; margin: 2px 0;">{asp_data['sentiment']} ({asp_data['polarity']:+.2f})</div>
                        <div style="font-size: 0.76rem; color: #CBD5E1; font-style: italic;">"{asp_data['snippet'][:75]}..."</div>
                    </div>
                    """, unsafe_allow_html=True)

        # ----------------- ENHANCEMENT: SENTENCE-BY-SENTENCE DISCOURSE MAP -----------------
        import re
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", user_review) if s.strip()]
        if len(sentences) > 1:
            st.markdown("##### 📑 Multi-Sentence Discourse Map")
            sentence_pills = ""
            for s_idx, sent_text in enumerate(sentences):
                s_res = default_nlp_engine.analyze(sent_text, include_vader=False)
                sent_color = (
                    THEME_TOKENS["positive_color"] if s_res.sentiment == "Positive"
                    else (THEME_TOKENS["negative_color"] if s_res.sentiment == "Negative" else THEME_TOKENS["neutral_color"])
                )
                sentence_pills += f"""
                <div style="background: rgba(22, 30, 46, 0.85); border-left: 3px solid {sent_color}; border-radius: 6px; padding: 8px 12px; margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center; gap: 10px;">
                    <span style="font-size: 0.84rem; color: #F8FAFC;">{sent_text}</span>
                    <span style="font-size: 0.75rem; font-weight: 700; color: {sent_color}; background: rgba(255,255,255,0.05); padding: 2px 8px; border-radius: 4px; white-space: nowrap;">{s_res.sentiment} ({s_res.polarity:+.2f})</span>
                </div>
                """
            st.markdown(sentence_pills, unsafe_allow_html=True)

        # ----------------- ENHANCEMENT: LINGUISTIC INTENSITY & CONTRAST -----------------
        has_contrast = any(w in user_review.lower().split() for w in ["but", "however", "although", "yet", "though"])
        excl_count = user_review.count("!")
        upper_tokens = [w for w in user_review.split() if w.isupper() and len(w) > 1]
        
        with st.expander("🔬 Advanced Linguistic Nuance & Intensity Telemetry"):
            lc1, lc2, lc3 = st.columns(3)
            with lc1:
                st.metric("Exclamation Mark Intensity", f"{excl_count} marks", delta="High emphasis" if excl_count >= 2 else "Normal")
            with lc2:
                st.metric("Capitalized Emphasis Tokens", f"{len(upper_tokens)} words", delta=", ".join(upper_tokens[:3]) if upper_tokens else "None")
            with lc3:
                st.metric("Contrastive Conjunction", "Detected" if has_contrast else "None", delta="Mixed opinion risk" if has_contrast else "Uniform stance", delta_color="inverse" if has_contrast else "normal")

            if result.vader_compound is not None:
                st.markdown("---")
                v_c1, v_c2, v_c3, v_c4 = st.columns(4)
                v_c1.metric("VADER Positivity", f"{result.vader_pos:.3f}")
                v_c2.metric("VADER Neutrality", f"{result.vader_neu:.3f}")
                v_c3.metric("VADER Negativity", f"{result.vader_neg:.3f}")
                v_c4.metric("Compound Score", f"{result.vader_compound:+.3f}")

                st.markdown("""
                **Lexicon Mechanics**:
                - **TextBlob** computes polarity using semantic orientation of adjectives and adverbs in pattern dictionaries.
                - **VADER** incorporates grammatical heuristics (capitalization emphasis, exclamation marks, and negation scope).
                - Minor divergence between analyzers is normal and reflects distinct scoring philosophies rather than model corruption.
                """)

# Session Evaluation History
if st.session_state["single_review_history"]:
    st.markdown("---")
    st.markdown("### 🕒 Recent Inferences This Session")
    import pandas as pd
    hist_df = pd.DataFrame(st.session_state["single_review_history"])
    st.dataframe(hist_df, use_container_width=True)

# Guided Next Steps
st.markdown("---")
st.markdown("### 👉 Where to Next?")
nx1, nx2 = st.columns(2)
with nx1:
    if st.button("📁 Analyze an Entire Spreadsheet (Bulk CSV) →", use_container_width=True):
        st.switch_page("pages/3_📁_Bulk_Analysis.py")
with nx2:
    if st.button("🤖 Ask Questions About Reviews in Plain English →", use_container_width=True):
        st.switch_page("pages/8_🤖_AI_Copilot.py")

