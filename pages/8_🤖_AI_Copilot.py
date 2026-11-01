"""
ReviewSense - Page 8: AI Review Intelligence Copilot
Conversational intelligence assistant providing natural language Q&A,
aspect deep-dives, complaint diagnostics, anomaly hunting, and instant review scoring.
Adheres strictly to zero-fabrication rules: all analytical answers are computed
in real time from actual active dataset execution.
"""

import sys
from pathlib import Path
import json

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import src
src.reload_src_modules()

import streamlit as st
import pandas as pd

from src.config import AppConfig, THEME_TOKENS
from src.data_manager import DataManager
from src.copilot_engine import default_copilot_engine
from src.ui_components import (
    inject_custom_css,
    render_sidebar,
    render_sentiment_badge,
    render_kpi_card
)

# Page Setup
st.set_page_config(
    page_title=f"{AppConfig.APP_NAME} - AI Review Copilot",
    page_icon="🤖",
    layout="wide"
)

inject_custom_css()
render_sidebar()

# Ensure active dataset is initialized in session state
if "active_df" not in st.session_state or st.session_state["active_df"] is None:
    demo_df, demo_meta = DataManager.load_demonstration_data()
    st.session_state["active_df"] = demo_df
    st.session_state["dataset_metadata"] = demo_meta

active_df = st.session_state["active_df"]
meta = st.session_state.get("dataset_metadata", {})
total_rows = len(active_df) if active_df is not None else 0
source_label = meta.get("source", "Demo Dataset")

# Header Section
st.markdown(f"""
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
    <div>
        <h1 style="margin: 0; font-size: 2.1rem; font-weight: 800; color: #F8FAFC; letter-spacing: -0.02em;">
            🤖 AI Review Intelligence Copilot
        </h1>
        <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 0.92rem;">
            Natural language conversational assistant for dataset queries, aspect diagnostics, and instant review evaluation.
        </p>
    </div>
    <div style="text-align: right; background: rgba(22, 30, 46, 0.85); border: 1px solid rgba(255,255,255,0.08); padding: 8px 16px; border-radius: 10px;">
        <span style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">Connected Ingestion</span><br>
        <span style="font-size: 0.88rem; color: #6366F1; font-weight: 700;">{source_label} ({total_rows:,} records)</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize Copilot Chat History
INITIAL_WELCOME = {
    "role": "assistant",
    "content": f"""👋 **Welcome to the ReviewSense AI Copilot!**

I am your conversational data analyst. I analyze your active dataset (**{total_rows:,} reviews**) and execute real-time dual-engine NLP inference on demand.

**What you can ask me:**
- 📊 *"What is our Net Sentiment Score and overall health?"*
- 🔋 *"How is customer feedback on battery life?"*
- 🎧 *"What are users saying about sound quality?"*
- 🛑 *"What are the biggest customer complaints and what should we fix?"*
- ⚠️ *"Are there any rating or sentiment anomalies?"*
- 📦 *"Compare category performance"*
- ⚡ Or score any custom text: `Analyze: Exceptional build quality, but the price is too steep.`

Select a quick prompt below or type your question in the chat bar!"""
}

if "copilot_messages" not in st.session_state:
    st.session_state["copilot_messages"] = [INITIAL_WELCOME]

# Suggested Quick Action Pills
st.markdown("##### ⚡ Quick Analysis Prompts")
p_col1, p_col2, p_col3, p_col4, p_col5, p_col6 = st.columns(6)

selected_prompt = None

with p_col1:
    if st.button("📊 Health & NSS", use_container_width=True, help="Overview of Net Sentiment Score and KPIs"):
        selected_prompt = "What is our Net Sentiment Score and overall health?"
with p_col2:
    if st.button("🔋 Battery Life", use_container_width=True, help="Aspect deep-dive into Battery & Power"):
        selected_prompt = "What are users saying about battery life?"
with p_col3:
    if st.button("🎧 Sound Quality", use_container_width=True, help="Aspect deep-dive into Audio & Acoustics"):
        selected_prompt = "What are users saying about sound quality?"
with p_col4:
    if st.button("🛑 Top Complaints", use_container_width=True, help="Extract top negative drivers & pain points"):
        selected_prompt = "What are the biggest negative drivers and customer complaints?"
with p_col5:
    if st.button("⚠️ Find Anomalies", use_container_width=True, help="Scan for rating-sentiment mismatches"):
        selected_prompt = "Are there any rating or sentiment anomalies in the dataset?"
with p_col6:
    if st.button("📦 Categories", use_container_width=True, help="Compare sentiment across product categories"):
        selected_prompt = "Compare category performance"

st.markdown("---")

# Top Controls Bar: Clear History & Export Transcript
ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([6, 1.5, 1.5])

with ctrl_col2:
    if st.button("🗑️ Clear History", use_container_width=True, help="Reset conversation history"):
        st.session_state["copilot_messages"] = [INITIAL_WELCOME]
        st.rerun()

with ctrl_col3:
    transcript_md = "\n\n---\n\n".join([f"**{m['role'].upper()}**:\n{m['content']}" for m in st.session_state["copilot_messages"]])
    st.download_button(
        label="📥 Export Chat",
        data=transcript_md,
        file_name="reviewsense_copilot_transcript.md",
        mime="text/markdown",
        use_container_width=True,
        help="Download complete conversational transcript as Markdown"
    )

# Render Chat History
chat_container = st.container()

with chat_container:
    for message in st.session_state["copilot_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)

# Chat Input Handler
user_input = st.chat_input("Ask ReviewSense anything about your reviews, dimensions, or custom text...")

# Process prompt from either text input or quick prompt pill
prompt_to_process = selected_prompt if selected_prompt else user_input

if prompt_to_process:
    # Append user message
    st.session_state["copilot_messages"].append({"role": "user", "content": prompt_to_process})
    with st.chat_message("user"):
        st.markdown(prompt_to_process)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing dataset telemetry and executing NLP engine..."):
            response = default_copilot_engine.process_query(prompt_to_process, df=active_df)
            st.markdown(response, unsafe_allow_html=True)

    # Append assistant message to history
    st.session_state["copilot_messages"].append({"role": "assistant", "content": response})
    st.rerun()
