"""
ReviewSense - Executive Intelligence Report Generator
Generates a standalone, beautifully styled HTML executive report
containing Net Sentiment Score (NSS), key drivers, aspect telemetry,
and actionable customer experience recommendations.
"""

import datetime
from typing import Dict, Any, Optional
import pandas as pd

from src.analytics import AnalyticsEngine
from src.aspect_engine import default_aspect_engine


class ReportGenerator:
    """Creates downloadable, printable HTML executive intelligence reports."""

    @classmethod
    def generate_html_report(
        cls,
        df: pd.DataFrame,
        metadata: Dict[str, Any],
        text_col: str = "review_text"
    ) -> str:
        """Constructs an executive HTML report."""
        kpis = AnalyticsEngine.calculate_kpis(df)
        aspect_df = default_aspect_engine.analyze_dataset_aspects(df, text_col=text_col)
        extremes = AnalyticsEngine.get_extreme_reviews(df, n=3, text_col=text_col)

        # Calculate Net Sentiment Score: Pos % - Neg %
        nss = round(kpis["positive_pct"] - kpis["negative_pct"], 1)
        nss_badge_color = "#10B981" if nss >= 30 else ("#F59E0B" if nss >= 0 else "#F43F5E")

        # Aspect rows HTML
        aspect_rows_html = ""
        if not aspect_df.empty:
            for _, r in aspect_df.iterrows():
                aspect_rows_html += f"""
                <tr>
                    <td style="font-weight: 600;">{r['aspect']}</td>
                    <td>{r['total_mentions']}</td>
                    <td style="color: #10B981; font-weight: 600;">{r['positive_rate']}%</td>
                    <td style="color: #F43F5E; font-weight: 600;">{r['negative_rate']}%</td>
                    <td style="font-weight: 600;">{r['avg_polarity']:+.2f}</td>
                    <td><b>{r['net_aspect_score']:+0.1f}</b></td>
                </tr>
                """
        else:
            aspect_rows_html = "<tr><td colspan='6' style='text-align:center;'>No specific aspect mentions detected.</td></tr>"

        # Quotes HTML
        pos_quotes_html = ""
        for _, r in extremes["top_positive"].iterrows():
            pos_quotes_html += f"""
            <div style="background: #F0FDF4; border-left: 4px solid #10B981; padding: 10px 14px; margin-bottom: 8px; border-radius: 4px;">
                <div style="font-size: 11px; color: #15803D; font-weight: 700;">Polarity: {r.get('tb_polarity', 0):+.2f} | {r.get('product_name', 'Product')}</div>
                <div style="font-size: 13px; color: #166534; margin-top: 4px;">"{r.get(text_col, '')}"</div>
            </div>
            """

        neg_quotes_html = ""
        for _, r in extremes["top_negative"].iterrows():
            neg_quotes_html += f"""
            <div style="background: #FFF1F2; border-left: 4px solid #F43F5E; padding: 10px 14px; margin-bottom: 8px; border-radius: 4px;">
                <div style="font-size: 11px; color: #BE123C; font-weight: 700;">Polarity: {r.get('tb_polarity', 0):+.2f} | {r.get('product_name', 'Product')}</div>
                <div style="font-size: 13px; color: #9F1239; margin-top: 4px;">"{r.get(text_col, '')}"</div>
            </div>
            """

        # Full HTML markup
        now_str = datetime.datetime.now().strftime("%B %d, %Y - %H:%M UTC")
        source_name = metadata.get("source", "ReviewSense Ingested Dataset")

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ReviewSense Executive Intelligence Report</title>
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #1E293B;
        background-color: #F8FAFC;
        margin: 0;
        padding: 30px;
    }}
    .container {{
        max-width: 900px;
        margin: 0 auto;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 40px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    }}
    .header {{
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 20px;
        margin-bottom: 30px;
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
    }}
    .title {{
        font-size: 26px;
        font-weight: 800;
        color: #0F172A;
        margin: 0;
    }}
    .subtitle {{
        color: #64748B;
        font-size: 14px;
        margin-top: 4px;
    }}
    .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 30px;
    }}
    .kpi-card {{
        background: #F1F5F9;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }}
    .kpi-num {{
        font-size: 24px;
        font-weight: 800;
        color: #0F172A;
        margin: 4px 0;
    }}
    .kpi-lbl {{
        font-size: 11px;
        text-transform: uppercase;
        color: #64748B;
        font-weight: 700;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 12px;
        margin-bottom: 30px;
    }}
    th {{
        background: #F8FAFC;
        border-bottom: 2px solid #CBD5E1;
        padding: 10px 14px;
        text-align: left;
        font-size: 12px;
        text-transform: uppercase;
        color: #475569;
    }}
    td {{
        padding: 10px 14px;
        border-bottom: 1px solid #E2E8F0;
        font-size: 13px;
    }}
    .section-title {{
        font-size: 18px;
        font-weight: 700;
        color: #0F172A;
        margin: 28px 0 10px 0;
    }}
    .badge {{
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        color: #FFFFFF;
    }}
    .footer {{
        border-top: 1px solid #E2E8F0;
        margin-top: 40px;
        padding-top: 20px;
        font-size: 11px;
        color: #94A3B8;
        text-align: center;
    }}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <div>
            <h1 class="title">⚡ ReviewSense Executive Report</h1>
            <div class="subtitle">Dataset: <b>{source_name}</b> | Generated: {now_str}</div>
        </div>
        <div>
            <span class="badge" style="background-color: {nss_badge_color};">Net Sentiment Score: {nss:+0.1f}</span>
        </div>
    </div>

    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-lbl">Total Valid Reviews</div>
            <div class="kpi-num">{kpis['valid_reviews']:,}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-lbl">Positive Rate</div>
            <div class="kpi-num" style="color: #10B981;">{kpis['positive_pct']}%</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-lbl">Negative Rate</div>
            <div class="kpi-num" style="color: #F43F5E;">{kpis['negative_pct']}%</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-lbl">Average Polarity</div>
            <div class="kpi-num" style="color: #6366F1;">{kpis['avg_polarity']:+.3f}</div>
        </div>
    </div>

    <div class="section-title">🎯 Granular Aspect-Based Intelligence Scorecard</div>
    <table>
        <thead>
            <tr>
                <th>Aspect Dimension</th>
                <th>Mentions</th>
                <th>Positive %</th>
                <th>Negative %</th>
                <th>Mean Polarity</th>
                <th>Net Score</th>
            </tr>
        </thead>
        <tbody>
            {aspect_rows_html}
        </tbody>
    </table>

    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
        <div>
            <div class="section-title" style="color: #15803D;">🌟 Top Customer Praise Drivers</div>
            {pos_quotes_html}
        </div>
        <div>
            <div class="section-title" style="color: #BE123C;">⚠️ Critical Dissatisfaction Areas</div>
            {neg_quotes_html}
        </div>
    </div>

    <div class="footer">
        Generated autonomously by <b>ReviewSense Platform</b> • Dual-Lexicon NLP Architecture (TextBlob + VADER)
    </div>
</div>
</body>
</html>
        """
        return html_content
