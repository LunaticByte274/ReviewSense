"""
ReviewSense - Data Visualization Layer
Plotly-powered interactive visualizations adhering to the Linear/Stripe-inspired
design system tokens and exact color semantics.
"""

from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.config import THEME_TOKENS, SentimentThresholds

# Standard color mapping
SENTIMENT_COLORS = {
    "Positive": THEME_TOKENS["positive_color"],
    "Negative": THEME_TOKENS["negative_color"],
    "Neutral": THEME_TOKENS["neutral_color"],
}

CHART_LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif", color=THEME_TOKENS["text_primary"], size=12),
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)


def create_sentiment_donut_chart(df: pd.DataFrame) -> go.Figure:
    """Generates a modern donut chart representing sentiment proportions."""
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False, font=dict(color=THEME_TOKENS["text_secondary"]))
        fig.update_layout(**CHART_LAYOUT_DEFAULTS)
        return fig

    sent_col = "tb_sentiment" if "tb_sentiment" in df.columns else "sentiment"
    counts = df[sent_col].value_counts().reset_index()
    counts.columns = ["Sentiment", "Count"]

    color_seq = [SENTIMENT_COLORS.get(s, THEME_TOKENS["neutral_color"]) for s in counts["Sentiment"]]

    fig = go.Figure(data=[
        go.Pie(
            labels=counts["Sentiment"],
            values=counts["Count"],
            hole=0.62,
            marker=dict(colors=color_seq, line=dict(color="#0F172A", width=2)),
            textinfo="percent+label",
            hoverinfo="label+value+percent",
            textfont=dict(size=13, color="#FFFFFF")
        )
    ])

    total = int(counts["Count"].sum())
    fig.add_annotation(
        text=f"<b>{total}</b><br><span style='font-size:11px; color:#94A3B8;'>Reviews</span>",
        x=0.5, y=0.5,
        font=dict(size=20, color=THEME_TOKENS["text_primary"]),
        showarrow=False
    )

    fig.update_layout(
        **CHART_LAYOUT_DEFAULTS,
        title=dict(text="<b>Sentiment Breakdown</b>", font=dict(size=15)),
        showlegend=False,
        height=320
    )
    return fig


def create_sentiment_bar_chart(df: pd.DataFrame) -> go.Figure:
    """Creates a clean bar chart showing counts and percentages of sentiments."""
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(**CHART_LAYOUT_DEFAULTS)
        return fig

    sent_col = "tb_sentiment" if "tb_sentiment" in df.columns else "sentiment"
    counts = df[sent_col].value_counts().reindex(["Positive", "Neutral", "Negative"]).fillna(0).reset_index()
    counts.columns = ["Sentiment", "Count"]
    total = max(1, counts["Count"].sum())
    counts["Percentage"] = (counts["Count"] / total * 100).round(1)

    colors = [SENTIMENT_COLORS.get(s) for s in counts["Sentiment"]]

    fig = go.Figure(data=[
        go.Bar(
            x=counts["Sentiment"],
            y=counts["Count"],
            marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.1)", width=1)),
            text=[f"<b>{int(c)}</b> ({p}%)" for c, p in zip(counts["Count"], counts["Percentage"])],
            textposition="auto",
            textfont=dict(color="#FFFFFF", size=12)
        )
    ])

    fig.update_layout(
        **CHART_LAYOUT_DEFAULTS,
        title=dict(text="<b>Volume by Sentiment Class</b>", font=dict(size=15)),
        xaxis=dict(showgrid=False, title=""),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title="Review Count"),
        height=320
    )
    return fig


def create_polarity_histogram(df: pd.DataFrame, thresholds: SentimentThresholds = SentimentThresholds()) -> go.Figure:
    """Displays polarity score distribution with exact threshold boundaries."""
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(**CHART_LAYOUT_DEFAULTS)
        return fig

    pol_col = "tb_polarity" if "tb_polarity" in df.columns else "polarity"
    valid_pols = df[pol_col].dropna()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=valid_pols,
        nbinsx=30,
        marker=dict(
            color=THEME_TOKENS["accent_primary"],
            line=dict(color="rgba(255,255,255,0.15)", width=1)
        ),
        hovertemplate="Polarity: %{x}<br>Count: %{y}<extra></extra>"
    ))

    # Add vertical dashed lines for thresholds
    fig.add_vline(
        x=thresholds.TB_NEGATIVE_MAX,
        line_dash="dash",
        line_color=THEME_TOKENS["negative_color"],
        annotation_text="Neg Threshold (-0.05)",
        annotation_position="top left",
        annotation_font=dict(size=10, color=THEME_TOKENS["negative_color"])
    )
    fig.add_vline(
        x=thresholds.TB_POSITIVE_MIN,
        line_dash="dash",
        line_color=THEME_TOKENS["positive_color"],
        annotation_text="Pos Threshold (+0.05)",
        annotation_position="top right",
        annotation_font=dict(size=10, color=THEME_TOKENS["positive_color"])
    )

    fig.update_layout(
        **CHART_LAYOUT_DEFAULTS,
        title=dict(text="<b>Polarity Score Distribution (-1.0 to +1.0)</b>", font=dict(size=15)),
        xaxis=dict(
            range=[-1.05, 1.05],
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            title="Polarity Score"
        ),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title="Count"),
        height=320
    )
    return fig


def create_subjectivity_distribution(df: pd.DataFrame) -> go.Figure:
    """Displays subjectivity score distribution (0.0 Objective to 1.0 Subjective)."""
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(**CHART_LAYOUT_DEFAULTS)
        return fig

    sub_col = "tb_subjectivity" if "tb_subjectivity" in df.columns else "subjectivity"
    valid_subs = df[sub_col].dropna()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=valid_subs,
        nbinsx=25,
        marker=dict(
            color=THEME_TOKENS["accent_secondary"],
            line=dict(color="rgba(255,255,255,0.15)", width=1)
        ),
        hovertemplate="Subjectivity: %{x}<br>Count: %{y}<extra></extra>"
    ))

    fig.update_layout(
        **CHART_LAYOUT_DEFAULTS,
        title=dict(text="<b>Subjectivity Spectrum (0.0 Objective -> 1.0 Subjective)</b>", font=dict(size=15)),
        xaxis=dict(
            range=[-0.02, 1.02],
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            title="Subjectivity Score"
        ),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title="Count"),
        height=320
    )
    return fig


def create_category_sentiment_chart(cat_df: pd.DataFrame) -> go.Figure:
    """Stacked horizontal bar chart of sentiment by category."""
    if cat_df is None or cat_df.empty:
        fig = go.Figure()
        fig.update_layout(**CHART_LAYOUT_DEFAULTS)
        return fig

    cat_col = cat_df.columns[0]
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=cat_df[cat_col],
        x=cat_df["positive"],
        name="Positive",
        orientation="h",
        marker=dict(color=THEME_TOKENS["positive_color"]),
        hovertemplate="%{y}<br>Positive: %{x}<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        y=cat_df[cat_col],
        x=cat_df["neutral"],
        name="Neutral",
        orientation="h",
        marker=dict(color=THEME_TOKENS["neutral_color"]),
        hovertemplate="%{y}<br>Neutral: %{x}<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        y=cat_df[cat_col],
        x=cat_df["negative"],
        name="Negative",
        orientation="h",
        marker=dict(color=THEME_TOKENS["negative_color"]),
        hovertemplate="%{y}<br>Negative: %{x}<extra></extra>"
    ))

    fig.update_layout(
        **CHART_LAYOUT_DEFAULTS,
        barmode="stack",
        title=dict(text="<b>Sentiment Composition by Category</b>", font=dict(size=15)),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title="Review Count"),
        yaxis=dict(autorange="reversed", showgrid=False),
        height=360
    )
    return fig


def create_rating_vs_sentiment_chart(rating_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart showing sentiment distributions across 1-5 star ratings."""
    if rating_df is None or rating_df.empty:
        fig = go.Figure()
        fig.update_layout(**CHART_LAYOUT_DEFAULTS)
        return fig

    fig = go.Figure()
    rating_col = rating_df.columns[0]
    star_labels = [f"★ {int(r)}" if float(r).is_integer() else f"★ {r}" for r in rating_df[rating_col]]

    fig.add_trace(go.Bar(
        x=star_labels,
        y=rating_df["positive"],
        name="Positive",
        marker=dict(color=THEME_TOKENS["positive_color"])
    ))
    fig.add_trace(go.Bar(
        x=star_labels,
        y=rating_df["neutral"],
        name="Neutral",
        marker=dict(color=THEME_TOKENS["neutral_color"])
    ))
    fig.add_trace(go.Bar(
        x=star_labels,
        y=rating_df["negative"],
        name="Negative",
        marker=dict(color=THEME_TOKENS["negative_color"])
    ))

    fig.update_layout(
        **CHART_LAYOUT_DEFAULTS,
        barmode="group",
        title=dict(text="<b>User Star Ratings vs. NLP Classified Sentiment</b>", font=dict(size=15)),
        xaxis=dict(showgrid=False, title="User Star Rating"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title="Reviews"),
        height=340
    )
    return fig


def create_timeline_trend_chart(trend_df: pd.DataFrame) -> go.Figure:
    """Displays sentiment trend and review volume over time."""
    if trend_df is None or trend_df.empty:
        fig = go.Figure()
        fig.update_layout(**CHART_LAYOUT_DEFAULTS)
        return fig

    fig = go.Figure()

    # Review count bars
    fig.add_trace(go.Bar(
        x=trend_df["date_str"],
        y=trend_df["review_count"],
        name="Review Volume",
        marker=dict(color="rgba(99, 102, 241, 0.4)"),
        yaxis="y2"
    ))

    # Average polarity line
    fig.add_trace(go.Scatter(
        x=trend_df["date_str"],
        y=trend_df["avg_polarity"],
        name="Average Polarity",
        mode="lines+markers",
        line=dict(color=THEME_TOKENS["accent_secondary"], width=3),
        marker=dict(size=6, color=THEME_TOKENS["accent_secondary"]),
        yaxis="y"
    ))

    fig.update_layout(
        **CHART_LAYOUT_DEFAULTS,
        title=dict(text="<b>Sentiment & Volume Timeline</b>", font=dict(size=15)),
        xaxis=dict(showgrid=False, title="Date"),
        yaxis=dict(
            title="Avg Polarity",
            range=[-1.0, 1.0],
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            side="left"
        ),
        yaxis2=dict(
            title="Volume",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        height=340
    )
    return fig


def create_confusion_matrix_heatmap(cm_df: pd.DataFrame, labels: list) -> go.Figure:
    """Generates an academic confusion matrix heatmap."""
    if cm_df is None or cm_df.empty:
        fig = go.Figure()
        fig.update_layout(**CHART_LAYOUT_DEFAULTS)
        return fig

    z = cm_df.values
    x_labels = [f"Pred: {l}" for l in labels]
    y_labels = [f"True: {l}" for l in labels]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x_labels,
        y=y_labels,
        colorscale="Viridis",
        showscale=True,
        text=z,
        texttemplate="%{text}",
        textfont=dict(size=14, color="#FFFFFF")
    ))

    fig.update_layout(
        **CHART_LAYOUT_DEFAULTS,
        title=dict(text="<b>Academic Model Evaluation: Confusion Matrix</b>", font=dict(size=15)),
        xaxis=dict(title="Predicted Sentiment", showgrid=False),
        yaxis=dict(title="Ground Truth Sentiment", autorange="reversed", showgrid=False),
        height=350
    )
    return fig


def create_polarity_gauge(polarity: float) -> go.Figure:
    """Generates a high-precision polarity gauge indicator for Single Review analysis."""
    pol = round(float(polarity), 2)
    # Color selection based on polarity
    if pol > 0.05:
        bar_color = THEME_TOKENS["positive_color"]
    elif pol < -0.05:
        bar_color = THEME_TOKENS["negative_color"]
    else:
        bar_color = THEME_TOKENS["neutral_color"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pol,
        number=dict(valueformat="+.2f", font=dict(size=28, color="#FFFFFF")),
        gauge=dict(
            axis=dict(range=[-1.0, 1.0], tickwidth=1, tickcolor="#94A3B8"),
            bar=dict(color=bar_color, thickness=0.35),
            bgcolor="rgba(255,255,255,0.05)",
            borderwidth=1,
            bordercolor="rgba(255,255,255,0.1)",
            steps=[
                dict(range=[-1.0, -0.05], color="rgba(244, 63, 94, 0.15)"),
                dict(range=[-0.05, 0.05], color="rgba(148, 163, 184, 0.15)"),
                dict(range=[0.05, 1.0], color="rgba(16, 185, 129, 0.15)")
            ]
        )
    ))

    fig.update_layout(
        **CHART_LAYOUT_DEFAULTS,
        height=180,
        margin=dict(l=25, r=25, t=25, b=10)
    )
    return fig


def create_aspect_sentiment_chart(aspect_df: pd.DataFrame) -> go.Figure:
    """Creates a horizontal comparison bar chart of positive vs negative sentiment per aspect."""
    if aspect_df is None or aspect_df.empty:
        fig = go.Figure()
        fig.update_layout(**CHART_LAYOUT_DEFAULTS)
        return fig

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=aspect_df["aspect"],
        x=aspect_df["positive_count"],
        name="Positive Mentions",
        orientation="h",
        marker=dict(color=THEME_TOKENS["positive_color"]),
        hovertemplate="%{y}<br>Positive: %{x}<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        y=aspect_df["aspect"],
        x=aspect_df["negative_count"],
        name="Negative Mentions",
        orientation="h",
        marker=dict(color=THEME_TOKENS["negative_color"]),
        hovertemplate="%{y}<br>Negative: %{x}<extra></extra>"
    ))

    fig.update_layout(
        **CHART_LAYOUT_DEFAULTS,
        barmode="group",
        title=dict(text="<b>Aspect Dimension Mentions: Positive vs. Negative</b>", font=dict(size=15)),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title="Mention Count"),
        yaxis=dict(autorange="reversed", showgrid=False),
        height=360
    )
    return fig


def create_ngram_bar_chart(ngram_tuples: list, title: str, bar_color: str) -> go.Figure:
    """Displays top keyword / N-gram frequency bar chart."""
    if not ngram_tuples:
        fig = go.Figure()
        fig.update_layout(**CHART_LAYOUT_DEFAULTS)
        return fig

    labels = [t[0] for t in reversed(ngram_tuples)]
    counts = [t[1] for t in reversed(ngram_tuples)]

    fig = go.Figure(go.Bar(
        y=labels,
        x=counts,
        orientation="h",
        marker=dict(color=bar_color, line=dict(color="rgba(255,255,255,0.1)", width=1)),
        text=counts,
        textposition="auto"
    ))

    fig.update_layout(
        **CHART_LAYOUT_DEFAULTS,
        title=dict(text=f"<b>{title}</b>", font=dict(size=14)),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title="Occurrences"),
        yaxis=dict(showgrid=False),
        height=320
    )
    return fig


def create_rating_polarity_scatter(df: pd.DataFrame, rating_col: str = "rating", pol_col: str = "tb_polarity") -> go.Figure:
    """Scatter chart of Star Rating vs Polarity with anomaly boundaries."""
    if df is None or df.empty or rating_col not in df.columns or pol_col not in df.columns:
        fig = go.Figure()
        fig.update_layout(**CHART_LAYOUT_DEFAULTS)
        return fig

    work_df = df.copy()
    work_df["num_rating"] = pd.to_numeric(work_df[rating_col], errors="coerce")
    clean = work_df.dropna(subset=["num_rating", pol_col])

    # Add jitter to rating to avoid severe point overlap
    jitter = np.random.uniform(-0.15, 0.15, size=len(clean))
    jittered_rating = clean["num_rating"] + jitter

    # Detect anomalies for coloring
    colors = []
    for _, row in clean.iterrows():
        r = row["num_rating"]
        p = row[pol_col]
        if (r >= 4 and p < -0.05) or (r <= 2 and p > 0.05):
            colors.append("#F43F5E")  # Anomaly Red
        else:
            colors.append(THEME_TOKENS["accent_primary"])

    fig = go.Figure(go.Scatter(
        x=jittered_rating,
        y=clean[pol_col],
        mode="markers",
        marker=dict(size=8, color=colors, opacity=0.75, line=dict(color="rgba(255,255,255,0.2)", width=1)),
        hovertemplate="Rating: %{customdata[0]}<br>Polarity: %{y:+.2f}<extra></extra>",
        customdata=clean[["num_rating"]].values
    ))

    fig.update_layout(
        **CHART_LAYOUT_DEFAULTS,
        title=dict(text="<b>Star Rating vs. NLP Polarity (Red = Anomaly)</b>", font=dict(size=14)),
        xaxis=dict(title="User Star Rating (1 to 5)", showgrid=True, gridcolor="rgba(255,255,255,0.06)", range=[0.5, 5.5]),
        yaxis=dict(title="Polarity Score", range=[-1.05, 1.05], showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
        height=320
    )
    return fig


def create_aspect_radar_chart(aspect_df: pd.DataFrame) -> go.Figure:
    """Generates an 8-axis Polar Spider / Radar chart of Net Aspect Scores."""
    if aspect_df is None or aspect_df.empty:
        fig = go.Figure()
        fig.update_layout(**CHART_LAYOUT_DEFAULTS)
        return fig

    categories = list(aspect_df["aspect"])
    values = list(aspect_df["positive_rate"])
    
    # Close the radar polygon
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(99, 102, 241, 0.25)",
        line=dict(color=THEME_TOKENS["accent_primary"], width=2),
        name="Positive Rate %"
    ))

    fig.update_layout(
        **CHART_LAYOUT_DEFAULTS,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=True,
                ticksuffix="%",
                gridcolor="rgba(255,255,255,0.08)",
                linecolor="rgba(255,255,255,0.1)"
            ),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,0.08)",
                linecolor="rgba(255,255,255,0.1)"
            ),
            bgcolor="rgba(0,0,0,0)"
        ),
        title=dict(text="<b>Aspect Dimension Performance (Positive %)</b>", font=dict(size=15)),
        height=360,
        showlegend=False
    )
    return fig


def create_aspect_treemap(aspect_df: pd.DataFrame) -> go.Figure:
    """Generates an executive Treemap of product aspects sized by mentions and colored by Net Aspect Score."""
    if aspect_df is None or aspect_df.empty:
        fig = go.Figure()
        fig.update_layout(**CHART_LAYOUT_DEFAULTS)
        return fig

    fig = go.Figure(go.Treemap(
        labels=aspect_df["aspect"],
        parents=["Product Dimensions"] * len(aspect_df),
        values=aspect_df["total_mentions"],
        textinfo="label+value+percent parent",
        marker=dict(
            colors=aspect_df["net_aspect_score"],
            colorscale="RdYlGn",
            colorbar=dict(title="Net Aspect Score"),
            cmid=0.0,
            line=dict(width=2, color="#0F172A")
        ),
        hovertemplate="<b>%{label}</b><br>Mentions: %{value}<br>Net Score: %{color:+0.1f}<extra></extra>"
    ))

    fig.update_layout(
        **CHART_LAYOUT_DEFAULTS,
        title=dict(text="<b>Aspect Dimension Density & Net Sentiment Treemap</b>", font=dict(size=15)),
        height=360
    )
    return fig


def create_word_cloud_bubble_chart(keyword_tuples: list, title: str = "Lexical Drivers Bubble Chart") -> go.Figure:
    """Creates a visual circular bubble chart for keywords sized by frequency."""
    if not keyword_tuples:
        fig = go.Figure()
        fig.update_layout(**CHART_LAYOUT_DEFAULTS)
        return fig

    words = [t[0] for t in keyword_tuples]
    counts = [t[1] for t in keyword_tuples]
    
    # Generate pseudo-random coordinates for organic word cloud packing
    np.random.seed(42)
    x = np.random.uniform(10, 90, size=len(words))
    y = np.random.uniform(10, 90, size=len(words))

    # Scale bubble sizes
    max_count = max(counts)
    min_count = min(counts)
    bubble_sizes = [20 + (c - min_count) / max(1, (max_count - min_count)) * 40 for c in counts]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="text+markers",
        text=words,
        textposition="middle center",
        textfont=dict(size=12, color="#FFFFFF", family="JetBrains Mono"),
        marker=dict(
            size=bubble_sizes,
            color=counts,
            colorscale="Purples",
            showscale=False,
            opacity=0.85,
            line=dict(color="rgba(255,255,255,0.3)", width=1.5)
        ),
        hovertemplate="<b>%{text}</b><br>Frequency: %{customdata} occurrences<extra></extra>",
        customdata=counts
    ))

    fig.update_layout(
        **CHART_LAYOUT_DEFAULTS,
        title=dict(text=f"<b>{title}</b>", font=dict(size=14)),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        height=320
    )
    return fig




