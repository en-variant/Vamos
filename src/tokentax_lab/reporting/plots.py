import plotly.graph_objects as go
import tiktoken


def tokenizer_usage_figure(comparison_rows: list[dict]):
    available = [row for row in comparison_rows if row.get("status") == "available"]
    fig = go.Figure()
    if available:
        fig.add_trace(go.Bar(
            name="Regional text",
            x=[row["tokenizer"] for row in available],
            y=[row["regional_tokens"] for row in available],
            marker_color="#8bc34a",
        ))
        fig.add_trace(go.Bar(
            name="English translation",
            x=[row["tokenizer"] for row in available],
            y=[row["English translation_tokens"] for row in available],
            marker_color="#d4a84f",
        ))
    fig.update_layout(
        title="Token usage by tokenizer",
        xaxis_title="Tokenizer",
        yaxis_title="Tokens",
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e7eadf"},
        legend={"title": "Text version"},
    )
    return fig


def tokenizer_cost_figure(
    comparison_rows: list[dict],
    usd_per_million_input_tokens: float,
    usd_to_inr: float,
    price_model: str,
):
    available = [row for row in comparison_rows if row.get("status") == "available"]
    fig = go.Figure()
    if available:
        for label, count_key, color in (
            ("Regional text", "regional_tokens", "#8bc34a"),
            ("English translation", "English translation_tokens", "#d4a84f"),
        ):
            costs = [
                row[count_key]
                * usd_per_million_input_tokens
                * usd_to_inr
                / 1_000_000
                for row in available
            ]
            fig.add_trace(go.Bar(
                name=label,
                x=[row["tokenizer"] for row in available],
                y=costs,
                marker_color=color,
                customdata=costs,
                hovertemplate=(
                    "%{x}<br>" + label + ": ₹%{y:.8f}<extra></extra>"
                ),
            ))
    fig.update_layout(
        title=f"Estimated input cost by tokenizer ({price_model})",
        xaxis_title="Tokenizer",
        yaxis_title="Counterfactual INR per prompt",
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e7eadf"},
        legend={"title": "Text version"},
        yaxis={"tickprefix": "₹", "tickformat": ".4f"},
    )
    return fig


def pareto_figure(points: list[dict]):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[p["tokens"] for p in points], y=[p["accuracy"] for p in points],
        mode="markers+text", text=[p["name"] for p in points], textposition="top center",
        marker={"size": 12, "color": [p.get("latency_ms", 0) for p in points], "colorscale": [[0, "#8bc34a"], [1, "#d4a84f"]], "showscale": True},
    ))
    fig.update_layout(
        title="Cost–quality frontier",
        xaxis_title="Tokens per sentence",
        yaxis_title="Accuracy",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e7eadf"},
    )
    return fig


def heatmap_figure(text: str, tokenizer_name: str = "cl100k_base"):
    enc = tiktoken.get_encoding(tokenizer_name)
    ids = enc.encode(text)
    pieces = [enc.decode([i]) for i in ids]
    fig = go.Figure(go.Bar(
        x=list(range(len(pieces))),
        y=[1] * len(pieces),
        text=pieces,
        marker={
            "color": list(range(len(pieces))),
            "colorscale": [[0, "#375b32"], [0.55, "#8bc34a"], [1, "#d4a84f"]],
        },
    ))
    fig.update_layout(
        title=f"Token heatmap: {tokenizer_name}",
        xaxis_title="Piece index",
        yaxis={"visible": False},
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e7eadf"},
    )
    return fig
