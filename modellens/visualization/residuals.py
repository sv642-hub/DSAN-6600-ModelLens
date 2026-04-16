"""Residual stream contribution plots."""

from __future__ import annotations
from typing import Any, Dict, Optional
from modellens.visualization.common import default_plotly_layout, truncate_label
from modellens.visualization.schemas import residual_dict_to_viz

try:
    import plotly.graph_objects as go
except ImportError as e:  # pragma: no cover
    raise ImportError("plotly is required; pip install plotly") from e


def plot_residual_contributions(
    residual_result: Dict[str, Any],
    *,
    mode: str = "relative",
    title: Optional[str] = None,
    width: int = 900,
    height: int = 520,
) -> "go.Figure":
    """
    Bar or line chart of per-layer residual metrics.

    ``mode``: ``relative`` | ``delta`` | ``cosine``
    """
    v = residual_dict_to_viz(residual_result)
    labels = [truncate_label(x, max_len=40) for x in v.layers]

    if mode == "relative":
        y = v.relative_contribution
        ylab = "Relative contribution (‖Δ‖ / ‖stream‖)"
    elif mode == "delta":
        y = v.delta_norm
        ylab = "‖Δ‖ (layer update magnitude)"
    elif mode == "cosine":
        y = v.cosine_similarity
        ylab = "Cosine sim (prev → current)"
    else:
        raise ValueError("mode must be relative, delta, or cosine")

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=y,
            marker_color="#7c3aed",
            hovertemplate="layer=%{x}<br>value=%{y:.4f}<extra></extra>",
        )
    )
    fig.update_yaxes(title_text=ylab)
    fig.update_xaxes(title_text="Layer (after block)")
    t = title or f"Residual stream — {mode}"
    fig.update_layout(**default_plotly_layout(title=t, width=width, height=height))
    fig.update_xaxes(tickangle=45)
    return fig


def plot_residual_lines(
    residual_result: Dict[str, Any],
    *,
    title: Optional[str] = None,
    width: int = 900,
    height: int = 520,
) -> "go.Figure":
    """Multi-series line chart: delta norm + relative contribution."""
    v = residual_dict_to_viz(residual_result)
    x = [truncate_label(l, max_len=32) for l in v.layers]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=v.delta_norm,
            name="‖Δ‖",
            mode="lines+markers",
            line=dict(color="#2563eb"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=v.relative_contribution,
            name="Relative",
            mode="lines+markers",
            line=dict(color="#db2777"),
            yaxis="y2",
        )
    )
    fig.update_layout(
        **default_plotly_layout(
            title=title or "Residual stream metrics", width=width, height=height
        ),
        yaxis2=dict(
            title="Relative contribution",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
    )
    fig.update_yaxes(title_text="‖Δ‖", side="left")
    fig.update_xaxes(tickangle=45)
    return fig
