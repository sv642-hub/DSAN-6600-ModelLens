"""Embedding similarity and neighbor views."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
from modellens.visualization.common import default_plotly_layout, truncate_labels

try:
    import plotly.graph_objects as go
except ImportError as e:  # pragma: no cover
    raise ImportError("plotly is required; pip install plotly") from e


def plot_embedding_similarity_heatmap(
    embedding_result: Dict[str, Any],
    *,
    title: Optional[str] = None,
    width: int = 700,
    height: int = 600,
) -> "go.Figure":
    """Cosine similarity matrix between token positions."""
    sim = embedding_result["similarity_matrix"]
    if hasattr(sim, "detach"):
        sim = sim.detach().cpu().numpy()
    z = np.asarray(sim, dtype=np.float64)

    labels = embedding_result.get("token_labels") or []
    if len(labels) != z.shape[0]:
        labels = [str(i) for i in range(z.shape[0])]
    labels = truncate_labels(labels, max_len=14)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            colorscale="RdBu",
            texttemplate="%{z:.2f}",
            textfont={"size": 11},
            zmid=0.0,
            hovertemplate="i=%{y}<br>j=%{x}<br>cos=%{z:.3f}<extra></extra>",
        )
    )
    t = title or "Input embedding cosine similarity"
    fig.update_layout(**default_plotly_layout(title=t, width=width, height=height))
    fig.update_xaxes(tickangle=45)
    return fig


def plot_embedding_norms(
    embedding_result: Dict[str, Any],
    *,
    title: Optional[str] = None,
    width: int = 800,
    height: int = 420,
) -> "go.Figure":
    """Bar chart of L2 norms per token position."""
    norms = embedding_result["norms"]
    if hasattr(norms, "detach"):
        norms = norms[0].detach().cpu().numpy()
    else:
        norms = np.asarray(norms[0], dtype=np.float64)

    labels = embedding_result.get("token_labels") or []
    if len(labels) != len(norms):
        labels = [str(i) for i in range(len(norms))]
    labels = truncate_labels(labels, max_len=16)

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=norms,
            marker_color="#0891b2",
            hovertemplate="token=%{x}<br>‖e‖=%{y:.3f}<extra></extra>",
        )
    )
    t = title or "Embedding L2 norms by position"
    fig.update_layout(**default_plotly_layout(title=t, width=width, height=height))
    fig.update_xaxes(tickangle=45)
    return fig
