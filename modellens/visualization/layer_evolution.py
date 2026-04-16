"""Layer evolution visualizations — KDE overlays, 2D heatmaps, divergence plots."""

from typing import Any, Dict, List, Optional
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from modellens.visualization import default_plotly_layout, truncate_label


def _short_layer(name: str) -> str:
    """Shorten layer names: 'transformer.h.0' -> 'h.0'."""
    parts = name.split(".")
    # Find 'h' and keep from there
    for i, p in enumerate(parts):
        if p == "h":
            return ".".join(parts[i:])
    # Fallback: last 2 segments
    return ".".join(parts[-2:]) if len(parts) > 2 else name


def plot_evolution_heatmap(
    heatmap_data: Dict,
    *,
    title: Optional[str] = None,
    colorscale: Optional[str] = None,
    width: int = 900,
    height: int = 600,
) -> "go.Figure":
    """
    2D heatmap: layers (y) × logit bins (x), gaussian smoothed.

    Args:
        heatmap_data: Output from compute_evolution_heatmap.
        title: Plot title. Auto-generated if None.
        colorscale: Plotly colorscale. Auto-chosen by mode if None.
        width: Figure width.
        height: Figure height.
    """
    hm = heatmap_data["heatmap"]
    centers = heatmap_data["bin_centers"]
    layers = heatmap_data["layer_names"]
    mode = heatmap_data.get("mode", "clean")

    # Short labels for y-axis — e.g. "transformer.h.0" -> "h.0"
    short_labels = [_short_layer(n) for n in layers]

    if colorscale is None:
        colorscale = "RdBu_r" if mode == "diff" else "Viridis"

    # For diff mode, center the colorscale at zero
    zmin, zmax = None, None
    if mode == "diff":
        abs_max = max(abs(hm.min()), abs(hm.max()))
        zmin, zmax = -abs_max, abs_max

    fig = go.Figure(
        go.Heatmap(
            z=hm,
            x=centers,
            y=short_labels,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title="density" if mode != "diff" else "Δ density"),
            hovertemplate=(
                "logit: %{x:.1f}<br>" "layer: %{y}<br>" "value: %{z:.4f}<extra></extra>"
            ),
        )
    )

    default_title = {
        "clean": "Logit distribution across layers (clean)",
        "corrupted": "Logit distribution across layers (corrupted)",
        "diff": "Logit distribution shift (clean − corrupted)",
    }
    t = title or default_title.get(mode, "Logit evolution heatmap")

    fig.update_layout(
        **default_plotly_layout(title=t, width=width, height=height),
        xaxis_title="Logit value",
        yaxis_title="Layer",
    )
    return fig


def plot_evolution_heatmap_comparison(
    hm_clean: Dict,
    hm_corrupted: Dict,
    hm_diff: Dict,
    *,
    title: str = "Logit evolution: clean vs corrupted",
    width: int = 1400,
    height: int = 550,
) -> "go.Figure":
    """Side-by-side 2D heatmaps: clean, corrupted, diff."""
    layers = hm_clean["layer_names"]
    short_labels = [_short_layer(n) for n in layers]
    centers_c = hm_clean["bin_centers"]
    centers_x = hm_corrupted["bin_centers"]
    centers_d = hm_diff["bin_centers"]

    abs_max = max(abs(hm_diff["heatmap"].min()), abs(hm_diff["heatmap"].max()))

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["Clean", "Corrupted", "Diff (clean − corrupted)"],
        horizontal_spacing=0.06,
    )

    fig.add_trace(
        go.Heatmap(
            z=hm_clean["heatmap"],
            x=centers_c,
            y=short_labels,
            colorscale="Viridis",
            showscale=False,
            hovertemplate="logit: %{x:.1f}<br>layer: %{y}<br>val: %{z:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            z=hm_corrupted["heatmap"],
            x=centers_x,
            y=short_labels,
            colorscale="Viridis",
            showscale=False,
            hovertemplate="logit: %{x:.1f}<br>layer: %{y}<br>val: %{z:.4f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Heatmap(
            z=hm_diff["heatmap"],
            x=centers_d,
            y=short_labels,
            colorscale="RdBu_r",
            zmin=-abs_max,
            zmax=abs_max,
            colorbar=dict(title="Δ", x=1.02),
            hovertemplate="logit: %{x:.1f}<br>layer: %{y}<br>Δ: %{z:.4f}<extra></extra>",
        ),
        row=1,
        col=3,
    )

    fig.update_layout(**default_plotly_layout(title=title, width=width, height=height))
    return fig


def plot_kde_overlay(
    kde_data: Dict,
    layer_name: str,
    *,
    title: Optional[str] = None,
    width: int = 700,
    height: int = 400,
) -> "go.Figure":
    """KDE overlay for a single layer: clean vs corrupted."""
    x = kde_data["x"]
    clean_y = kde_data["clean_kdes"][layer_name]
    corrupted_y = kde_data["corrupted_kdes"][layer_name]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=clean_y,
            mode="lines",
            fill="tozeroy",
            name="Clean",
            line=dict(color="#4ade80", width=2),
            fillcolor="rgba(74, 222, 128, 0.25)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=corrupted_y,
            mode="lines",
            fill="tozeroy",
            name="Corrupted",
            line=dict(color="#f87171", width=2),
            fillcolor="rgba(248, 113, 113, 0.25)",
        )
    )

    short = truncate_label(layer_name, 40)
    t = title or f"Logit distribution — {short}"

    fig.update_layout(
        **default_plotly_layout(title=t, width=width, height=height),
        xaxis_title="Logit value",
        yaxis_title="Density",
        legend=dict(x=0.98, y=0.98, xanchor="right"),
    )
    return fig


def plot_kde_grid(
    kde_data: Dict,
    layer_names: List[str],
    *,
    cols: int = 3,
    title: str = "KDE: clean vs corrupted",
    width: int = 1200,
    height_per_row: int = 280,
) -> "go.Figure":
    """Grid of KDE overlays for multiple layers."""
    n = len(layer_names)
    rows = (n + cols - 1) // cols
    short_labels = [truncate_label(name, 28) for name in layer_names]

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=short_labels,
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    x = kde_data["x"]

    for i, name in enumerate(layer_names):
        r = i // cols + 1
        c = i % cols + 1
        show_legend = i == 0

        fig.add_trace(
            go.Scatter(
                x=x,
                y=kde_data["clean_kdes"][name],
                mode="lines",
                fill="tozeroy",
                name="Clean",
                legendgroup="clean",
                showlegend=show_legend,
                line=dict(color="#4ade80", width=1.5),
                fillcolor="rgba(74, 222, 128, 0.2)",
            ),
            row=r,
            col=c,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=kde_data["corrupted_kdes"][name],
                mode="lines",
                fill="tozeroy",
                name="Corrupted",
                legendgroup="corrupted",
                showlegend=show_legend,
                line=dict(color="#f87171", width=1.5),
                fillcolor="rgba(248, 113, 113, 0.2)",
            ),
            row=r,
            col=c,
        )

    fig.update_layout(
        **default_plotly_layout(title=title, width=width, height=rows * height_per_row),
        legend=dict(x=0.98, y=1.02, xanchor="right"),
    )
    return fig


def plot_divergence_by_layer(
    comparison: Dict,
    *,
    metric: str = "js",
    title: Optional[str] = None,
    width: int = 1000,
    height: int = 420,
) -> "go.Figure":
    """Bar chart of per-layer divergence between clean and corrupted."""
    layers = comparison["common_layers"]
    divs = comparison["divergences"]
    values = [divs[n][metric] for n in layers]
    short_labels = [_short_layer(n) for n in layers]

    metric_names = {"kl": "KL", "js": "Jensen-Shannon", "l2": "L2"}
    metric_label = metric_names.get(metric, metric.upper())

    colors = _divergence_colors(values)

    fig = go.Figure(
        go.Bar(
            x=short_labels,
            y=values,
            marker_color=colors,
            hovertemplate="layer: %{x}<br>"
            + metric_label
            + ": %{y:.4f}<extra></extra>",
        )
    )

    t = title or f"Per-layer divergence ({metric_label}): clean vs corrupted"
    fig.update_layout(
        **default_plotly_layout(title=t, width=width, height=height),
        xaxis_title="Layer",
        yaxis_title=metric_label + " divergence",
        xaxis_tickangle=-45,
    )
    return fig


def _divergence_colors(values: List[float]) -> List[str]:
    """Color bars by relative magnitude — low=teal, high=red."""
    if not values:
        return []
    max_val = max(values) if max(values) > 0 else 1.0
    colors = []
    for v in values:
        ratio = v / max_val
        if ratio < 0.33:
            colors.append("#2dd4bf")  # teal
        elif ratio < 0.66:
            colors.append("#fbbf24")  # amber
        else:
            colors.append("#f87171")  # red
    return colors


def plot_trajectory_comparison(
    comparison: Dict,
    *,
    metric: str = "entropy",
    title: Optional[str] = None,
    width: int = 1000,
    height: int = 400,
) -> "go.Figure":
    """
    Line plot comparing a trajectory metric between clean and corrupted.

    metric: "entropy", "confidence", "margin"
    """
    metric_key = {
        "entropy": "entropy_trajectory",
        "confidence": "confidence_trajectory",
        "margin": "margin_trajectory",
    }[metric]

    clean_vals = comparison["clean"][metric_key]
    corrupted_vals = comparison["corrupted"][metric_key]
    clean_layers = comparison["clean"]["layers_ordered"]
    corrupted_layers = comparison["corrupted"]["layers_ordered"]

    short_clean = [truncate_label(n, 20) for n in clean_layers]
    short_corrupted = [truncate_label(n, 20) for n in corrupted_layers]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=short_clean,
            y=clean_vals,
            mode="lines+markers",
            name="Clean",
            line=dict(color="#4ade80", width=2),
            marker=dict(size=4),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=short_corrupted,
            y=corrupted_vals,
            mode="lines+markers",
            name="Corrupted",
            line=dict(color="#f87171", width=2),
            marker=dict(size=4),
        )
    )

    metric_labels = {
        "entropy": "Entropy",
        "confidence": "Top-1 probability",
        "margin": "Top-1 − Top-2 margin",
    }
    ylabel = metric_labels.get(metric, metric)
    t = title or f"{ylabel} across layers: clean vs corrupted"

    fig.update_layout(
        **default_plotly_layout(title=t, width=width, height=height),
        xaxis_title="Layer",
        yaxis_title=ylabel,
        xaxis_tickangle=-45,
        legend=dict(x=0.98, y=0.98, xanchor="right"),
    )
    return fig


def plot_token_trajectories(
    results: Dict,
    *,
    top_n: int = 8,
    title: str = "Token probability trajectories",
    width: int = 1000,
    height: int = 450,
) -> "go.Figure":
    """Line plot showing how individual tokens rise and fall across layers."""
    trajectories = results["token_trajectories"]
    layers = results["layers_ordered"]
    short_labels = [truncate_label(n, 20) for n in layers]

    # Pick top_n tokens by peak probability
    sorted_toks = sorted(
        trajectories.values(),
        key=lambda t: t["max_prob"],
        reverse=True,
    )[:top_n]

    # Color palette
    palette = [
        "#4ade80",
        "#f87171",
        "#60a5fa",
        "#fbbf24",
        "#c084fc",
        "#fb923c",
        "#2dd4bf",
        "#f472b6",
    ]

    fig = go.Figure()
    for i, tok in enumerate(sorted_toks):
        color = palette[i % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=short_labels,
                y=tok["probs_per_layer"],
                mode="lines+markers",
                name=tok["token_str"],
                line=dict(color=color, width=2),
                marker=dict(size=4),
            )
        )

    fig.update_layout(
        **default_plotly_layout(title=title, width=width, height=height),
        xaxis_title="Layer",
        yaxis_title="Probability",
        xaxis_tickangle=-45,
        legend=dict(x=1.02, y=1, xanchor="left"),
    )
    return fig


def format_evolution_summary_html(comparison: Dict) -> str:
    """HTML summary card for Streamlit."""
    clean = comparison["clean"]
    corrupted = comparison["corrupted"]
    divs = comparison["divergences"]
    common = comparison["common_layers"]

    if not common:
        return "<p>No common layers.</p>"

    # Find most / least divergent
    max_layer = max(common, key=lambda n: divs[n]["js"])
    min_layer = min(common, key=lambda n: divs[n]["js"])
    avg_js = np.mean([divs[n]["js"] for n in common])

    clean_moments = clean.get("key_moments", {})
    corrupted_moments = corrupted.get("key_moments", {})

    rows = [
        ("Layers analyzed", str(len(common))),
        ("Avg JS divergence", f"{avg_js:.4f}"),
        (
            "Most divergent",
            f"{truncate_label(max_layer, 30)} ({divs[max_layer]['js']:.4f})",
        ),
        (
            "Least divergent",
            f"{truncate_label(min_layer, 30)} ({divs[min_layer]['js']:.4f})",
        ),
    ]

    if "first_confidence" in clean_moments:
        rows.append(
            (
                "Clean first confidence",
                truncate_label(clean_moments["first_confidence"]["layer"], 30),
            )
        )
    if "first_confidence" in corrupted_moments:
        rows.append(
            (
                "Corrupted first confidence",
                truncate_label(corrupted_moments["first_confidence"]["layer"], 30),
            )
        )

    html = (
        '<div style="background:#1e293b;border-radius:8px;padding:16px;'
        'font-family:sans-serif;font-size:14px;color:#e2e8f0;">'
        '<table style="width:100%;border-collapse:collapse;">'
    )
    for label, value in rows:
        html += (
            f"<tr>"
            f'<td style="padding:6px 12px;color:#94a3b8;">{label}</td>'
            f'<td style="padding:6px 12px;font-weight:600;">{value}</td>'
            f"</tr>"
        )
    html += "</table></div>"
    return html
