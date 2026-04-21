"""Layered flow-style visualization for heuristic circuit discovery results."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from modellens.visualization.common import default_plotly_layout, truncate_label
from modellens.visualization.module_families import pretty_module_name

try:
    import plotly.graph_objects as go
except ImportError as e:  # pragma: no cover
    raise ImportError("plotly is required; pip install plotly") from e


ROLE_COLORS = {
    "critical": "#ef4444",
    "booster": "#3b82f6",
    "gate": "#f59e0b",
    "processor": "#94a3b8",
}

EDGE_COLOR_SEQ = "#64748b"
EDGE_COLOR_ATTN = "#0ea5e9"


def _node_sort_key(n: Dict[str, Any]) -> Tuple[int, int, str]:
    bn = n.get("block_num")
    if bn is None:
        bn = -1
    return (bn, int(n.get("order", 0)), str(n.get("name", "")))


def _select_nodes_for_flow(
    nodes: List[Dict[str, Any]], max_nodes: int
) -> Tuple[List[Dict[str, Any]], bool]:
    """Return up to ``max_nodes`` nodes by strongest |effect|, preserving discovery intent."""
    if len(nodes) <= max_nodes:
        return nodes, False
    ranked = sorted(
        nodes,
        key=lambda n: abs(float(n.get("normalized_effect", 0.0))),
        reverse=True,
    )
    return ranked[:max_nodes], True


def _layout_positions(
    nodes: List[Dict[str, Any]],
) -> Dict[str, Tuple[float, float]]:
    """Place nodes left→right by block depth; stagger vertically within a block."""
    by_block: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for n in sorted(nodes, key=_node_sort_key):
        bn = n.get("block_num")
        if bn is None:
            bn = -1
        by_block[int(bn)].append(n)

    pos: Dict[str, Tuple[float, float]] = {}
    x_scale = 140.0
    for bn in sorted(by_block.keys()):
        group = by_block[bn]
        x = float(bn + 1) * x_scale if bn >= 0 else 40.0
        n_g = len(group)
        for i, node in enumerate(group):
            # Vertical stack centered
            y = (i - (n_g - 1) / 2.0) * 72.0
            # Slight family-based nudge to separate attn vs mlp in same block
            fam = str(node.get("family", "")).lower()
            if "attention" in fam:
                y -= 18.0
            elif "mlp" in fam:
                y += 18.0
            pos[str(node["name"])] = (x, y)
    return pos


def _suggested_route_text(
    nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]
) -> Optional[str]:
    """One-line path hint from sequential edges ordered by source block."""
    seq = [e for e in edges if e.get("type") == "sequential"]
    if not seq:
        return None
    # Order edges by source block
    def src_block(e: Dict[str, Any]) -> int:
        for n in nodes:
            if n["name"] == e["from"]:
                return int(n.get("block_num") or -1)
        return -1

    ordered = sorted(seq, key=src_block)
    parts = []
    seen = set()
    for e in ordered:
        label = truncate_label(pretty_module_name(str(e["from"])), 18)
        if label not in seen:
            parts.append(label)
            seen.add(label)
    if ordered:
        last = ordered[-1]["to"]
        label = truncate_label(pretty_module_name(str(last)), 18)
        if label not in seen:
            parts.append(label)
    if len(parts) < 2:
        return None
    return " → ".join(parts)


def plot_circuit_story_flow(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    *,
    max_nodes: int = 12,
    title: str = "Candidate circuit pathway",
    width: int = 980,
    height: int = 460,
) -> "go.Figure":
    """
    Layered left-to-right flow diagram: nodes by block depth, edges as curved hints.

    This is a presentation-oriented *sketch* — not a formal DAG layout engine.
    """
    if not nodes:
        fig = go.Figure()
        fig.update_layout(
            **default_plotly_layout(
                title=title,
                width=width,
                height=280,
            ),
            annotations=[
                dict(
                    text="No components passed the threshold — try a lower importance threshold.",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=14, color="#94a3b8"),
                )
            ],
        )
        return fig

    flow_nodes, truncated = _select_nodes_for_flow(nodes, max_nodes)
    name_set = {str(n["name"]) for n in flow_nodes}
    flow_edges = [
        e
        for e in edges
        if str(e.get("from")) in name_set and str(e.get("to")) in name_set
    ]

    pos = _layout_positions(flow_nodes)
    max_per_block = 1
    if flow_nodes:
        bc: Dict[int, int] = defaultdict(int)
        for n in flow_nodes:
            bn = n.get("block_num")
            if bn is None:
                bn = -1
            bc[int(bn)] += 1
        max_per_block = max(bc.values()) if bc else 1
    dyn_height = int(min(height, max(420, 120 + max_per_block * 95)))

    fig = go.Figure()

    # Edge traces (under nodes)
    max_w = max((float(e.get("weight", 0.1)) for e in flow_edges), default=0.1)

    def edge_width(w: float) -> float:
        w = float(w)
        return 1.5 + 4.0 * min(1.0, w / max(max_w, 1e-6))

    for e in flow_edges:
        src, dst = str(e["from"]), str(e["to"])
        if src not in pos or dst not in pos:
            continue
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        ec = EDGE_COLOR_ATTN if e.get("type") == "attention_routing" else EDGE_COLOR_SEQ
        dash = "solid" if e.get("type") == "attention_routing" else "dot"
        w = edge_width(float(e.get("weight", 0.2)))
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color=ec, width=w, dash=dash),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    xs, ys, texts, sizes, colors, hovers, roles = [], [], [], [], [], [], []
    for n in flow_nodes:
        name = str(n["name"])
        if name not in pos:
            continue
        x, y = pos[name]
        xs.append(x)
        ys.append(y)
        eff = float(n.get("normalized_effect", 0.0))
        label = truncate_label(pretty_module_name(name), 26)
        texts.append(label)
        sizes.append(18 + min(34, abs(eff) * 28.0))
        role = str(n.get("role", "processor"))
        roles.append(role)
        colors.append(ROLE_COLORS.get(role, ROLE_COLORS["processor"]))
        hovers.append(
            f"<b>{pretty_module_name(name)}</b><br>"
            f"<span style='font-family:monospace;font-size:11px'>{name}</span><br>"
            f"role: {role} · family: {n.get('family', '?')}<br>"
            f"normalized effect: {eff:+.3f}<br>"
            f"recovery fraction: {float(n.get('recovery_fraction', 0.0)):.3f}"
        )

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            marker=dict(
                size=sizes,
                color=colors,
                line=dict(width=2, color="rgba(15,23,42,0.85)"),
                opacity=0.92,
            ),
            text=texts,
            textposition="top center",
            textfont=dict(size=11, color="#e2e8f0"),
            hoverinfo="text",
            hovertext=hovers,
            name="components",
            showlegend=False,
        )
    )

    # Legend proxies (discrete)
    for role, col in ROLE_COLORS.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=12, color=col, symbol="circle"),
                name=role.capitalize(),
                showlegend=True,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color=EDGE_COLOR_SEQ, width=3, dash="dot"),
            name="Sequential link (hypothesis)",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color=EDGE_COLOR_ATTN, width=3),
            name="Attention routing (hypothesis)",
            showlegend=True,
        )
    )

    route = _suggested_route_text(flow_nodes, flow_edges)
    ann = []
    if route:
        ann.append(
            dict(
                text=f"<b>Suggested route</b> (sequential edges):<br>{route}",
                xref="paper",
                yref="paper",
                x=0.01,
                y=-0.14,
                xanchor="left",
                yanchor="top",
                showarrow=False,
                align="left",
                font=dict(size=11, color="#cbd5e1"),
            )
        )
    if truncated:
        ann.append(
            dict(
                text=f"Showing top {max_nodes} components by |effect|; see bars below for the full ranked list.",
                xref="paper",
                yref="paper",
                x=0.99,
                y=1.02,
                xanchor="right",
                yanchor="bottom",
                showarrow=False,
                font=dict(size=10, color="#94a3b8"),
            )
        )

    _base = default_plotly_layout(title=title, width=width, height=dyn_height)
    fig.update_layout(
        title=_base.get("title"),
        font=_base.get("font"),
        width=_base.get("width"),
        height=dyn_height,
        template="plotly_dark",
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0f172a",
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(148,163,184,0.12)",
            zeroline=False,
            showticklabels=False,
            title=dict(text="Earlier blocks → later blocks (approximate depth)", font=dict(size=11)),
        ),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            font=dict(size=10, color="#cbd5e1"),
            bgcolor="rgba(15,23,42,0.6)",
        ),
        margin=dict(l=50, r=40, t=72, b=100 if route else 72),
        annotations=ann,
    )
    return fig
