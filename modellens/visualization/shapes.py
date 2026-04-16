"""Module shape trace: tables and simple flow-style plots."""

from __future__ import annotations

from typing import Any, Dict, List

from modellens.visualization.common import default_plotly_layout, truncate_label

try:
    import plotly.graph_objects as go
except ImportError as e:  # pragma: no cover
    raise ImportError("plotly is required; pip install plotly") from e

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None


def compute_shape_trace(lens, inputs, **kwargs) -> List[Dict[str, Any]]:
    """
    Run a forward with hooks on all modules and collect output shapes.

    Returns ordered rows compatible with ``named_modules()`` order (excluding root).
    """
    lens.clear()
    lens.attach_all()
    lens.run(inputs, **kwargs)
    rows: List[Dict[str, Any]] = []
    for name, t in lens.get_activations().items():
        if t is None:
            continue
        if hasattr(t, "shape"):
            shape = tuple(t.shape)
            dtype = str(t.dtype).replace("torch.", "")
        else:
            shape = ()
            dtype = "unknown"
        rows.append(
            {
                "module": name,
                "shape": shape,
                "dtype": dtype,
            }
        )
    return rows


def shape_trace_to_dataframe(rows: List[Dict[str, Any]]):
    """pandas DataFrame for display/export."""
    if pd is None:
        raise ImportError("pandas is required; pip install pandas")
    return pd.DataFrame(rows)


def plot_shape_trace_table(
    rows: List[Dict[str, Any]],
    *,
    max_rows: int = 80,
    title: str = "Shape trace (hook outputs)",
    width: int = 920,
    height: int = 640,
) -> "go.Figure":
    """Scroll-friendly table visualization of module outputs."""
    rows = rows[:max_rows]
    modules = [truncate_label(r["module"], max_len=56) for r in rows]
    shapes = [str(r["shape"]) for r in rows]
    dtypes = [r["dtype"] for r in rows]

    # Alternating row colors for readability
    row_colors = [["#1e293b", "#0f172a"] * ((len(modules) + 1) // 2)]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["Module", "Shape", "dtype"],
                    fill_color="#0f172a",
                    font=dict(color="#e2e8f0", size=12),
                    line_color="#334155",
                    align="left",
                ),
                cells=dict(
                    values=[modules, shapes, dtypes],
                    fill_color=row_colors,
                    font=dict(color="#cbd5e1", size=11),
                    line_color="#334155",
                    align="left",
                ),
            )
        ]
    )
    fig.update_layout(
        **default_plotly_layout(title=title, width=width, height=height),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def shape_trace_mermaid(rows: List[Dict[str, Any]], max_nodes: int = 24) -> str:
    """
    Generate a Mermaid ``flowchart`` snippet (static) for README/slides.

    This is a *linear* chain of the first ``max_nodes`` modules — good for
    quick architecture sketches, not exact dataflow.
    """
    lines = ["flowchart LR"]
    subset = [r["module"] for r in rows[:max_nodes]]
    for i in range(len(subset) - 1):
        a = subset[i].replace('"', "'")[:40]
        b = subset[i + 1].replace('"', "'")[:40]
        lines.append(f'    n{i}["{a}"] --> n{i+1}["{b}"]')
    return "\n".join(lines)
