from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional
import torch

from modellens.visualization.module_families import (
    infer_module_family,
    family_color_map,
)


def discover_circuit(
    lens,
    clean_input: Any,
    corrupted_input: Any,
    *,
    metric_fn: Optional[Callable] = None,
    importance_threshold: float = 0.15,
    layer_names: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Automatically discover the causal circuit for a specific behavior.

    Combines activation patching (which components matter) with attention
    analysis (how information flows between them) to produce a directed
    graph of the circuit.

    Args:
        lens: ModelLens instance
        clean_input: Input that produces the "correct" behavior
        corrupted_input: Modified input that produces different behavior
        metric_fn: Custom metric function for patching
        importance_threshold: Minimum |normalized_effect| to be included
        layer_names: Sublayers to patch. If None, auto-detects attn/mlp.

    Returns:
        Dict with nodes, edges, and metadata for circuit visualization.
    """
    from modellens.analysis.activation_patching import run_activation_patching

    # Step 1: Find causally important components via activation patching
    patch_results = run_activation_patching(
        lens,
        clean_input,
        corrupted_input,
        layer_names=layer_names,
        metric_fn=metric_fn,
        **kwargs,
    )

    # Step 2: Filter to important sublayers
    nodes = _build_nodes(patch_results, importance_threshold)

    if not nodes:
        return {
            "nodes": [],
            "edges": [],
            "patch_results": patch_results,
            "attention_results": None,
            "message": "No components exceeded the importance threshold.",
        }

    # Step 3: Get attention patterns to find information routing
    attn_results = _safe_attention_analysis(lens, clean_input, **kwargs)

    # Step 4: Build edges from attention patterns and sequential flow
    edges = _build_edges(nodes, attn_results)

    # Step 5: Assign roles based on component type and effect direction
    _assign_roles(nodes, patch_results)

    # Sort nodes by layer order
    nodes.sort(key=lambda n: n["order"])

    return {
        "nodes": nodes,
        "edges": edges,
        "patch_results": patch_results,
        "attention_results": attn_results,
        "clean_metric": patch_results["clean_metric"],
        "corrupted_metric": patch_results["corrupted_metric"],
        "total_effect": patch_results["total_effect"],
        "num_components": len(nodes),
        "num_connections": len(edges),
    }


def _build_nodes(patch_results: Dict, threshold: float) -> List[Dict]:
    """Extract important components as circuit nodes."""
    nodes = []
    layers_ordered = patch_results.get("layers_ordered", [])

    for i, name in enumerate(layers_ordered):
        data = patch_results["patch_effects"].get(name, {})
        norm_effect = abs(data.get("normalized_effect", 0.0))

        if norm_effect < threshold:
            continue

        family = infer_module_family(name)
        colors = family_color_map()
        block_num = _extract_block_number(name)

        nodes.append(
            {
                "name": name,
                "order": i,
                "block_num": block_num,
                "family": family,
                "color": colors.get(family, "#64748b"),
                "normalized_effect": data.get("normalized_effect", 0.0),
                "patched_metric": data.get("patched_metric", 0.0),
                "recovery_fraction": data.get("recovery_fraction_of_gap", 0.0),
                "prediction_restored": data.get("prediction_restored", False),
                "role": None,  # assigned later
            }
        )

    return nodes


def _safe_attention_analysis(lens, inputs, **kwargs) -> Optional[Dict]:
    """Run attention analysis, returning None if it fails."""
    try:
        from modellens.analysis.attention import run_attention_analysis

        return run_attention_analysis(lens, inputs, **kwargs)
    except Exception:
        return None


def _build_edges(nodes: List[Dict], attn_results: Optional[Dict]) -> List[Dict]:
    """
    Build directed edges between circuit nodes.

    Two types of edges:
    1. Sequential: MLP/attn in block N connects to components in block N+1
       (information flows through the residual stream)
    2. Attention-based: attention heads route information from specific
       token positions, connecting to downstream components
    """
    edges = []
    node_names = {n["name"] for n in nodes}
    node_by_name = {n["name"]: n for n in nodes}

    # Sequential edges: connect components across consecutive blocks
    sorted_nodes = sorted(nodes, key=lambda n: n["order"])
    for i in range(len(sorted_nodes)):
        for j in range(i + 1, len(sorted_nodes)):
            src = sorted_nodes[i]
            dst = sorted_nodes[j]

            # Connect if dst is in the next block or within 2 blocks
            if dst["block_num"] is not None and src["block_num"] is not None:
                gap = dst["block_num"] - src["block_num"]
                if 0 < gap <= 2:
                    edges.append(
                        {
                            "from": src["name"],
                            "to": dst["name"],
                            "type": "sequential",
                            "weight": min(
                                abs(src["normalized_effect"]),
                                abs(dst["normalized_effect"]),
                            ),
                        }
                    )
                    break  # only connect to nearest downstream node

    # Attention-based edges: which tokens does each attention head focus on
    if attn_results and "attention_maps" in attn_results:
        attn_nodes = [n for n in nodes if ".attn" in n["name"]]
        for node in attn_nodes:
            attn_key = node["name"]
            if attn_key not in attn_results["attention_maps"]:
                continue

            weights = attn_results["attention_maps"][attn_key]["weights"]

            # Find which token position gets most attention at the last position
            if weights.dim() == 4:
                # (batch, heads, seq, seq) — average across heads
                avg_attn = weights[0].mean(dim=0)  # (seq, seq)
            elif weights.dim() == 3:
                avg_attn = weights[0]  # (seq, seq)
            else:
                continue

            # Last token's attention distribution
            last_attn = avg_attn[-1]  # (seq,)
            max_attended_pos = int(last_attn.argmax().item())
            max_attn_weight = float(last_attn.max().item())

            # Find downstream components this attention feeds into
            for other in nodes:
                if other["name"] == node["name"]:
                    continue
                if other["block_num"] is not None and node["block_num"] is not None:
                    if (
                        other["block_num"] == node["block_num"]
                        and ".mlp" in other["name"]
                    ):
                        # Attention feeds into MLP within same block
                        edges.append(
                            {
                                "from": node["name"],
                                "to": other["name"],
                                "type": "attention_routing",
                                "weight": max_attn_weight,
                                "attended_position": max_attended_pos,
                            }
                        )

    # Deduplicate edges
    seen = set()
    unique_edges = []
    for e in edges:
        key = (e["from"], e["to"])
        if key not in seen:
            seen.add(key)
            unique_edges.append(e)

    return unique_edges


def _assign_roles(nodes: List[Dict], patch_results: Dict) -> None:
    """
    Assign functional roles to circuit nodes based on their properties.

    Roles:
    - "critical": corrupting this component nearly destroys the prediction
    - "booster": corrupted version actually helps the prediction
    - "processor": moderate effect, part of the processing pipeline
    - "gate": final layer component that shapes the output
    """
    clean_metric = patch_results["clean_metric"]

    for node in nodes:
        effect = node["normalized_effect"]
        family = node["family"]
        patched = node["patched_metric"]

        # Component whose corruption kills the prediction
        if effect > 0.7:
            node["role"] = "critical"
        # Corrupted version helps — this component suppresses in clean run
        elif effect < -0.3 and patched > clean_metric:
            node["role"] = "booster"
        # Last block components that shape final output
        elif node["block_num"] is not None and _is_late_layer(node, nodes):
            node["role"] = "gate"
        # Everything else is processing
        else:
            node["role"] = "processor"


def _is_late_layer(node: Dict, all_nodes: List[Dict]) -> bool:
    """Check if node is in the last 20% of blocks."""
    block_nums = [n["block_num"] for n in all_nodes if n["block_num"] is not None]
    if not block_nums:
        return False
    max_block = max(block_nums)
    return node["block_num"] >= max_block * 0.8


def _extract_block_number(name: str) -> Optional[int]:
    """Extract the block/layer number from a module name."""
    import re

    match = re.search(r"\.(\d+)\.", name)
    if match:
        return int(match.group(1))
    # Try end of string
    match = re.search(r"\.(\d+)$", name)
    if match:
        return int(match.group(1))
    return None


def summarize_circuit(circuit: Dict) -> str:
    """Generate a human-readable summary of the discovered circuit."""
    nodes = circuit.get("nodes", [])
    edges = circuit.get("edges", [])

    if not nodes:
        return "No significant circuit components found."

    lines = [
        f"Circuit: {len(nodes)} components, {len(edges)} connections",
        f"Clean metric: {circuit.get('clean_metric', 0):.4f}",
        f"Corrupted metric: {circuit.get('corrupted_metric', 0):.4f}",
        "",
    ]

    # Group by role
    for role in ["critical", "booster", "gate", "processor"]:
        role_nodes = [n for n in nodes if n["role"] == role]
        if role_nodes:
            lines.append(f"{role.upper()} components:")
            for n in role_nodes:
                effect_str = f"effect: {n['normalized_effect']:+.3f}"
                lines.append(f"  {n['name']} ({n['family']}) — {effect_str}")
            lines.append("")

    # Key connections
    attn_edges = [e for e in edges if e["type"] == "attention_routing"]
    if attn_edges:
        lines.append("Attention routing:")
        for e in attn_edges:
            lines.append(
                f"  {e['from']} → {e['to']} "
                f"(weight: {e['weight']:.3f}, attends to pos {e.get('attended_position', '?')})"
            )

    return "\n".join(lines)
