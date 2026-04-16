from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter


def run_layer_evolution(
    lens,
    inputs: Any,
    *,
    top_k: int = 10,
    position: int = -1,
    tokenizer=None,
    layer_names: Optional[List[str]] = None,
    capture_full_logits: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Track the full prediction distribution evolution across layers.

    For each layer, captures:
    - Top-k token predictions with probabilities
    - Entropy (how confident the model is)
    - KL divergence from the previous layer (how much the distribution changed)
    - Rank of specific tokens of interest across layers
    - Optionally, the full logit vector for KDE / heatmap analysis

    Args:
        lens: ModelLens instance
        inputs: Model input
        top_k: Number of top predictions to track per layer
        position: Token position to analyze (default: last)
        tokenizer: For decoding token IDs to strings
        layer_names: Specific layers to analyze. If None, uses all hooked layers.
        capture_full_logits: If True, store the entire logit vector per layer
                            (needed for KDE and heatmap visualizations).

    Returns:
        Dict with per-layer evolution data and trajectory metrics.
    """
    unembed = lens.adapter.get_unembedding(lens.model)
    if unembed is None:
        raise ValueError("Could not find unembedding matrix.")

    lens.clear()
    if layer_names:
        lens.attach_layers(layer_names)
    elif len(lens.hooks) == 0:
        lens.attach_all()

    output = lens.run(inputs, **kwargs)
    activations = lens.get_activations()

    if layer_names:
        activations = {k: v for k, v in activations.items() if k in layer_names}

    hidden_dim = unembed.shape[-1]

    layers = []
    prev_probs = None

    for name, activation in activations.items():
        if activation.shape[-1] != hidden_dim:
            continue

        # Skip activations without a batch dimension (e.g. pos_embed)
        if activation.dim() < 3:
            continue

        logits = activation @ unembed.T
        probs = F.softmax(logits, dim=-1)

        seq_len = probs.shape[1]
        pos = position if position >= 0 else seq_len + position
        pos = max(0, min(pos, seq_len - 1))

        p = probs[0, pos]  # (vocab_size,)
        raw_logits = logits[0, pos]  # (vocab_size,)

        # Guard against empty or scalar tensors
        if p.dim() == 0 or p.shape[0] == 0:
            continue

        actual_k = min(top_k, p.shape[0])
        top_probs, top_indices = torch.topk(p, k=actual_k)

        entropy = -(p * torch.log(p + 1e-12)).sum().item()

        kl_from_prev = None
        if prev_probs is not None:
            kl_from_prev = F.kl_div(
                torch.log(p + 1e-12),
                prev_probs,
                reduction="sum",
            ).item()

        sorted_p = torch.sort(p, descending=True).values
        margin = (sorted_p[0] - sorted_p[1]).item() if sorted_p.shape[0] > 1 else 0.0

        layer_data = {
            "layer_name": name,
            "top_k_indices": top_indices.detach(),
            "top_k_probs": top_probs.detach(),
            "entropy": entropy,
            "kl_from_prev": kl_from_prev,
            "top1_prob": sorted_p[0].item(),
            "margin_top1_top2": margin,
            "position_used": pos,
        }

        if capture_full_logits:
            layer_data["full_logits"] = raw_logits.detach().cpu().numpy()

        if tokenizer is not None:
            tokens = []
            for idx in top_indices:
                try:
                    tokens.append(tokenizer.decode([idx.item()]))
                except Exception:
                    tokens.append(str(idx.item()))
            layer_data["top_k_tokens"] = tokens

        layers.append(layer_data)
        prev_probs = p.detach()

    layers_ordered = [l["layer_name"] for l in layers]
    entropy_trajectory = [l["entropy"] for l in layers]
    confidence_trajectory = [l["top1_prob"] for l in layers]
    kl_trajectory = [l["kl_from_prev"] for l in layers]
    margin_trajectory = [l["margin_top1_top2"] for l in layers]

    token_trajectories = _build_token_trajectories(layers, top_k)
    moments = _find_key_moments(layers, entropy_trajectory, kl_trajectory)

    return {
        "layers": layers,
        "layers_ordered": layers_ordered,
        "entropy_trajectory": entropy_trajectory,
        "confidence_trajectory": confidence_trajectory,
        "kl_trajectory": kl_trajectory,
        "margin_trajectory": margin_trajectory,
        "token_trajectories": token_trajectories,
        "key_moments": moments,
        "num_layers": len(layers),
        "position_used": layers[0]["position_used"] if layers else None,
    }


def run_layer_evolution_comparison(
    lens,
    clean_inputs: Any,
    corrupted_inputs: Any,
    *,
    top_k: int = 10,
    position: int = -1,
    tokenizer=None,
    layer_names: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run layer evolution on both clean and corrupted inputs, then compute
    per-layer divergence metrics between them.

    Full logits are always captured for KDE and heatmap visualizations.

    Args:
        lens: ModelLens instance.
        clean_inputs: Tokenized clean input.
        corrupted_inputs: Tokenized corrupted input.
        top_k: Top-k predictions per layer.
        position: Token position to analyze (-1 = last).
        tokenizer: For decoding.
        layer_names: Layers to analyze. None = all.

    Returns:
        Dict with clean results, corrupted results, per-layer divergences,
        and precomputed KDE / heatmap data.
    """
    clean = run_layer_evolution(
        lens,
        clean_inputs,
        top_k=top_k,
        position=position,
        tokenizer=tokenizer,
        layer_names=layer_names,
        capture_full_logits=True,
        **kwargs,
    )

    corrupted = run_layer_evolution(
        lens,
        corrupted_inputs,
        top_k=top_k,
        position=position,
        tokenizer=tokenizer,
        layer_names=layer_names,
        capture_full_logits=True,
        **kwargs,
    )

    lens.clear()

    # Match layers present in both runs
    clean_map = {l["layer_name"]: l for l in clean["layers"]}
    corr_map = {l["layer_name"]: l for l in corrupted["layers"]}
    common = [n for n in clean["layers_ordered"] if n in corr_map]

    # Per-layer divergences between clean and corrupted
    divergences = {}
    for name in common:
        c_logits = clean_map[name]["full_logits"]
        x_logits = corr_map[name]["full_logits"]
        divergences[name] = _compute_divergences(c_logits, x_logits)

    return {
        "clean": clean,
        "corrupted": corrupted,
        "common_layers": common,
        "divergences": divergences,
    }


def _compute_divergences(
    logits_a: np.ndarray, logits_b: np.ndarray
) -> Dict[str, float]:
    """Compute KL, Jensen-Shannon, and L2 between two logit vectors."""
    p = _softmax_np(logits_a)
    q = _softmax_np(logits_b)

    eps = 1e-10
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)

    kl_pq = float(np.sum(p * np.log(p / q)))
    kl_qp = float(np.sum(q * np.log(q / p)))
    js = 0.5 * kl_pq + 0.5 * kl_qp
    l2 = float(np.linalg.norm(p - q))

    return {"kl": kl_pq, "kl_reverse": kl_qp, "js": js, "l2": l2}


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max()
    e = np.exp(x)
    return e / e.sum()


def compute_layer_kdes(
    comparison: Dict,
    n_points: int = 512,
    logit_range: Optional[Tuple[float, float]] = None,
) -> Dict:
    """
    Compute 1D gaussian KDE for clean and corrupted logits at each layer.

    Args:
        comparison: Output from run_layer_evolution_comparison.
        n_points: Number of evaluation points for the KDE curves.
        logit_range: (min, max) for the x-axis. Auto-detected if None.

    Returns:
        Dict with:
            - x: shared 1D evaluation grid
            - clean_kdes: layer_name -> density array
            - corrupted_kdes: layer_name -> density array
    """
    clean_map = {l["layer_name"]: l for l in comparison["clean"]["layers"]}
    corr_map = {l["layer_name"]: l for l in comparison["corrupted"]["layers"]}
    common = comparison["common_layers"]

    if logit_range is None:
        all_vals = np.concatenate(
            [clean_map[n]["full_logits"] for n in common]
            + [corr_map[n]["full_logits"] for n in common]
        )
        lo = float(np.percentile(all_vals, 1))
        hi = float(np.percentile(all_vals, 99))
        margin = (hi - lo) * 0.1
        lo -= margin
        hi += margin
    else:
        lo, hi = logit_range

    x = np.linspace(lo, hi, n_points)

    clean_kdes, corrupted_kdes = {}, {}
    for name in common:
        clean_kdes[name] = _safe_kde(clean_map[name]["full_logits"], x)
        corrupted_kdes[name] = _safe_kde(corr_map[name]["full_logits"], x)

    return {"x": x, "clean_kdes": clean_kdes, "corrupted_kdes": corrupted_kdes}


def _safe_kde(data: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Compute KDE with fallback for degenerate distributions."""
    try:
        kde = gaussian_kde(data, bw_method="scott")
        return kde(x)
    except (np.linalg.LinAlgError, ValueError):
        counts, edges = np.histogram(data, bins=len(x), density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return np.interp(x, centers, counts)


def compute_evolution_heatmap(
    comparison: Dict,
    n_bins: int = 128,
    sigma: float = 1.5,
    mode: str = "clean",
    logit_range: Optional[Tuple[float, float]] = None,
) -> Dict:
    """
    Build a 2D heatmap: layers (y-axis) × logit bins (x-axis), smoothed
    with a gaussian kernel.

    Args:
        comparison: Output from run_layer_evolution_comparison.
        n_bins: Number of logit bins on the x-axis.
        sigma: Gaussian kernel sigma for smoothing.
        mode: "clean", "corrupted", or "diff" (clean minus corrupted).
        logit_range: (min, max) for binning. Auto-detected if None.

    Returns:
        Dict with:
            - heatmap: 2D numpy array (n_layers × n_bins)
            - bin_edges: 1D array of bin edges
            - bin_centers: 1D array of bin centers
            - layer_names: ordered list of layer names
            - mode: which mode was used
    """
    clean_map = {l["layer_name"]: l for l in comparison["clean"]["layers"]}
    corr_map = {l["layer_name"]: l for l in comparison["corrupted"]["layers"]}
    common = comparison["common_layers"]

    if logit_range is None:
        all_vals = np.concatenate(
            [clean_map[n]["full_logits"] for n in common]
            + [corr_map[n]["full_logits"] for n in common]
        )
        lo = float(np.percentile(all_vals, 1))
        hi = float(np.percentile(all_vals, 99))
    else:
        lo, hi = logit_range

    bin_edges = np.linspace(lo, hi, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    n_layers = len(common)
    heatmap = np.zeros((n_layers, n_bins), dtype=np.float64)

    for i, name in enumerate(common):
        if mode == "clean":
            data = clean_map[name]["full_logits"]
            counts, _ = np.histogram(data, bins=bin_edges, density=True)
            heatmap[i] = counts

        elif mode == "corrupted":
            data = corr_map[name]["full_logits"]
            counts, _ = np.histogram(data, bins=bin_edges, density=True)
            heatmap[i] = counts

        elif mode == "diff":
            c_counts, _ = np.histogram(
                clean_map[name]["full_logits"], bins=bin_edges, density=True
            )
            x_counts, _ = np.histogram(
                corr_map[name]["full_logits"], bins=bin_edges, density=True
            )
            heatmap[i] = c_counts - x_counts

    # Smooth with gaussian kernel
    if sigma > 0:
        heatmap = gaussian_filter(heatmap, sigma=sigma)

    return {
        "heatmap": heatmap,
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "layer_names": common,
        "mode": mode,
    }


def _build_token_trajectories(
    layers: List[Dict], top_k: int
) -> Dict[int, Dict[str, Any]]:
    """
    Track how individual tokens move through the rankings across layers.

    For every token that appears in any layer's top-k, record its
    probability at each layer.
    """
    all_token_ids = set()
    for layer in layers:
        for idx in layer["top_k_indices"]:
            all_token_ids.add(idx.item())

    trajectories = {}
    for tid in all_token_ids:
        probs_per_layer = []
        for layer in layers:
            indices = layer["top_k_indices"]
            match = (indices == tid).nonzero(as_tuple=True)
            if len(match[0]) > 0:
                prob = layer["top_k_probs"][match[0][0]].item()
            else:
                prob = 0.0
            probs_per_layer.append(prob)

        token_str = str(tid)
        if layers and "top_k_tokens" in layers[0]:
            for layer in layers:
                indices = layer["top_k_indices"]
                match = (indices == tid).nonzero(as_tuple=True)
                if len(match[0]) > 0:
                    token_str = layer["top_k_tokens"][match[0][0]]
                    break

        trajectories[tid] = {
            "token_id": tid,
            "token_str": token_str,
            "probs_per_layer": probs_per_layer,
            "max_prob": max(probs_per_layer),
            "final_prob": probs_per_layer[-1] if probs_per_layer else 0.0,
        }

    return trajectories


def _find_key_moments(
    layers: List[Dict],
    entropy_trajectory: List[float],
    kl_trajectory: List[float],
) -> Dict[str, Any]:
    """
    Identify important moments in the layer evolution:
    - First confidence (entropy drops below half of initial)
    - Biggest distribution shift (max KL)
    - Top-1 stabilization (last layer where top-1 changes)
    """
    moments = {}

    if len(entropy_trajectory) >= 2:
        initial_entropy = entropy_trajectory[0]
        half_entropy = initial_entropy / 2
        for i, ent in enumerate(entropy_trajectory):
            if ent < half_entropy:
                moments["first_confidence"] = {
                    "layer": layers[i]["layer_name"],
                    "layer_index": i,
                    "entropy": ent,
                }
                break

    kl_values = [k for k in kl_trajectory if k is not None]
    if kl_values:
        max_kl = max(kl_values)
        max_kl_idx = kl_trajectory.index(max_kl)
        moments["biggest_shift"] = {
            "layer": layers[max_kl_idx]["layer_name"],
            "layer_index": max_kl_idx,
            "kl_divergence": max_kl,
        }

    if len(layers) >= 2:
        last_change = 0
        prev_top1 = layers[0]["top_k_indices"][0].item()
        for i in range(1, len(layers)):
            curr_top1 = layers[i]["top_k_indices"][0].item()
            if curr_top1 != prev_top1:
                last_change = i
            prev_top1 = curr_top1
        moments["top1_stabilizes"] = {
            "layer": layers[last_change]["layer_name"],
            "layer_index": last_change,
        }

    return moments


def summarize_evolution(results: Dict) -> str:
    """Generate a human-readable summary of the layer evolution."""
    layers = results.get("layers", [])
    moments = results.get("key_moments", {})
    trajectories = results.get("token_trajectories", {})

    if not layers:
        return "No layer evolution data."

    lines = [
        f"Layer evolution: {results['num_layers']} layers analyzed",
        f"Position: {results.get('position_used', '?')}",
        "",
    ]

    if "first_confidence" in moments:
        m = moments["first_confidence"]
        lines.append(f"First confidence: {m['layer']} (entropy: {m['entropy']:.3f})")

    if "biggest_shift" in moments:
        m = moments["biggest_shift"]
        lines.append(f"Biggest shift: {m['layer']} (KL: {m['kl_divergence']:.3f})")

    if "top1_stabilizes" in moments:
        m = moments["top1_stabilizes"]
        lines.append(f"Top-1 stabilizes after: {m['layer']}")

    lines.append("")

    sorted_tokens = sorted(
        trajectories.values(),
        key=lambda t: t["final_prob"],
        reverse=True,
    )

    lines.append("Top token trajectories (final probability):")
    for t in sorted_tokens[:5]:
        first = t["probs_per_layer"][0]
        final = t["final_prob"]
        peak = t["max_prob"]
        direction = "↑" if final > first else "↓" if final < first else "→"
        lines.append(
            f"  {t['token_str']:15s} | "
            f"start: {first:.4f} → final: {final:.4f} | "
            f"peak: {peak:.4f} {direction}"
        )

    return "\n".join(lines)


def summarize_comparison(comparison: Dict) -> str:
    """Summarize the clean vs corrupted evolution comparison."""
    common = comparison["common_layers"]
    divs = comparison["divergences"]

    if not common:
        return "No common layers between clean and corrupted runs."

    lines = [
        f"Layer evolution comparison: {len(common)} layers",
        "",
        "Per-layer divergence (clean vs corrupted):",
        f"  {'Layer':<35s} {'KL':>8s} {'JS':>8s} {'L2':>8s}",
        f"  {'─' * 35} {'─' * 8} {'─' * 8} {'─' * 8}",
    ]

    for name in common:
        d = divs[name]
        lines.append(f"  {name:<35s} {d['kl']:8.4f} {d['js']:8.4f} {d['l2']:8.4f}")

    # Highlight most divergent layer
    max_js_layer = max(common, key=lambda n: divs[n]["js"])
    lines.append("")
    lines.append(
        f"Most divergent layer: {max_js_layer} " f"(JS: {divs[max_js_layer]['js']:.4f})"
    )

    return "\n".join(lines)
