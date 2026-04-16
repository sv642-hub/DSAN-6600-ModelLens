from __future__ import annotations
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
from modellens.analysis.activation_patching import run_activation_patching


def run_batch_patching(
    lens,
    input_pairs: List[Tuple[Any, Any]],
    *,
    metric_fn: Optional[Callable] = None,
    layer_names: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run activation patching over multiple clean/corrupted pairs and aggregate.

    Args:
        lens: ModelLens instance
        input_pairs: List of (clean_input, corrupted_input) tuples.
                     All pairs should test the same type of behavior
                     (e.g., capital-country knowledge).
        metric_fn: Custom metric function (same for all pairs)
        layer_names: Sublayers to patch. If None, auto-detects.

    Returns:
        Dict with per-sublayer aggregated importance scores,
        individual run results, and consistency metrics.
    """
    if not input_pairs:
        raise ValueError("input_pairs must not be empty.")

    model = lens.model

    # Clear hooks before starting
    for _, module in model.named_modules():
        module._forward_hooks.clear()

    # Run patching for each pair
    all_results = []
    for i, (clean, corrupted) in enumerate(input_pairs):
        # Clear hooks between runs
        for _, module in model.named_modules():
            module._forward_hooks.clear()

        try:
            result = run_activation_patching(
                lens,
                clean,
                corrupted,
                layer_names=layer_names,
                metric_fn=metric_fn,
                **kwargs,
            )
            result["pair_index"] = i
            all_results.append(result)
        except Exception as e:
            all_results.append(
                {
                    "pair_index": i,
                    "error": str(e),
                    "patch_effects": {},
                }
            )

    # Aggregate across all successful runs
    successful = [r for r in all_results if "error" not in r]

    if not successful:
        return {
            "aggregated": {},
            "all_results": all_results,
            "num_pairs": len(input_pairs),
            "num_successful": 0,
            "layers_ordered": [],
        }

    aggregated = _aggregate_results(successful)
    consistency = _compute_consistency(successful, aggregated)

    # Sort by mean absolute effect
    sorted_layers = sorted(
        aggregated.keys(),
        key=lambda k: abs(aggregated[k]["mean_normalized_effect"]),
        reverse=True,
    )

    return {
        "aggregated": aggregated,
        "all_results": all_results,
        "num_pairs": len(input_pairs),
        "num_successful": len(successful),
        "layers_ordered": sorted_layers,
        "consistency": consistency,
    }


def _aggregate_results(results: List[Dict]) -> Dict[str, Dict[str, Any]]:
    """Aggregate patching effects across multiple runs."""
    # Collect per-sublayer effects
    effects_by_layer: Dict[str, List[Dict]] = defaultdict(list)

    for result in results:
        for name, data in result.get("patch_effects", {}).items():
            effects_by_layer[name].append(data)

    # Compute aggregated stats
    aggregated = {}
    for name, effects in effects_by_layer.items():
        norm_effects = [e["normalized_effect"] for e in effects]
        patched_metrics = [e["patched_metric"] for e in effects]

        recovery_fracs = [
            e.get("recovery_fraction_of_gap", 0.0)
            for e in effects
            if "recovery_fraction_of_gap" in e
        ]

        restored_count = sum(1 for e in effects if e.get("prediction_restored", False))

        n = len(norm_effects)
        mean_effect = sum(norm_effects) / n
        abs_effects = [abs(x) for x in norm_effects]
        mean_abs_effect = sum(abs_effects) / n

        # Standard deviation
        if n > 1:
            variance = sum((x - mean_effect) ** 2 for x in norm_effects) / (n - 1)
            std_effect = variance**0.5
        else:
            std_effect = 0.0

        aggregated[name] = {
            "mean_normalized_effect": mean_effect,
            "mean_abs_normalized_effect": mean_abs_effect,
            "std_normalized_effect": std_effect,
            "min_normalized_effect": min(norm_effects),
            "max_normalized_effect": max(norm_effects),
            "mean_patched_metric": sum(patched_metrics) / n,
            "mean_recovery_fraction": (
                sum(recovery_fracs) / len(recovery_fracs) if recovery_fracs else 0.0
            ),
            "prediction_restored_rate": restored_count / n,
            "num_runs": n,
        }

    return aggregated


def _compute_consistency(results: List[Dict], aggregated: Dict) -> Dict[str, Any]:
    """
    Measure how consistent patching effects are across runs.

    High consistency = the component matters for all inputs of this type.
    Low consistency = the component only matters for specific inputs.
    """
    if len(results) < 2:
        return {"overall_consistency": 1.0, "per_layer": {}}

    per_layer = {}
    for name, agg in aggregated.items():
        std = agg["std_normalized_effect"]
        mean_abs = agg["mean_abs_normalized_effect"]

        # Coefficient of variation (lower = more consistent)
        if mean_abs > 1e-10:
            cv = std / mean_abs
            # Invert and clamp to 0-1 range (1 = perfectly consistent)
            consistency = max(0.0, 1.0 - cv)
        else:
            consistency = 0.0

        # How often does this component have the same sign effect?
        effects = []
        for r in results:
            e = r.get("patch_effects", {}).get(name, {})
            if "normalized_effect" in e:
                effects.append(e["normalized_effect"])

        if effects:
            positive = sum(1 for e in effects if e > 0)
            sign_agreement = max(positive, len(effects) - positive) / len(effects)
        else:
            sign_agreement = 0.0

        per_layer[name] = {
            "consistency_score": consistency,
            "sign_agreement": sign_agreement,
            "coefficient_of_variation": cv if mean_abs > 1e-10 else float("inf"),
        }

    # Overall consistency: mean across layers
    scores = [v["consistency_score"] for v in per_layer.values()]
    overall = sum(scores) / len(scores) if scores else 0.0

    return {
        "overall_consistency": overall,
        "per_layer": per_layer,
    }


def summarize_batch_patching(results: Dict, top_n: int = 10) -> str:
    """Generate a human-readable summary of batch patching results."""
    agg = results.get("aggregated", {})
    layers = results.get("layers_ordered", [])
    consistency = results.get("consistency", {})

    lines = [
        f"Batch patching: {results['num_successful']}/{results['num_pairs']} pairs successful",
        f"Overall consistency: {consistency.get('overall_consistency', 0):.3f}",
        "",
        f"Top {min(top_n, len(layers))} most important components:",
        "",
    ]

    per_layer_cons = consistency.get("per_layer", {})

    for name in layers[:top_n]:
        data = agg[name]
        cons = per_layer_cons.get(name, {})
        sign_agr = cons.get("sign_agreement", 0)
        cons_score = cons.get("consistency_score", 0)

        lines.append(
            f"  {name:35s} | mean effect: {data['mean_normalized_effect']:+.3f} "
            f"± {data['std_normalized_effect']:.3f} | "
            f"consistency: {cons_score:.2f} | "
            f"sign agreement: {sign_agr:.0%}"
        )

    return "\n".join(lines)
