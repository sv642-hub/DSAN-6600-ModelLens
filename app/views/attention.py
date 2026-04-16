import streamlit as st
import numpy as np
from modellens.analysis.attention import (
    run_attention_analysis,
    head_summary,
    compute_attention_pattern_metrics,
    run_comparative_attention,
)
from modellens.visualization import (
    plot_attention_heatmap,
    plot_attention_head_grid,
    plot_attention_head_entropy,
)


def render():
    st.header("Attention")
    st.caption(
        "Visualize which tokens each attention head focuses on. "
        "Low entropy = focused head. High entropy = diffuse."
    )

    # ── Check model is loaded ──
    model_info = st.session_state.get("model_info")
    if not model_info:
        st.warning("Load a model first in ⚙️ Model Setup.")
        return

    lens = model_info["lens"]
    tokenizer = model_info["tokenizer"]

    # ── Controls ──
    col1, col2, col3 = st.columns([3, 5, 1])
    with col1:
        viz_mode = st.pills(
            "Visualization",
            ["Heatmap", "Head Grid", "Entropy", "Comparative"],
            default="Heatmap",
            label_visibility="collapsed",
        )
    with col3:
        with st.popover("⚙️ Settings"):
            layer_idx = st.slider(
                "Layer index",
                min_value=0,
                max_value=48,
                value=0,
                help="Which transformer layer to inspect.",
            )
            head_idx = st.slider(
                "Head index",
                min_value=0,
                max_value=24,
                value=0,
                help="Which attention head within the layer.",
            )
            max_heads = st.slider(
                "Max heads (grid/entropy)",
                min_value=1,
                max_value=16,
                value=8,
                help="Maximum heads to display in grid and entropy views.",
            )

    # ── Display results ──
    if "attention_results" in st.session_state:
        attn_results = st.session_state["attention_results"]

        # Get actual layer count for clamping
        ordered = attn_results.get("layers_ordered") or list(
            attn_results.get("attention_maps", {}).keys()
        )
        safe_layer = min(layer_idx, len(ordered) - 1) if ordered else 0
        num_heads = 0
        if ordered:
            w = attn_results["attention_maps"][ordered[safe_layer]]["weights"]
            if hasattr(w, "dim") and w.dim() == 4:
                num_heads = w.shape[1]
        safe_head = min(head_idx, max(num_heads - 1, 0))

        if viz_mode == "Heatmap":
            fig = plot_attention_heatmap(
                attn_results,
                layer_index=safe_layer,
                head_index=safe_head,
            )
            st.plotly_chart(fig, use_container_width=True)

        elif viz_mode == "Head Grid":
            fig = plot_attention_head_grid(
                attn_results,
                layer_index=safe_layer,
                max_heads=max_heads,
            )
            st.plotly_chart(fig, use_container_width=True)

        elif viz_mode == "Entropy":
            fig = plot_attention_head_entropy(
                attn_results,
                layer_index=safe_layer,
                max_heads=max_heads,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Pattern metrics summary
            metrics = compute_attention_pattern_metrics(attn_results)
            layer_name = ordered[safe_layer] if ordered else "unknown"
            layer_metrics = metrics.get("per_layer", {}).get(layer_name)
            if layer_metrics:
                st.divider()
                st.subheader("Pattern Metrics")
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Mean Entropy", f"{layer_metrics['mean_entropy']:.3f}")
                mc2.metric(
                    "Argmax Distance", f"{layer_metrics['mean_argmax_distance']:.2f}"
                )
                mc3.metric("Pattern Hint", layer_metrics["pattern_hint"])

        elif viz_mode == "Comparative":
            if "comparative_attention" in st.session_state:
                comp = st.session_state["comparative_attention"]
                if comp.get("error"):
                    st.error(f"Comparative attention error: {comp['error']}")
                else:
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots
                    from modellens.visualization.common import default_plotly_layout

                    # Re-run comparative with updated layer/head
                    clean_prompt = st.session_state.get("attention_prompt", "")
                    corrupted_prompt = st.session_state.get(
                        "attention_corrupted_prompt", ""
                    )
                    if clean_prompt and corrupted_prompt and tokenizer:
                        comp_col1, comp_col2 = st.columns(2)
                        with comp_col1:
                            comp_layer = st.slider(
                                "Layer",
                                min_value=0,
                                max_value=len(ordered) - 1 if ordered else 0,
                                value=(
                                    min(layer_idx, len(ordered) - 1) if ordered else 0
                                ),
                                key="comp_layer",
                            )
                        with comp_col2:
                            comp_head = st.slider(
                                "Head",
                                min_value=0,
                                max_value=24,
                                value=0,
                                key="comp_head",
                            )

                        clean_tokens = tokenizer(clean_prompt, return_tensors="pt")
                        corrupted_tokens = tokenizer(
                            corrupted_prompt, return_tensors="pt"
                        )
                        comp = run_comparative_attention(
                            lens,
                            clean_tokens,
                            corrupted_tokens,
                            layer_index=comp_layer,
                            head_index=comp_head,
                        )
                        lens.clear()

                    labels = comp["token_labels"]
                    fig = make_subplots(
                        rows=1,
                        cols=3,
                        subplot_titles=["Clean", "Corrupted", "Delta"],
                        horizontal_spacing=0.08,
                    )
                    for i, (mat, cs) in enumerate(
                        [
                            (comp["clean_weights"], "Blues"),
                            (comp["corrupted_weights"], "Reds"),
                            (comp["delta_weights"], "RdBu_r"),
                        ]
                    ):
                        z = (
                            mat.detach().cpu().numpy()
                            if hasattr(mat, "detach")
                            else mat
                        )
                        fig.add_trace(
                            go.Heatmap(
                                z=z,
                                x=labels,
                                y=labels,
                                colorscale=cs,
                                showscale=(i == 2),
                                texttemplate="%{z:.2f}",
                            ),
                            row=1,
                            col=i + 1,
                        )
                    fig.update_layout(
                        **default_plotly_layout(
                            title=f"Comparative — {comp.get('layer_name_clean', '')} head {comp.get('head_index', 0)}",
                            width=1200,
                            height=450,
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Entropy delta
                    ent_delta = comp.get("entropy_delta_per_head", [])
                    if ent_delta:
                        with st.expander("Entropy delta per head"):
                            for i, d in enumerate(ent_delta):
                                st.text(f"Head {i}: {d:+.4f}")
            else:
                st.info(
                    "Enter both a clean and corrupted prompt to see comparative attention."
                )

        # ── Head summary expander ──
        with st.expander("Head summary"):
            summaries = head_summary(attn_results)
            for name, data in summaries.items():
                entropy = data.get("entropy", [])
                max_attn = data.get("max_attention", [])
                st.text(f"{name}")
                if entropy:
                    st.text(f"  Entropy:       {[f'{e:.2f}' for e in entropy]}")
                if max_attn:
                    st.text(f"  Max attention:  {[f'{a:.2f}' for a in max_attn]}")

    # ── Prompt input ──
    prompt = st.chat_input("Enter a prompt to analyze attention")
    if prompt:
        with st.spinner("Running attention analysis..."):
            lens.clear()

            from config.utils import tokenize_prompt

            tokens = tokenize_prompt(prompt, model_info)

            if tokenizer:
                lens.adapter.set_tokenizer(tokenizer)

            try:
                attn_results = run_attention_analysis(lens, tokens)
            except Exception as e:
                st.error(f"Attention analysis failed: {e}")
                import traceback

                traceback.print_exc()
                return

            if not attn_results.get("attention_maps"):
                st.error(f"No attention maps returned. Debug: {attn_results}")
                return

            lens.clear()

            st.session_state["attention_results"] = attn_results
            st.session_state["attention_prompt"] = prompt

            # If we have a corrupted prompt stored, run comparative
            corrupted = st.session_state.get("attention_corrupted_prompt")
            if corrupted:
                from config.utils import tokenize_prompt

                clean_tokens = tokenize_prompt(prompt, model_info)
                corrupted_tokens = tokenize_prompt(corrupted, model_info)
                comp = run_comparative_attention(
                    lens,
                    clean_tokens,
                    corrupted_tokens,
                    layer_index=layer_idx,
                    head_index=head_idx,
                )
                st.session_state["comparative_attention"] = comp
                lens.clear()

            st.rerun()

    # ── Corrupted prompt for comparative mode ──
    if viz_mode == "Comparative":
        with st.sidebar:
            st.divider()
            corrupted_prompt = st.text_input(
                "Corrupted prompt (for comparative)",
                value=st.session_state.get("attention_corrupted_prompt", ""),
                help="Enter a corrupted version of your prompt to compare attention patterns.",
            )
            if corrupted_prompt != st.session_state.get(
                "attention_corrupted_prompt", ""
            ):
                st.session_state["attention_corrupted_prompt"] = corrupted_prompt
