import streamlit as st
from modellens.analysis.activation_patching import run_activation_patching
from modellens.visualization import (
    plot_patching_importance_bar,
    plot_patching_importance_heatmap,
    plot_patching_recovery_fraction,
    plot_patching_family_effect_recovery_heatmap,
    format_patching_summary_html,
)


def render():
    st.header("Activation Patching")
    st.caption(
        "Measure each layer's causal importance by swapping activations "
        "between a clean and corrupted input."
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
            ["Effect Bar", "Recovery", "Heatmap", "Family Summary"],
            default="Effect Bar",
            label_visibility="collapsed",
        )
    with col3:
        with st.popover("⚙️ Settings"):
            display_mode = st.selectbox(
                "Display mode",
                ["full", "top_n", "family"],
                help="'full' shows all modules, 'top_n' shows the most impactful, 'family' groups by module type.",
            )
            top_n = st.slider(
                "Top N",
                min_value=5,
                max_value=50,
                value=20,
                help="Number of modules to show in 'top_n' mode.",
            )
            use_normalized = st.toggle(
                "Normalized effects",
                value=True,
                help="Normalize effects relative to the clean-corrupted gap.",
            )

    # ── Display results ──
    if "patching_results" in st.session_state:
        results = st.session_state["patching_results"]

        # Summary cards
        summary_html = format_patching_summary_html(results)
        st.markdown(summary_html, unsafe_allow_html=True)
        st.divider()

        if viz_mode == "Effect Bar":
            fig = plot_patching_importance_bar(
                results,
                use_normalized=use_normalized,
                display_mode=display_mode,
                top_n=top_n,
            )
            st.plotly_chart(fig, use_container_width=True)

        elif viz_mode == "Recovery":
            fig = plot_patching_recovery_fraction(
                results,
                display_mode=display_mode,
                top_n=top_n,
            )
            st.plotly_chart(fig, use_container_width=True)

        elif viz_mode == "Heatmap":
            fig = plot_patching_importance_heatmap(results)
            st.plotly_chart(fig, use_container_width=True)

        elif viz_mode == "Family Summary":
            fig = plot_patching_family_effect_recovery_heatmap(
                results,
                use_normalized=use_normalized,
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Raw effects expander ──
        with st.expander("Raw patch effects"):
            pe = results.get("patch_effects", {})
            for layer, data in pe.items():
                eff = data.get("normalized_effect", data.get("effect", 0))
                rec = data.get("recovery_fraction_of_gap", 0)
                st.text(f"{layer:40s}  effect={eff:+.4f}  recovery={rec:.4f}")

    # ── Prompt inputs ──
    # Corrupted prompt lives in sidebar when on this tab
    with st.sidebar:
        st.divider()
        st.markdown("**Patching Prompts**")
        corrupted_prompt = st.text_input(
            "Corrupted prompt",
            value=st.session_state.get("patching_corrupted", ""),
            help="A modified version of your prompt (e.g., replace a key word).",
        )
        st.session_state["patching_corrupted"] = corrupted_prompt

    prompt = st.chat_input("Enter a clean prompt to analyze")
    if prompt:
        corrupted = st.session_state.get("patching_corrupted", "")
        if not corrupted:
            st.error("Enter a corrupted prompt in the sidebar first.")
            return

        with st.spinner("Running activation patching..."):
            from config.utils import tokenize_prompt

            lens.clear()
            clean_inputs = tokenize_prompt(prompt, model_info)
            corrupted_inputs = tokenize_prompt(corrupted, model_info)

            results = run_activation_patching(lens, clean_inputs, corrupted_inputs)
            lens.clear()

            st.session_state["patching_results"] = results
            st.session_state["patching_clean_prompt"] = prompt

            st.rerun()
