import streamlit as st
from modellens.analysis.logit_lens import run_logit_lens, decode_logit_lens
from modellens.visualization import (
    plot_logit_lens_evolution,
    plot_logit_lens_heatmap,
    plot_logit_lens_top_token_bars,
)


def render():
    st.header("Logit Lens")
    st.caption(
        "Project each layer's hidden state through the unembedding matrix "
        "to see how the model's prediction evolves layer by layer."
    )

    # ── Check model is loaded ──
    model_info = st.session_state.get("model_info")
    if not model_info:
        st.warning("Load a model first in Model Setup.")
        return

    lens = model_info["lens"]
    tokenizer = model_info["tokenizer"]
    vocab = model_info.get("vocab")

    # ── Settings row ──
    col1, col2, col3 = st.columns([3, 10, 1])
    with col1:
        viz_mode = st.pills(
            "Visualization",
            ["Evolution", "Heatmap", "Top Token Bars"],
            default="Top Token Bars",
            label_visibility="collapsed",
        )
    with col3:
        with st.popover("Settings"):
            top_k = st.slider(
                "Top-K",
                min_value=1,
                max_value=20,
                value=5,
                help="Number of top predicted tokens to show per layer.",
            )
            max_tokens = st.slider(
                "Max tokens",
                min_value=5,
                max_value=100,
                value=20,
                help="How many tokens the model generates after your prompt.",
            )
            layer_filter = st.selectbox(
                "Layer filter",
                ["blocks", "attn", "mlp", "all"],
                index=0,
                help="Filter which layers appear in the plot.",
            )

    # ── Display results if we have them ──
    if "logit_lens_results" in st.session_state:
        decoded = st.session_state["logit_lens_decoded"]
        results = st.session_state["logit_lens_results"]

        if viz_mode == "Evolution":
            fig = plot_logit_lens_evolution(results, layer_filter=layer_filter)
        elif viz_mode == "Heatmap":
            fig = plot_logit_lens_heatmap(results, layer_filter=layer_filter)
        elif viz_mode == "Top Token Bars":
            fig = plot_logit_lens_top_token_bars(results, decoded=decoded)

        st.plotly_chart(fig, use_container_width=True)

        # ── Raw predictions expander ──
        with st.expander("Raw predictions"):
            for layer_name, predictions in decoded.items():
                tokens_str = ", ".join(
                    f"{tok!r} ({prob:.3f})" for tok, prob in predictions
                )
                st.text(f"{layer_name:40s} → {tokens_str}")

    # ── Model generation output ──
    if "logit_lens_generation" in st.session_state:
        st.divider()
        st.subheader("Model Prompt")
        st.markdown(f"{st.session_state['logit_lens_prompt']}")
        st.divider()
        st.subheader("Model Output")
        st.markdown(f"{st.session_state['logit_lens_generation']}")

    # ── Prompt input ──
    prompt = st.chat_input("Enter a prompt to analyze")
    if prompt:
        with st.spinner("Running logit lens..."):
            lens.clear()
            lens.attach_all()

            from config.utils import tokenize_prompt, generate_local

            tokens = tokenize_prompt(prompt, model_info)

            results = run_logit_lens(lens, tokens, top_k=top_k)
            decoded = decode_logit_lens(results, tokenizer=tokenizer, vocab=vocab)
            lens.clear()

            # ── Generate actual output ──
            generation = ""
            if tokenizer:
                import torch

                input_ids = tokens["input_ids"]
                with torch.no_grad():
                    output_ids = model_info["model"].generate(
                        input_ids,
                        max_new_tokens=max_tokens,
                        do_sample=False,
                    )
                generation = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            else:
                vocab = model_info.get("vocab", {})
                if vocab:
                    generation = generate_local(model_info["model"], tokens, vocab)
                else:
                    generation = "(No vocab available for generation)"

            st.session_state["logit_lens_results"] = results
            st.session_state["logit_lens_decoded"] = decoded
            st.session_state["logit_lens_prompt"] = prompt
            st.session_state["logit_lens_generation"] = generation

            st.rerun()
