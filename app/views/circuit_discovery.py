"""Causal circuit sketch from patching-style importance (package ``discover_circuit``)."""

import os
import sys

import streamlit as st

_vdir = os.path.dirname(os.path.abspath(__file__))
_appdir = os.path.dirname(_vdir)
if _appdir not in sys.path:
    sys.path.insert(0, _appdir)

from components import run_circuit_discovery_fig
from config.interpretability import compute_output_comparison, render_prompt_output_cards
from config.prompt_sync import (
    get_shared_corrupted,
    merge_chat_and_shared_clean,
    shared_prompt_status_row,
    shared_prompts_callout,
    shared_run_hint,
)


def render():
    st.header("Circuit discovery")
    st.caption(
        "This page builds a candidate circuit from patching-style importance plus lightweight "
        "routing hints. It is useful for hypotheses and follow-up tests — not a standalone causal proof."
    )
    st.caption(
        "Read the pathway graphic left to right for approximate depth; bar charts below rank "
        "components and connection weights for inspection."
    )
    shared_prompts_callout()
    shared_prompt_status_row()
    shared_run_hint()

    model_info = st.session_state.get("model_info")
    if not model_info:
        st.warning("Load a model first in Model Setup.")
        return

    lens = model_info["lens"]
    tokenizer = model_info.get("tokenizer")
    if tokenizer is not None and hasattr(lens.adapter, "set_tokenizer"):
        lens.adapter.set_tokenizer(tokenizer)

    with st.popover("Settings"):
        threshold = st.slider(
            "Importance threshold",
            min_value=0.05,
            max_value=0.6,
            value=0.22,
            step=0.02,
            help=(
                "Modules below this normalized-effect level are dropped from the graph. "
                "On GPT-2-sized models, ~0.15–0.35 often yields a readable sketch; raise it if the graph is crowded."
            ),
        )

    if "circuit_discovery_cache" not in st.session_state:
        st.info(
            "Discover builds the graph from a full clean vs corrupt patching pass — "
            "it can take longer than a single Patching run. Lower the threshold slightly if the graph is empty."
        )

    if "circuit_discovery_cache" in st.session_state:
        c = st.session_state["circuit_discovery_cache"]
        if c.get("clean") and c.get("corrupted"):
            render_prompt_output_cards(
                model_info,
                str(c["clean"]),
                str(c["corrupted"]),
                c.get("forward_compare"),
                patched_summary=(
                    "The pathway diagram proposes how important components connect; "
                    "validate with targeted patching or ablations."
                ),
            )
        st.markdown(c["summary_html"], unsafe_allow_html=True)

        st.subheader("Candidate pathway")
        st.caption(
            "This view organizes discovered components into a candidate flow through the model. "
            "Larger nodes reflect stronger estimated effect; thicker lines reflect stronger proposed links."
        )
        if c.get("fig_flow") is not None:
            st.plotly_chart(
                c["fig_flow"], use_container_width=True, key="circuit_fig_flow"
            )
        else:
            st.info(
                "Run **Discover** again to generate the pathway diagram (layout was updated)."
            )
        st.caption(
            "Connections summarize proposed routing and should be treated as hypotheses for follow-up testing, "
            "not confirmed mechanisms."
        )

        st.subheader("Supporting charts")
        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(
                c["fig_nodes"], use_container_width=True, key="circuit_fig_nodes"
            )
            st.caption(
                "Ranked effect magnitudes help compare candidates when the pathway graphic is crowded."
            )
        with g2:
            st.plotly_chart(
                c["fig_edges"], use_container_width=True, key="circuit_fig_edges"
            )
            st.caption(
                "Connection weights summarize inferred sequential and attention routing edges."
            )

    c1, _ = st.columns([1, 5])
    with c1:
        run_sb = st.button(
            "Discover",
            type="primary",
            key="circuit_run_sidebar",
            help="Use shared clean & corrupted prompts from the Analysis sidebar",
        )
    chat = st.chat_input("Clean prompt (or sidebar + Discover)")
    clean = merge_chat_and_shared_clean(chat, run_sb)
    if run_sb and not clean:
        st.error("Set a clean prompt in the sidebar, or use the chat bar.")
    elif clean:
        corrupted = get_shared_corrupted()
        if not corrupted:
            st.error("Set a corrupted prompt in the Analysis sidebar.")
        else:
            with st.spinner("Discovering circuit…"):
                try:
                    summary_html, fig_flow, fig_nodes, fig_edges = (
                        run_circuit_discovery_fig(
                            lens,
                            clean,
                            corrupted,
                            float(threshold),
                        )
                    )
                except Exception as e:
                    st.error(f"{type(e).__name__}: {e}")
                    return
                try:
                    fwd_cmp = compute_output_comparison(model_info, clean, corrupted)
                except Exception:
                    fwd_cmp = None
                st.session_state["circuit_discovery_cache"] = {
                    "summary_html": summary_html,
                    "fig_flow": fig_flow,
                    "fig_nodes": fig_nodes,
                    "fig_edges": fig_edges,
                    "clean": clean,
                    "corrupted": corrupted,
                    "forward_compare": fwd_cmp,
                }
                st.rerun()
