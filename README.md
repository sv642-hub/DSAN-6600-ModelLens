# ModelLens

An open-source interpretability toolkit for PyTorch neural networks. ModelLens provides a unified interface for analyzing model internals across different architectures, sitting between architecture-specific tools like TransformerLens and general-purpose libraries like Captum.

## Features

- **Activation Hooks** — Capture intermediate activations from any layer using PyTorch forward hooks
- **Logit Lens** — Project hidden states through the unembedding matrix to observe how predictions evolve across layers
- **Attention Analysis** — Extract and analyze attention weight maps across heads and layers
- **Activation Patching** — Causal intervention analysis to identify which components drive specific behaviors
- **Embeddings Inspection** — Analyze input embedding representations and token similarity
- **Residual Stream Analysis** — Trace information flow through skip connections and measure per-layer contributions
- **Visualization** — Plotly-based figures (heatmaps, trajectories, patching bars, shape traces) for notebooks and slides
- **Streamlit App** — Guided multi-tab explorer

## Supported Backends

- **HuggingFace** — Any `PreTrainedModel` (GPT-2, BERT, LLaMA, etc.); attention weights via `output_attentions=True`
- **PyTorch** — Vanilla `nn.Module` models; attention hooks work when modules expose weights in outputs (see limitations below)

## Installation

```bash
git clone https://github.com/your-username/modellens.git
cd modellens
pip install -e .
```

Visualization (Plotly, pandas) and the web app (Streamlit):

```bash
pip install -e ".[viz,app]"
```

Optional development extras:

```bash
pip install -e ".[viz,app,dev]"
```

## Quick Start

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from modellens import ModelLens

model = GPT2LMHeadModel.from_pretrained("gpt2-medium", attn_implementation="eager")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")

lens = ModelLens(model)
lens.adapter.set_tokenizer(tokenizer)

tokens = tokenizer("The capital of France is", return_tensors="pt")
lens.attach_all()
results = lens.logit_lens(tokens, top_k=5)  # passes tokenizer from adapter for decoded top-k
```

## Visualization (package API)

Analysis functions return dicts with stable keys for plotting (`token_labels`, `layers_ordered`, etc.). Build figures from `modellens.visualization`:

```python
from modellens.analysis.attention import run_attention_analysis
from modellens.visualization import plot_attention_heatmap, plot_logit_lens_heatmap

attn = run_attention_analysis(lens, tokens)
fig = plot_attention_heatmap(attn, layer_index=0, head_index=0)
fig.show()

ll = lens.logit_lens(tokens, top_k=5)
fig2 = plot_logit_lens_heatmap(ll, top_ranks=5)
fig2.show()
```

Export HTML for slides:

```python
fig.write_html("attention.html")
```

Quick script (writes `viz_out/*.html`):

```bash
python examples/quick_viz_demo.py --out ./viz_out
```

## Streamlit App

After `pip install -e ".[app]"`:

```bash
modellens-streamlit
# or
python -m app.main
```

Open the URL printed in the terminal. **Load a model** on the Overview tab first, then use Attention, Logit Lens, Patching, Residual/Embedding, or **Presentation story** (curated narrative).

## Analysis Modules

### Logit Lens

```python
from modellens.analysis.logit_lens import run_logit_lens, decode_logit_lens

results = run_logit_lens(lens, tokens, top_k=5, tokenizer=tokenizer)
decoded = decode_logit_lens(results, tokenizer=tokenizer)
```

`results` includes `layers_ordered` and, when `tokenizer` is passed, `top_tokens_per_layer` / `top_probs_per_layer`.

### Attention Analysis

```python
from modellens.analysis.attention import run_attention_analysis, head_summary

attn_results = run_attention_analysis(lens, tokens)  # prefer tokenized dict for labels
summary = head_summary(attn_results)
```

Returns `token_labels`, `layers_ordered`, and `backend`.

### Activation Patching

```python
from modellens.analysis.activation_patching import run_activation_patching

clean = tokenizer("The capital of France is", return_tensors="pt")
corrupted = tokenizer("The MX of France is", return_tensors="pt")

results = run_activation_patching(lens, clean, corrupted)  # optional layer_names=...
```

Clean and corrupted sequences must have the **same length**. Patching calls `lens.clear()` at the start (ModelLens hooks only), not a global hook wipe.

### Residual Stream Analysis

```python
from modellens.analysis.residual_stream import run_residual_analysis, identify_critical_layers

results = run_residual_analysis(lens, tokens, layer_names=block_layers)
critical = identify_critical_layers(results, threshold=0.05)
```

### Embeddings Inspection

```python
from modellens.analysis.embeddings import run_embeddings_analysis

results = run_embeddings_analysis(lens, tokens)
print(results["similarity_matrix"])
print(results.get("token_labels"))
```

## Project Structure

```
modellens/
│   ├── core/            # ModelLens, HookManager
│   ├── adapters/        # HuggingFace and PyTorch adapters
│   ├── analysis/        # Interpretability routines
│   └── visualization/   # Plotly helpers + shared styling
app/                     # Streamlit presentation shell (main.py, components.py, demo_data.py)
examples/                # Notebooks and quick_viz_demo.py
```

## Limitations

- **Vanilla PyTorch attention**: `nn.MultiheadAttention` does not return attention weights unless configured (`need_weights=True`). Hook-based attention capture may not apply to all custom modules.
- **Residual analysis** compares consecutive hooked activations with **matching shapes**; `attach_all` order follows `named_modules()`.
