"""Activation features for linear probes.

A linear probe needs one fixed-length vector per input. ``run_probe_features``
runs one forward pass, reads the output of one or more layers, and pools each
over tokens into a vector:

- ``last``: the last real token (padding is skipped when an attention mask is
  given).
- ``mean``: the average over real tokens.
- an int: the token at that position (negative counts from the end).

The hooks are registered for this call only and removed afterwards, so hooks
attached through ``ModelLens.attach_*`` are left as they were.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import torch

from modellens.analysis.hf_inputs import hf_inputs_to_dict

Pooling = Union[str, int]
POOLINGS = ("last", "mean")


def _layer_output(output: Any) -> torch.Tensor:
    """The hidden-state tensor from a layer's output."""
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, tuple):
        return output[0]
    if isinstance(output, torch.Tensor):
        return output
    raise TypeError(f"Cannot read a tensor from layer output {type(output)}.")


def pool_activation(
    activation: torch.Tensor,
    pooling: Pooling = "last",
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pool ``[batch, tokens, hidden]`` to ``[batch, hidden]``.

    With ``attention_mask`` (``[batch, tokens]``, 1 for real tokens), ``last``
    and ``mean`` use only real tokens, so right-padded batches pool correctly.
    """
    if activation.dim() != 3:
        raise ValueError(
            f"Expected an activation of shape [batch, tokens, hidden], "
            f"got {tuple(activation.shape)}."
        )
    activation = activation.float()
    batch, tokens, _ = activation.shape
    if tokens == 0:
        raise ValueError("No tokens to pool.")

    if attention_mask is None:
        mask = torch.ones(batch, tokens, device=activation.device)
    else:
        mask = attention_mask.to(activation.device).float()
        if mask.shape != (batch, tokens):
            raise ValueError("attention_mask must have shape [batch, tokens].")
        if (mask.sum(dim=1) == 0).any():
            raise ValueError("Every row needs at least one real token.")

    if isinstance(pooling, bool) or not isinstance(pooling, (str, int)):
        raise ValueError(f"pooling must be one of {POOLINGS} or a token index.")
    if isinstance(pooling, int):
        if not -tokens <= pooling < tokens:
            raise ValueError(f"Token index {pooling} is out of range.")
        pooled = activation[:, pooling, :]
    elif pooling == "last":
        # Index of the last real token in each row (handles right padding).
        positions = torch.arange(tokens, device=activation.device).expand(batch, -1)
        last = (positions * mask).argmax(dim=1)
        pooled = activation[torch.arange(batch), last, :]
    elif pooling == "mean":
        weights = mask.unsqueeze(-1)
        pooled = (activation * weights).sum(dim=1) / weights.sum(dim=1)
    else:
        raise ValueError(f"pooling must be one of {POOLINGS} or a token index.")

    if not torch.isfinite(pooled).all():
        raise ValueError("Pooled activations contain NaN or infinite values.")
    return pooled


def run_probe_features(
    lens,
    inputs,
    layer_names: Union[str, Sequence[str]],
    pooling: Pooling = "last",
) -> Dict[str, torch.Tensor]:
    """Pooled activations from one forward pass, keyed by layer name.

    ``inputs`` is a string (when the adapter has a tokenizer), tokenizer
    output, or a tensor of token ids. Each value has shape ``[batch, hidden]``
    on the CPU, as float32.

    For a HuggingFace Qwen3 or LLaMA model the decoder layers are named
    ``model.layers.0``, ``model.layers.1``, and so on; ``lens.layer_names()``
    lists every name. The output is the raw layer output: for the last layer
    it is before the final norm, unlike the last entry of transformers'
    ``hidden_states``.
    """
    names: List[str] = [layer_names] if isinstance(layer_names, str) else list(layer_names)
    if not names:
        raise ValueError("Give at least one layer name.")
    modules = dict(lens.model.named_modules())
    missing = [name for name in names if name not in modules]
    if missing:
        raise ValueError(f"Layers not found: {missing}. See lens.layer_names().")

    if isinstance(inputs, str):
        tokenizer = getattr(lens.adapter, "_tokenizer", None)
        if tokenizer is None:
            raise ValueError("A string input needs adapter.set_tokenizer(tokenizer).")
        inputs = tokenizer(inputs, return_tensors="pt")

    attention_mask = None
    if not isinstance(inputs, torch.Tensor):
        attention_mask = hf_inputs_to_dict(inputs).get("attention_mask")

    captured: Dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name: str):
        def hook(module, args, output):
            captured[name] = _layer_output(output).detach()

        return hook

    try:
        for name in names:
            handles.append(modules[name].register_forward_hook(make_hook(name)))
        with torch.no_grad():
            lens.adapter.forward(lens.model, inputs)
    finally:
        for handle in handles:
            handle.remove()

    return {
        name: pool_activation(captured[name], pooling, attention_mask).cpu()
        for name in names
    }
