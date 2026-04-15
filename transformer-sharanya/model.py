"""transformer-sharanya/model.py

A small decoder-only Transformer for sentiment analysis.

This mirrors the style of `transformer-fareeza/` (clean, modular) and the
minimalism of `transformer-jeff/` (simple Transformer blocks).

Task: binary sentiment classification (negative/positive).

Model:
- token embedding + learned positional embedding
- N pre-norm transformer blocks (causal self-attention)
- mean pool (masked) over sequence
- linear classification head (2 logits)

This is intentionally lightweight and CPU/MPS friendly.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        x = self.ln_1(x)
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = residual + attn_out

        residual = x
        x = self.ln_2(x)
        x = residual + self.mlp(x)
        return x


class SentimentTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        max_seq_len: int = 128,
        num_classes: int = 2,
        pad_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id

        self.embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_id)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return logits of shape (batch, num_classes)."""
        bsz, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}")

        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.embed(input_ids) + self.pos_embed(positions)

        # causal mask: True where positions should be masked
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )

        for block in self.blocks:
            x = block(x, attn_mask=causal_mask)

        x = self.ln_f(x)

        # masked mean pooling
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_id).to(x.dtype)
        else:
            attention_mask = attention_mask.to(x.dtype)

        denom = attention_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (x * attention_mask.unsqueeze(-1)).sum(dim=1) / denom

        return self.classifier(pooled)
