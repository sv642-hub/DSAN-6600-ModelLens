"""
Transformer model for the natural-language arithmetic task.

Architecture is a standard pre-norm decoder-only transformer:
  - Token embedding + learned positional embedding
  - Stack of N transformer blocks (each: pre-norm, self-attention, residual,
    pre-norm, MLP, residual)
  - Final layer norm
  - Unembedding (linear projection back to vocab size)

This is intentionally structured to be compatible with the ModelLens PyTorch
adapter — submodules are named so the adapter can hook them, and each block
exposes `.attn` and `.mlp` so activation patching can target them separately.
"""

import torch
import torch.nn as nn

from data import VOCAB_SIZE


class TransformerBlock(nn.Module):
    """Single pre-norm transformer block: attention then MLP, both with residuals."""

    def __init__(self, hidden_dim, num_heads, mlp_ratio=4):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        )

    def forward(self, x):
        # Attention sublayer with residual.
        normed = self.ln_1(x)
        seq_len = x.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_out, _ = self.attn(
            normed, normed, normed,
            attn_mask=causal_mask,
            need_weights=False,
        )
        x = x + attn_out

        # MLP sublayer with residual.
        normed = self.ln_2(x)
        x = x + self.mlp(normed)

        return x


class ArithmeticTransformer(nn.Module):
    """
    Decoder-only transformer for natural-language arithmetic.

    Defaults are tuned to be small enough to train in minutes on a laptop
    while being big enough to actually learn the task.
    """

    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        hidden_dim=128,
        num_heads=4,
        num_layers=4,
        max_seq_len=32,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(hidden_dim)
        self.unembed = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids):
        # input_ids: (batch, seq_len) of token IDs
        batch_size, seq_len = input_ids.shape

        # Build position indices [0, 1, 2, ..., seq_len-1] for each batch row.
        positions = torch.arange(seq_len, device=input_ids.device)

        # Add token and position embeddings.
        x = self.embed(input_ids) + self.pos_embed(positions)

        # Pass through transformer blocks.
        for block in self.blocks:
            x = block(x)

        # Final norm and projection back to vocab.
        x = self.ln_f(x)
        logits = self.unembed(x)

        return logits


if __name__ == "__main__":
    # Sanity check: build the model and run a dummy forward pass.
    model = ArithmeticTransformer()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created.")
    print(f"Vocab size:  {VOCAB_SIZE}")
    print(f"Hidden dim:  {model.hidden_dim}")
    print(f"Parameters:  {num_params:,}")
    print()

    # Forward pass on a fake batch.
    fake_input = torch.randint(0, VOCAB_SIZE, (2, 16))
    output = model(fake_input)
    print(f"Input shape:  {fake_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"(Expected output: (batch, seq_len, vocab_size) = (2, 16, {VOCAB_SIZE}))")
    print()
 
    # Show the named submodules — useful for verifying ModelLens compatibility later.
    print("Named modules:")
    for name, _ in model.named_modules():
        if name and name.count(".") <= 2:  # don't print too deep
            print(f"  {name}")