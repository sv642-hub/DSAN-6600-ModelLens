"""transformer-sharanya/predict.py

Load a trained checkpoint and run sentiment predictions.

Usage:
  python predict.py "this movie was great"
"""

from __future__ import annotations

import sys

import torch

from data import Vocab, encode
from model import SentimentTransformer


def load_checkpoint(path: str = "trained_sentiment_transformer_sharanya.pt"):
    ckpt = torch.load(path, map_location="cpu")
    token_to_id = ckpt["vocab"]
    id_to_token = {i: t for t, i in token_to_id.items()}
    vocab = Vocab(token_to_id=token_to_id, id_to_token=id_to_token)

    model = SentimentTransformer(
        vocab_size=len(token_to_id),
        hidden_dim=ckpt["hidden_dim"],
        num_heads=ckpt["num_heads"],
        num_layers=ckpt["num_layers"],
        max_seq_len=ckpt["max_len"],
        pad_id=ckpt.get("pad_id", 0),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, vocab, ckpt["max_len"]


@torch.no_grad()
def predict(text: str, model: SentimentTransformer, vocab: Vocab, max_len: int):
    input_ids, attn = encode(text, vocab, max_len)
    logits = model(input_ids.unsqueeze(0), attention_mask=attn.unsqueeze(0))
    prob = torch.softmax(logits, dim=-1)[0]
    label = int(prob.argmax().item())
    return label, prob.tolist()


def main(argv: list[str]):
    if len(argv) < 2:
        print("Provide text: python predict.py \"some review text\"")
        return 2

    text = " ".join(argv[1:])
    model, vocab, max_len = load_checkpoint()
    label, probs = predict(text, model, vocab, max_len)

    name = "positive" if label == 1 else "negative"
    print(f"pred: {name} | probs [neg, pos] = {[round(p, 4) for p in probs]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
