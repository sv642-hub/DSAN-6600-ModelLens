from __future__ import annotations

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    # When run as a module: python -m transformer_sharanya.train
    from .data import SentimentDataset, build_vocab, load_imdb_subset, load_toy_sentiment
    from .model import SentimentTransformer
except ImportError:  # pragma: no cover
    # When run as a script: python transformer_sharanya/train.py
    from data import SentimentDataset, build_vocab, load_imdb_subset, load_toy_sentiment
    from model import SentimentTransformer


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids, attention_mask=attn)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    model.train()
    return correct / max(total, 1)


def train(
    *,
    max_len: int = 128,
    hidden_dim: int = 128,
    num_heads: int = 4,
    num_layers: int = 4,
    batch_size: int = 32,
    lr: float = 3e-4,
    epochs: int = 3,
    train_size: int = 5000,
    test_size: int = 1000,
    device: str | None = None,
):
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    # Load data
    try:
        (train_texts, train_labels), (test_texts, test_labels) = load_imdb_subset(
            train_size=train_size, test_size=test_size
        )
        dataset_name = "imdb"
    except Exception:
        (train_texts, train_labels), (test_texts, test_labels) = load_toy_sentiment()
        dataset_name = "toy"

    vocab = build_vocab(train_texts, min_freq=2, max_size=20000)

    train_ds = SentimentDataset(train_texts, train_labels, vocab=vocab, max_len=max_len)
    test_ds = SentimentDataset(test_texts, test_labels, vocab=vocab, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = SentimentTransformer(
        vocab_size=vocab.size,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=max_len,
        pad_id=vocab.pad_id,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"Dataset: {dataset_name} | device: {device}")
    print(f"Vocab size: {vocab.size} | max_len: {max_len}")

    start = time.time()
    for epoch in range(1, epochs + 1):
        running = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask=attn)
            loss = criterion(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += loss.item()

        acc = evaluate(model, test_loader, device=device)
        print(f"epoch {epoch:2d} | loss {running/len(train_loader):.4f} | acc {acc:.1%}")

    elapsed = time.time() - start
    print(f"Training done in {elapsed:.1f}s")

    ckpt = {
        "model_state_dict": model.state_dict(),
        "vocab": vocab.token_to_id,
        "max_len": max_len,
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "pad_id": vocab.pad_id,
    }
    save_path = "trained_sentiment_transformer_sharanya.pt"
    torch.save(ckpt, save_path)
    print(f"Saved checkpoint to {save_path}")

    return model, vocab


if __name__ == "__main__":
    train()
