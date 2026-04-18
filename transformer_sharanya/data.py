from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import torch
from torch.utils.data import Dataset


SPECIAL_TOKENS = ["<pad>", "<unk>"]
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def basic_tokenize(text: str) -> list[str]:
    text = text.lower()
    # keep words and apostrophes, drop other punctuation
    return re.findall(r"[a-z0-9]+'?[a-z0-9]+|[a-z0-9]+", text)


@dataclass
class Vocab:
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    @property
    def size(self) -> int:
        return len(self.token_to_id)


def build_vocab(texts: Iterable[str], min_freq: int = 2, max_size: int = 20000) -> Vocab:
    from collections import Counter

    counter: Counter[str] = Counter()
    for t in texts:
        counter.update(basic_tokenize(t))

    tokens = SPECIAL_TOKENS[:]
    for tok, freq in counter.most_common():
        if freq < min_freq:
            continue
        if tok in tokens:
            continue
        tokens.append(tok)
        if len(tokens) >= max_size:
            break

    token_to_id = {t: i for i, t in enumerate(tokens)}
    id_to_token = {i: t for t, i in token_to_id.items()}
    return Vocab(token_to_id=token_to_id, id_to_token=id_to_token)


def encode(text: str, vocab: Vocab, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    toks = basic_tokenize(text)[:max_len]
    ids = [vocab.token_to_id.get(tok, vocab.unk_id) for tok in toks]
    attn = [1] * len(ids)

    # pad
    if len(ids) < max_len:
        pad_n = max_len - len(ids)
        ids = ids + [vocab.pad_id] * pad_n
        attn = attn + [0] * pad_n

    return torch.tensor(ids, dtype=torch.long), torch.tensor(attn, dtype=torch.long)


class SentimentDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], vocab: Vocab, max_len: int):
        assert len(texts) == len(labels)
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        input_ids, attention_mask = encode(self.texts[idx], self.vocab, self.max_len)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label,
        }


def load_imdb_subset(train_size: int = 5000, test_size: int = 1000, seed: int = 0):
    """Load a small IMDB subset via `datasets`.

    Labels: 0 = negative, 1 = positive
    """

    try:
        from datasets import load_dataset  
    except Exception as e:  
        raise RuntimeError(
            "Hugging Face `datasets` is required for IMDB loading. "
            "Install with `pip install datasets`."
        ) from e

    ds = load_dataset("imdb")
    train = ds["train"].shuffle(seed=seed).select(range(train_size))
    test = ds["test"].shuffle(seed=seed).select(range(test_size))

    train_texts = list(train["text"])
    train_labels = list(train["label"])
    test_texts = list(test["text"])
    test_labels = list(test["label"])

    return (train_texts, train_labels), (test_texts, test_labels)


def load_toy_sentiment():
    texts = [
        "I loved this movie it was fantastic and heartwarming",
        "Absolutely terrible film, boring and too long",
        "Great acting and a wonderful story",
        "Worst experience ever, I hate this",
        "It was okay, some parts were good",
        "Brilliant and inspiring, would watch again",
        "Not good, plot made no sense",
        "Amazing! best thing I've seen all year",
    ]
    labels = [1, 0, 1, 0, 1, 1, 0, 1]
    # simple split
    train_texts, train_labels = texts[:6], labels[:6]
    test_texts, test_labels = texts[6:], labels[6:]
    return (train_texts, train_labels), (test_texts, test_labels)
