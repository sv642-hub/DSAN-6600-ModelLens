"""run_probe_features on a tiny randomly initialized Qwen3 (no download)."""

import pytest
import torch
import transformers

from modellens import ModelLens
from modellens.analysis.probe_features import pool_activation, run_probe_features


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    config = transformers.Qwen3Config(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    return transformers.Qwen3ForCausalLM(config).eval()


def ids(*rows):
    return torch.tensor(rows)


def test_last_token_matches_hidden_states(model):
    lens = ModelLens(model, backend="huggingface")
    tokens = {"input_ids": ids([5, 6, 7, 8]), "attention_mask": torch.ones(1, 4)}
    features = lens.probe_features(tokens, ["model.layers.0", "model.layers.1"])
    hidden = model(**tokens, output_hidden_states=True).hidden_states
    # hidden_states[0] is the embeddings, so layer i is hidden_states[i + 1].
    # (The last entry has the final norm applied, so the last layer differs.)
    assert torch.allclose(features["model.layers.0"], hidden[1][:, -1], atol=1e-5)
    assert torch.allclose(features["model.layers.1"], hidden[2][:, -1], atol=1e-5)
    assert features["model.layers.0"].shape == (1, 32)


def test_mean_and_index_pooling(model):
    lens = ModelLens(model, backend="huggingface")
    tokens = ids([5, 6, 7])
    hidden = model(tokens, output_hidden_states=True).hidden_states[2]
    mean = run_probe_features(lens, tokens, "model.layers.1", pooling="mean")
    first = run_probe_features(lens, tokens, "model.layers.1", pooling=0)
    assert torch.allclose(mean["model.layers.1"], hidden.mean(dim=1), atol=1e-5)
    assert torch.allclose(first["model.layers.1"], hidden[:, 0], atol=1e-5)


def test_right_padding_uses_last_real_token(model):
    lens = ModelLens(model, backend="huggingface")
    alone = lens.probe_features(ids([5, 6]), "model.layers.1")["model.layers.1"]
    padded = {"input_ids": ids([5, 6, 0]), "attention_mask": torch.tensor([[1, 1, 0]])}
    batch = lens.probe_features(padded, "model.layers.1")["model.layers.1"]
    assert torch.allclose(batch, alone, atol=1e-5)


def test_existing_hooks_are_left_alone(model):
    lens = ModelLens(model, backend="huggingface")
    lens.attach_layers(["model.layers.0"])
    lens.probe_features(ids([1, 2]), "model.layers.2")
    assert len(lens.hooks) == 1
    lens.run(ids([1, 2]))
    assert list(lens.get_activations()) == ["model.layers.0"]
    lens.clear()


def test_bad_inputs_are_rejected(model):
    lens = ModelLens(model, backend="huggingface")
    with pytest.raises(ValueError, match="not found"):
        lens.probe_features(ids([1]), "model.layers.9")
    with pytest.raises(ValueError, match="at least one"):
        lens.probe_features(ids([1]), [])
    with pytest.raises(ValueError, match="tokenizer"):
        lens.probe_features("text", "model.layers.0")
    with pytest.raises(ValueError, match="pooling"):
        pool_activation(torch.zeros(1, 2, 3), "max")
    with pytest.raises(ValueError, match="out of range"):
        pool_activation(torch.zeros(1, 2, 3), 5)
    with pytest.raises(ValueError, match="shape"):
        pool_activation(torch.zeros(2, 3))
    with pytest.raises(ValueError, match="NaN"):
        pool_activation(torch.full((1, 1, 2), float("nan")))
