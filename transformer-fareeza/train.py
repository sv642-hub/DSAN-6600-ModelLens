"""
Training loop for the natural-language arithmetic transformer.

We train the model to predict the digit tokens of the answer given an
English-word-encoded addition prompt. Loss is only computed on the answer
positions (using the mask from data.generate_batch).
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import generate_batch, decode, VOCAB_SIZE, TOKEN_TO_ID, END_ID
from model import ArithmeticTransformer


def compute_loss(logits, targets, loss_mask):
    """
    Compute masked cross-entropy loss.

    Args:
        logits:    (batch, seq_len, vocab_size) — raw model outputs
        targets:   (batch, seq_len) — token IDs the model should predict
        loss_mask: (batch, seq_len) — 1.0 where loss should count, 0.0 elsewhere

    Returns:
        A scalar loss tensor.
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Flatten everything to (batch * seq_len, vocab_size) and (batch * seq_len,)
    # so we can use F.cross_entropy in its standard form.
    flat_logits = logits.reshape(-1, vocab_size)
    flat_targets = targets.reshape(-1)
    flat_mask = loss_mask.reshape(-1)

    # Per-token cross-entropy loss (no reduction yet — one loss per position).
    per_token_loss = F.cross_entropy(flat_logits, flat_targets, reduction="none")

    # Apply the mask and average over the positions that count.
    masked_loss = per_token_loss * flat_mask
    loss = masked_loss.sum() / flat_mask.sum().clamp(min=1.0)

    return loss


@torch.no_grad()
def evaluate(model, num_problems=200, max_value=999, device="cpu"):
    """
    Generate fresh addition problems and measure accuracy.

    A problem counts as correct if the model produces exactly the right
    digit sequence followed by <end>.
    """
    model.eval()

    correct = 0
    for _ in range(num_problems):
        a = torch.randint(0, max_value + 1, (1,)).item()
        b = torch.randint(0, max_value + 1, (1,)).item()
        true_answer = a + b

        # Build the prompt portion only (everything up to and including "equals").
        from data import number_to_words, TOKEN_TO_ID
        prompt_tokens = number_to_words(a) + ["plus"] + number_to_words(b) + ["equals"]
        prompt_ids = [TOKEN_TO_ID[t] for t in prompt_tokens]

        # Greedy decode: feed the prompt, take argmax, append, repeat until <end>
        # or until we hit a max length safety cap.
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        max_new_tokens = 6  # answer is at most 4 digits + <end>, plus a safety margin

        generated_digits = []
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            next_id = logits[0, -1, :].argmax().item()
            if next_id == END_ID:
                break
            generated_digits.append(next_id)
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_id]], device=device)],
                dim=1,
            )

        # Convert generated digit IDs back to a number string.
        from data import ID_TO_TOKEN
        generated_str = "".join(ID_TO_TOKEN[i] for i in generated_digits)

        if generated_str == str(true_answer):
            correct += 1

    accuracy = correct / num_problems
    model.train()
    return accuracy


def train(
    num_steps=20000,
    batch_size=64,
    learning_rate=1e-3,
    eval_every=500,
    device=None,
):
    # Auto-detect best device if not specified.
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    print(f"Training on device: {device}")

    model = ArithmeticTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Steps: {num_steps}, batch size: {batch_size}, lr: {learning_rate}")
    print()

    model.train()
    start_time = time.time()

    for step in range(1, num_steps + 1):
        input_ids, target_ids, loss_mask = generate_batch(batch_size)
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        loss_mask = loss_mask.to(device)

        logits = model(input_ids)
        loss = compute_loss(logits, target_ids, loss_mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            elapsed = time.time() - start_time
            print(f"step {step:5d} | loss {loss.item():.4f} | {elapsed:.1f}s")

        if step % eval_every == 0:
            acc = evaluate(model, num_problems=200, device=device)
            print(f"  -> eval accuracy: {acc:.1%}")

    total_time = time.time() - start_time
    print(f"\nTraining done in {total_time:.1f}s")

    final_acc = evaluate(model, num_problems=500, device=device)
    print(f"Final accuracy on 500 fresh problems: {final_acc:.1%}")

    return model


if __name__ == "__main__":
    model = train()

    # Save the trained weights so we can load them in the demo notebook later.
    torch.save(model.state_dict(), "trained_model.pt")
    print("\nModel saved to trained_model.pt")