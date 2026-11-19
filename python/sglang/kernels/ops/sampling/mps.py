"""Torch MPS greedy reduction without a vocabulary-wide serial argmax."""

import torch


def greedy_argmax(logits: torch.Tensor) -> torch.Tensor:
    """Return int64 [batch] token IDs, preserving first-maximum/first-NaN ties."""
    if (
        logits.device.type != "mps"
        or logits.ndim != 2
        or logits.shape[-1] == 0
        or logits.dtype not in (torch.float32, torch.float16, torch.bfloat16)
    ):
        raise ValueError("MPS greedy argmax requires floating [batch, nonempty vocab]")
    batch, vocab = logits.shape
    block_size = 1024
    if batch == 0 or vocab <= block_size:
        return torch.argmax(logits, dim=-1)
    padding = (-vocab) % block_size
    if padding:
        logits = torch.cat(
            (logits, logits.new_full((batch, padding), -float("inf"))), dim=-1
        )
    blocks = logits.reshape(batch, -1, block_size)
    local_ids = torch.argmax(blocks, dim=-1, keepdim=True)
    winners = blocks.gather(-1, local_ids).squeeze(-1)
    # Ordered blocks preserve first-index ties, including NaNs and all -inf.
    block_ids = torch.argmax(winners, dim=-1, keepdim=True)
    return (
        local_ids.squeeze(-1).gather(-1, block_ids) + block_ids * block_size
    ).squeeze(-1)
