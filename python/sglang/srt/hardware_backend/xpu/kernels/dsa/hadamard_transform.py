"""Torch-native Walsh-Hadamard transform for XPU (no JIT/Triton needed)."""

import torch


def hadamard_transform(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Normalized Walsh-Hadamard transform along the last dimension.

    Uses the recursive butterfly decomposition: O(N log N) multiplies.
    The last dimension must be a power of 2.

    Args:
        x: Input tensor with power-of-2 last dimension.
        scale: Scaling factor to apply after the transform (e.g., hidden_size^{-0.5}).

    Returns:
        Normalized Hadamard transform of x (scale = hidden_size^{-0.5}).
    """
    hidden_size = x.size(-1)
    assert (
        hidden_size & (hidden_size - 1)
    ) == 0, "Hidden size must be a power of 2 for Hadamard transform."

    orig_shape = x.shape
    x = x.reshape(-1, hidden_size).float()
    h = 1
    while h < hidden_size:
        # Butterfly: split into pairs of blocks of size h
        x = x.view(-1, hidden_size // (2 * h), 2, h)
        a = x[:, :, 0, :]
        b = x[:, :, 1, :]
        x = torch.stack([a + b, a - b], dim=2)
        x = x.view(-1, hidden_size)
        h *= 2
    return (x * scale).to(torch.bfloat16).view(orig_shape)
