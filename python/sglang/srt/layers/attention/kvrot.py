import math
from typing import Optional

import torch

from sglang.kernels.ops.quantization.hadamard import hadamard_transform


def get_qkrot_block() -> Optional[int]:
    """Return the configured Q/K Hadamard rotation block size, if any."""
    try:
        from sglang.srt.runtime_context import get_server_args

        return getattr(get_server_args(), "qkrot_block", None)
    except ValueError:
        return None


def apply_block_hadamard_rotation(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Apply an orthonormal block Hadamard rotation over the last dimension."""
    if block_size <= 0:
        raise ValueError(f"rotation block size must be positive, got {block_size}.")

    head_dim = x.shape[-1]
    if head_dim % block_size != 0:
        raise ValueError(
            f"rotation block size {block_size} must divide head_dim {head_dim}."
        )

    if head_dim == block_size:
        return hadamard_transform(x.contiguous(), scale=1.0 / math.sqrt(block_size))

    original_shape = x.shape
    x = x.reshape(*original_shape[:-1], head_dim // block_size, block_size)
    x = hadamard_transform(x.contiguous(), scale=1.0 / math.sqrt(block_size))
    return x.reshape(original_shape)
