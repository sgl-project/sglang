# Adapted from https://github.com/thinking-machines-lab/batch_invariant_ops/blob/main/batch_invariant_ops/__init__.py

from typing import Optional

import torch
import torch.nn.functional as F

from .batch_invariant_ops import (
    AttentionBlockSize,
    disable_batch_invariant_mode,
    enable_batch_invariant_mode,
    get_batch_invariant_attention_block_size,
    is_batch_invariant_mode_enabled,
    log_softmax,
    matmul_persistent,
    mean_dim,
    rms_norm_batch_invariant,
    set_batch_invariant_mode,
)

__version__ = "0.1.0"


def batch_invariant_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Batch-invariant drop-in replacement for F.linear.

    Flattens leading dims to 2-D, routes through ``matmul_persistent``
    (which gives identical results regardless of padding / batch size),
    then restores the original leading shape.

    Falls back to ``F.linear`` for fp32 inputs (persistent matmul is
    bf16/fp16-only) while correctly toggling the mode flag.
    """
    x_shape = x.shape
    x_2d = x.reshape(-1, x_shape[-1])
    if weight.dtype != x_2d.dtype:
        weight = weight.to(x_2d.dtype)
    if bias is not None and bias.dtype != x_2d.dtype:
        bias = bias.to(x_2d.dtype)

    if x_2d.dtype == torch.float32:
        mode_was_enabled = is_batch_invariant_mode_enabled()
        if mode_was_enabled:
            disable_batch_invariant_mode()
        try:
            output = F.linear(x_2d, weight, bias)
        finally:
            if mode_was_enabled:
                enable_batch_invariant_mode()
    else:
        output = matmul_persistent(x_2d, weight.t(), bias)

    return output.reshape(*x_shape[:-1], weight.shape[0])


__all__ = [
    "set_batch_invariant_mode",
    "is_batch_invariant_mode_enabled",
    "disable_batch_invariant_mode",
    "enable_batch_invariant_mode",
    "batch_invariant_linear",
    "matmul_persistent",
    "log_softmax",
    "mean_dim",
    "get_batch_invariant_attention_block_size",
    "AttentionBlockSize",
    "rms_norm_batch_invariant",
]
