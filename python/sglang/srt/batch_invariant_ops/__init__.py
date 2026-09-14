# Adapted from https://github.com/thinking-machines-lab/batch_invariant_ops/blob/main/batch_invariant_ops/__init__.py

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
    true_on_policy_rms_norm,
)

__version__ = "0.1.0"

__all__ = [
    "set_batch_invariant_mode",
    "is_batch_invariant_mode_enabled",
    "disable_batch_invariant_mode",
    "enable_batch_invariant_mode",
    "matmul_persistent",
    "log_softmax",
    "mean_dim",
    "get_batch_invariant_attention_block_size",
    "AttentionBlockSize",
    "rms_norm_batch_invariant",
    "true_on_policy_rms_norm",
]
