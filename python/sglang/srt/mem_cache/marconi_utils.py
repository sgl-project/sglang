from __future__ import annotations

from typing import List


def normalize(values: List[float]) -> List[float]:
    if len(values) > 1:
        min_val = min(values)
        max_val = max(values)
        if min_val != max_val:
            return [(val - min_val) / (max_val - min_val) for val in values]
    return [1.0] * len(values)


def get_attn_flops(seq_len: int, model_dim: int) -> int:
    return 8 * seq_len * model_dim**2 + 4 * seq_len**2 * model_dim


def get_mlp_flops(seq_len: int, model_dim: int) -> int:
    return 16 * seq_len * model_dim**2


def get_mamba1_flops(seq_len: int, model_dim: int, state_size: int) -> int:
    return (
        12 * seq_len * model_dim**2
        + 16 * seq_len * model_dim * state_size
        + 10 * seq_len * model_dim
    )


def get_kv_cache_size_bytes(seq_len: int, model_dim: int, dtype_size: int) -> int:
    return 2 * seq_len * model_dim * dtype_size
