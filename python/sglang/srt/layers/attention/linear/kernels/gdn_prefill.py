from dataclasses import dataclass
from typing import Tuple

import torch

from sglang.srt.utils import is_cuda, is_hip

MAX_FUSED_QKV_SPLIT_DIM = 8192


@dataclass(frozen=True, slots=True)
class GDNQKVShape:
    num_q_heads: int
    num_k_heads: int
    num_v_heads: int
    head_q_dim: int
    head_k_dim: int
    head_v_dim: int

    @property
    def q_dim(self) -> int:
        return self.num_q_heads * self.head_q_dim

    @property
    def k_dim(self) -> int:
        return self.num_k_heads * self.head_k_dim

    @property
    def v_dim(self) -> int:
        return self.num_v_heads * self.head_v_dim

    @property
    def total_dim(self) -> int:
        return self.q_dim + self.k_dim + self.v_dim


def split_gdn_prefill_qkv(
    mixed_qkv: torch.Tensor, shape: GDNQKVShape
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split packed post-convolution QKV into the canonical GDN layout."""
    assert mixed_qkv.shape[1] == shape.total_dim
    if (is_cuda() or is_hip()) and shape.total_dim <= MAX_FUSED_QKV_SPLIT_DIM:
        from sglang.jit_kernel.triton.gdn_fused_proj import (
            fused_qkv_split_gdn_prefill,
        )

        return fused_qkv_split_gdn_prefill(
            mixed_qkv,
            shape.num_q_heads,
            shape.num_k_heads,
            shape.num_v_heads,
            shape.head_q_dim,
            shape.head_k_dim,
            shape.head_v_dim,
        )

    query, key, value = torch.split(
        mixed_qkv, [shape.q_dim, shape.k_dim, shape.v_dim], dim=-1
    )
    seq_len = mixed_qkv.shape[0]
    return (
        query.view(1, seq_len, shape.num_q_heads, shape.head_q_dim),
        key.view(1, seq_len, shape.num_k_heads, shape.head_k_dim),
        value.view(1, seq_len, shape.num_v_heads, shape.head_v_dim),
    )
