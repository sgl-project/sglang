from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from sglang.srt.configs.draft_parallel import DraftParallelLayout
from sglang.srt.utils.common import is_float4_e2m1fn_x2

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig


@dataclass(frozen=True)
class DFlashKVCacheBudget:
    layout: DraftParallelLayout
    cell_size: int
    fixed_bytes: int


def dflash_draft_cell_size_per_token(
    *,
    model_config: ModelConfig,
    num_layers: int,
    kv_cache_dtype: torch.dtype,
    layout: DraftParallelLayout,
) -> int:
    """Price the draft's own MHA geometry, using its execution TP width."""
    heads = model_config.get_num_kv_heads(layout.attn_tp_size)
    key_dim = model_config.head_dim
    value_dim = model_config.v_head_dim
    element_size = torch._utils._element_size(kv_cache_dtype)
    size = heads * (key_dim + value_dim) * num_layers * element_size
    if is_float4_e2m1fn_x2(kv_cache_dtype):
        # Match MHATokenToKVPoolFP4's packed values and per-16 scales.
        size = size // 2 + heads * key_dim * num_layers * 2 * element_size // 16
    if size <= 0:
        raise ValueError("DFlash draft KV geometry must have a positive cell size")
    return size
