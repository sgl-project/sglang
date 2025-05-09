import torch
from typing import Tuple

from .utils import import_torch_op_if_available


def flash_mla_with_kvcache(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    dv: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.sgl_kernel.flash_mla_with_kvcache(
        q,
        kv_cache,
        block_table,
        cache_seqlens,
        dv,
        tile_scheduler_metadata,
        num_splits,
        softmax_scale,
        causal,
    )


def get_mla_metadata(
    cache_seqlens: torch.Tensor, q_heads_per_kv_head: int, num_kv_heads: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.sgl_kernel.get_mla_metadata(
        cache_seqlens, q_heads_per_kv_head, num_kv_heads
    )
