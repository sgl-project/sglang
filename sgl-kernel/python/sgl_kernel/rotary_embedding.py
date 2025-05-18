from typing import Optional

import torch


# Adapted from https://github.com/vllm-project/vllm/blob/9214e60631a79506e7669650de87806a123e0b0b/vllm/_custom_ops.py#L249
# pos encoding ops
def rotary_embedding(
    cos: torch.Tensor,
    sin: torch.Tensor,
    query: torch.Tensor,
    key: Optional[torch.Tensor],
    head_size: int,
    # cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    torch.ops.sgl_kernel.rotary_embedding.default(
        cos,
        sin,
        query,
        key,
        head_size,
        # cos_sin_cache,
        is_neox,
    )


# def batched_rotary_embedding(positions: torch.Tensor, query: torch.Tensor,
#                              key: Optional[torch.Tensor], head_size: int,
#                              cos_sin_cache: torch.Tensor, is_neox: bool,
#                              rot_dim: int,
#                              cos_sin_cache_offsets: torch.Tensor) -> None:
#     torch.ops._C.batched_rotary_embedding(positions, query, key, head_size,
#                                           cos_sin_cache, is_neox, rot_dim,
#                                           cos_sin_cache_offsets)
