"""Pure-torch reference for TRT-LLM GenMHA's native NVFP4 V-scale layout."""

import torch


def nvfp4_v_scale_swizzle_indices(
    token_offsets: torch.Tensor, scale_indices: torch.Tensor, scale_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return TRT-LLM's four-token-interleaved V-scale coordinates.

    This small torch reference is intentionally device agnostic so the layout
    contract can be unit-tested without a GPU. ``scale_dim`` is head_dim / 16.

    FlashInfer's ``nvfp4_block_scale_interleave`` is the 128x4 scale layout for
    GEMM/MoE and is not this KV-cache layout. Its slot-mapping paged-KV append
    writes linear V scales; its complete-cache conversion helper applies this
    four-token permutation, but only while rewriting the whole cache. The store
    kernel in ``sglang.srt.layers.quantization.nvfp4_kv_cache`` instead fuses the
    permutation with SGLang's incremental slot scatter so it does not materialize
    another cache-sized scale tensor.
    """
    if scale_dim % 4 != 0:
        raise ValueError(f"NVFP4 scale_dim must be divisible by 4, got {scale_dim}.")
    scale_group = scale_dim // 4
    swizzled_token = (token_offsets // 4) * 4 + scale_indices // scale_group
    swizzled_scale = (scale_indices % scale_group) * 4 + token_offsets % 4
    return swizzled_token, swizzled_scale
