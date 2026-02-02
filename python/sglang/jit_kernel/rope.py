from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_apply_rope_pos_ids_cos_sin_cache_module() -> Module:
    return load_jit(
        "apply_rope_pos_ids_cos_sin_cache",
        cuda_files=["elementwise/rope.cuh"],
        cuda_wrappers=[
            (
                "apply_rope_pos_ids_cos_sin_cache",
                "ApplyRopePosIdsCosSinCacheKernel::run",
            )
        ],
        extra_ldflags=[
            "-L/usr/local/lib/python3.12/dist-packages/torch/lib",
            "-lc10",
            "-ltorch",
        ],
        extra_include_paths=[
            "/usr/local/lib/python3.12/dist-packages/flashinfer/data/include",
            "/usr/local/lib/python3.12/dist-packages/torch/include",
        ],
    )


def apply_rope_pos_ids_cos_sin_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    pos_ids: torch.Tensor,
    interleave: bool = False,
    enable_pdl: bool = False,
    v: Optional[torch.Tensor] = None,
    k_buffer: Optional[torch.Tensor] = None,
    v_buffer: Optional[torch.Tensor] = None,
    kv_cache_loc: Optional[torch.Tensor] = None,
) -> None:
    """
    Apply RoPE (Rotary Positional Embedding) with position IDs and cos/sin cache.

    Args:
        q: Input Q tensor of shape [nnz, num_qo_heads, head_dim]
        k: Input K tensor of shape [nnz, num_kv_heads, head_dim]
        q_rope: Output Q tensor with RoPE applied, same shape as q
        k_rope: Output K tensor with RoPE applied, same shape as k
        cos_sin_cache: Cos/sin cache of shape [max_seq_len, rotary_dim]
        pos_ids: Position IDs of shape [nnz]
        interleave: Whether to use interleaved RoPE
        enable_pdl: Enable PDL (Programmable Data Layout)
        v: Optional V tensor for KV caching
        k_buffer: Optional K buffer for KV caching
        v_buffer: Optional V buffer for KV caching
        kv_cache_loc: Optional KV cache location tensor
    """
    module = _jit_apply_rope_pos_ids_cos_sin_cache_module()

    module.apply_rope_pos_ids_cos_sin_cache(
        q,
        k,
        q_rope,
        k_rope,
        cos_sin_cache,
        pos_ids,
        interleave,
        enable_pdl,
        v,
        k_buffer,
        v_buffer,
        kv_cache_loc,
    )
