"""ROCm-only index-K store of the DeepSeek-V4 FP4 indexer into the split FlyDSL layout.

The HIP counterpart of ``index_k_norm_rope_pack_store`` in fp4_rope.py: the same
rmsnorm + RoPE + FP4 pack, written as separate payload and scale buffers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

from .utils import make_name

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_index_k_split_module(
    head_dim: int, rope_dim: int, page_size: int, ratio: int
) -> Module:
    args = make_cpp_args(head_dim, rope_dim, page_size, ratio, is_arch_support_pdl())
    return load_jit(
        make_name("fp4_rope_split_hip"),
        *args,
        cuda_files=["deepseek_v4/fp4_rope_hip.cuh"],
        cuda_wrappers=[("index_k_split", f"FlashIndexKSplitKernel<{args}>::run")],
    )


def index_k_norm_rope_pack_store_split(
    input: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    loc: torch.Tensor,
    payload: torch.Tensor,
    scale: torch.Tensor,
    *,
    ratio: int,
) -> None:
    """:func:`~sglang.kernels.ops.attention.dsv4.fp4_rope.index_k_norm_rope_pack_store`
    into the split FlyDSL K layout of the ROCm low-ratio indexer pool: ``payload``
    ``[npages, 1, 4, page_size, 16]`` (chunk ``c`` holds elements ``[32c, 32c + 32)`` of every
    slot) and ``scale`` ``[npages, 1, 4, page_size]`` uint8, one ue8m0 exponent per chunk with
    the slot axis transposed as a 16 x 4 tile; the bytes equal
    ``store_fp4_index_k_cache_split``'s."""
    head_dim = input.shape[-1]
    page_size = payload.shape[3]
    assert payload.shape[1:] == (1, 4, page_size, 16), payload.shape
    assert scale.shape[1:] == (1, 4, page_size), scale.shape
    _jit_index_k_split_module(
        head_dim, freqs_cis.shape[-1], page_size, ratio
    ).index_k_split(
        input,
        norm_weight,
        freqs_cis,
        positions,
        loc,
        payload.view(torch.uint8),
        scale,
        float(eps),
    )
