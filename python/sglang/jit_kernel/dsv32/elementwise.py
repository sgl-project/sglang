"""DSA only. Indexer K kernels (JIT)."""

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

_CUDA_FILE = "deepseek_v32/indexer_k.cuh"


@cache_once
def _jit_k_indexer_norm_rope_module(dtype: torch.dtype):
    args = make_cpp_args(dtype, is_arch_support_pdl())
    return load_jit(
        "dpsk_v32_k_indexer_norm_rope",
        *args,
        cuda_files=[_CUDA_FILE],
        cuda_wrappers=[
            ("forward", f"FusedKIndexerNormRopeKernel<{args}>::forward"),
        ],
    )


@cache_once
def _jit_k_indexer_norm_rope_store_module(
    dtype: torch.dtype, page_size: int, is_fp4: bool
):
    args = make_cpp_args(dtype, is_arch_support_pdl(), page_size, is_fp4)
    return load_jit(
        f"dpsk_v32_k_indexer_norm_rope_store_p{page_size}{'_fp4' if is_fp4 else ''}",
        *args,
        cuda_files=[_CUDA_FILE],
        cuda_wrappers=[
            ("forward", f"FusedKIndexerNormRopeStoreKernel<{args}>::forward"),
        ],
    )


def fused_k_indexer_norm_rope(
    k_input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """V3.2 indexer K: LayerNorm + RoPE on leading dims -> bf16. CUDA only."""
    # k_input may be a non-contiguous wk slice; output is always contiguous.
    k_out = torch.empty(k_input.shape, dtype=k_input.dtype, device=k_input.device)
    module = _jit_k_indexer_norm_rope_module(k_input.dtype)
    module.forward(
        k_input,
        k_out,
        weight,
        bias,
        cos_sin_cache,
        positions,
        float(eps),
    )
    return k_out


def fused_k_indexer_norm_rope_store(
    k_input: torch.Tensor,
    cache: torch.Tensor,
    out_cache_loc: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    page_size: int,
    *,
    is_fp4: bool = False,
) -> None:
    """V3.2 indexer K + fused store: LayerNorm + RoPE on leading dims +
    act-quant + paged index-k cache write, in one launch. CUDA only.

    is_fp4 selects the quant/store format: False writes the FP8 132 B/token
    layout (128 B fp8 key + 4 B fp32 scale); True writes the MXFP4 68 B/token
    layout (64 B packed E2M1 pairs + 4 UE8M0 group exponents), bit-identical
    to store_fp4_index_k_cache on the bf16-rounded key."""
    if not out_cache_loc.is_contiguous():
        out_cache_loc = out_cache_loc.contiguous()
    module = _jit_k_indexer_norm_rope_store_module(k_input.dtype, page_size, is_fp4)
    module.forward(
        k_input,
        cache,
        out_cache_loc,
        weight,
        bias,
        cos_sin_cache,
        positions,
        float(eps),
    )
