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
def _jit_k_indexer_norm_rope_module(dtype: torch.dtype, is_neox: bool):
    args = make_cpp_args(dtype, is_arch_support_pdl(), is_neox)
    return load_jit(
        f"dpsk_v32_k_indexer_norm_rope{'_neox' if is_neox else ''}",
        *args,
        cuda_files=[_CUDA_FILE],
        cuda_wrappers=[
            ("forward", f"FusedKIndexerNormRopeKernel<{args}>::forward"),
        ],
    )


@cache_once
def _jit_k_indexer_norm_rope_store_module(
    dtype: torch.dtype, page_size: int, is_neox: bool
):
    args = make_cpp_args(dtype, is_arch_support_pdl(), page_size, is_neox)
    return load_jit(
        f"dpsk_v32_k_indexer_norm_rope_store_p{page_size}{'_neox' if is_neox else ''}",
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
    *,
    is_neox: bool = False,
) -> torch.Tensor:
    """V3.2 indexer K: LayerNorm + RoPE on leading dims -> bf16. CUDA only.

    is_neox selects the RoPE convention: True pairs dim i with i + rope_dim/2
    (NeoX, DeepSeek-V3.2), False pairs (2i, 2i+1) (interleave/GPT-J, GLM-5.x).
    The cos_sin_cache halves layout [cos..., sin...] is the same for both.
    """
    # k_input may be a non-contiguous wk slice; output is always contiguous.
    k_out = torch.empty(k_input.shape, dtype=k_input.dtype, device=k_input.device)
    module = _jit_k_indexer_norm_rope_module(k_input.dtype, is_neox)
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
    is_neox: bool = False,
) -> None:
    """V3.2 indexer K + fused store: LayerNorm + RoPE on leading dims + fp8
    act-quant + paged index-k cache write, in one launch. CUDA only.

    See fused_k_indexer_norm_rope for the is_neox convention.
    """
    if not out_cache_loc.is_contiguous():
        out_cache_loc = out_cache_loc.contiguous()
    module = _jit_k_indexer_norm_rope_store_module(k_input.dtype, page_size, is_neox)
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
