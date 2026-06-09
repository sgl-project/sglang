"""DeepSeek-V3.2 DSA indexer K kernels (JIT).

V3.2-specific: the indexer K is single-head with a LayerNorm (not RMS), ropes
the leading dims (kRopeFirst=true), and V3.2 drops the Hadamard rotation. The
CUDA lives in csrc/deepseek_v32/indexer_k.cuh (separate from the shared V4
main_norm_rope.cuh so V3.2 edits don't touch V4 code).
"""

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

from .utils import make_name

_CUDA_FILE = "deepseek_v32/indexer_k.cuh"


@cache_once
def _jit_k_indexer_norm_rope_module(dtype: torch.dtype, hadamard: bool):
    """K kernel -> bf16 (no store). kRopeFirst=true; hadamard toggles the rotation."""
    args = make_cpp_args(dtype, is_arch_support_pdl(), True, hadamard)
    return load_jit(
        make_name(f"k_indexer_norm_rope_h{int(hadamard)}"),
        *args,
        cuda_files=[_CUDA_FILE],
        cuda_wrappers=[
            ("forward", f"FusedKIndexerNormRopeHadamardKernel<{args}>::forward"),
        ],
    )


@cache_once
def _jit_k_indexer_norm_rope_store_module(
    dtype: torch.dtype, page_size: int, hadamard: bool
):
    """K kernel + fused store: LayerNorm + RoPE (+ optional Hadamard) + fp8 quant
    + paged index-k cache write, in one launch."""
    args = make_cpp_args(dtype, is_arch_support_pdl(), True, hadamard, page_size)
    return load_jit(
        make_name(f"k_indexer_norm_rope_store_p{page_size}_h{int(hadamard)}"),
        *args,
        cuda_files=[_CUDA_FILE],
        cuda_wrappers=[
            ("forward", f"FusedKIndexerNormRopeStoreKernel<{args}>::forward"),
        ],
    )


def fused_k_indexer_norm_rope_first_hadamard(
    k_input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    hadamard: bool = True,
) -> torch.Tensor:
    """V3.2 indexer K: LayerNorm + RoPE on leading dims (+ optional Hadamard) -> bf16. CUDA only."""
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)
    # k_input may be a non-contiguous wk slice; output is always contiguous.
    k_out = torch.empty(k_input.shape, dtype=k_input.dtype, device=k_input.device)
    module = _jit_k_indexer_norm_rope_module(k_input.dtype, hadamard)
    module.forward(
        k_input,
        k_out,
        weight,
        bias,
        freqs_real,
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
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    page_size: int,
    hadamard: bool = True,
) -> None:
    """V3.2 indexer K + fused store: LayerNorm + RoPE on leading dims (+ optional
    Hadamard) + fp8 act-quant + paged index-k cache write, in one launch. CUDA only."""
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)
    module = _jit_k_indexer_norm_rope_store_module(k_input.dtype, page_size, hadamard)
    module.forward(
        k_input,
        cache,
        out_cache_loc,
        weight,
        bias,
        freqs_real,
        positions,
        float(eps),
    )
