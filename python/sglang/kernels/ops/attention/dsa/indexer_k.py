"""DSA only. Indexer K kernels (JIT)."""

import logging

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

_CUDA_FILE = "deepseek_v32/indexer_k.cuh"
logger = logging.getLogger(__name__)


def _rope_suffix(hadamard: bool, neox_style: bool) -> str:
    return ("_hadamard" if hadamard else "") + ("_neox" if neox_style else "")


@cache_once
def _jit_k_indexer_norm_rope_module(
    dtype: torch.dtype, hadamard: bool, neox_style: bool
):
    args = make_cpp_args(dtype, is_arch_support_pdl(), hadamard, neox_style)
    return load_jit(
        "dpsk_v32_k_indexer_norm_rope" + _rope_suffix(hadamard, neox_style),
        *args,
        cuda_files=[_CUDA_FILE],
        cuda_wrappers=[
            ("forward", f"FusedKIndexerNormRopeKernel<{args}>::forward"),
        ],
    )


@cache_once
def _jit_k_indexer_norm_rope_store_module(
    dtype: torch.dtype, page_size: int, hadamard: bool, neox_style: bool
):
    args = make_cpp_args(dtype, is_arch_support_pdl(), page_size, hadamard, neox_style)
    return load_jit(
        f"dpsk_v32_k_indexer_norm_rope_store_p{page_size}"
        + _rope_suffix(hadamard, neox_style),
        *args,
        cuda_files=[_CUDA_FILE],
        cuda_wrappers=[
            ("forward", f"FusedKIndexerNormRopeStoreKernel<{args}>::forward"),
        ],
    )


@cache_once
def _jit_k_indexer_norm_store_module(page_size: int):
    args = make_cpp_args(
        torch.bfloat16, is_arch_support_pdl(), page_size, False, False, False
    )
    return load_jit(
        f"dsa_k_indexer_norm_store_p{page_size}",
        *args,
        cuda_files=[_CUDA_FILE],
        cuda_wrappers=[
            ("forward", f"FusedKIndexerNormRopeStoreKernel<{args}>::forward_no_rope"),
        ],
    )


@cache_once
def can_use_fused_k_indexer_norm_store(page_size: int, device: torch.device) -> bool:
    try:
        from flashinfer import norm

        # The no-RoPE specialization matches the CuTe LayerNorm reduction.
        # An alternate or unknown FlashInfer backend keeps its original path.
        use_cuda_norm = getattr(norm, "_use_cuda_norm", None)
        if use_cuda_norm is None or use_cuda_norm(device):
            return False
        _jit_k_indexer_norm_store_module(page_size)
        return True
    except Exception as exc:
        logger.warning("Failed to load fused index-K norm/store JIT kernel: %s", exc)
        return False


def fused_k_indexer_norm_store(
    k_input: torch.Tensor,
    cache: torch.Tensor,
    out_cache_loc: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    page_size: int,
    *,
    dcp_size: int = 1,
    dcp_rank: int = 0,
) -> None:
    """LayerNorm, BF16 rounding, FP8 quantization and index-K store without RoPE.

    Accepts the strided [tokens, 128] projection slice. The normal FP32
    per-token scales and paged cache layout match fused_store_index_k_cache.
    When sharded by DCP, write only this rank's widened slots.
    """
    module = _jit_k_indexer_norm_store_module(page_size)
    module.forward(
        k_input,
        cache,
        out_cache_loc.contiguous(),
        weight,
        bias,
        float(eps),
        dcp_size,
        dcp_rank,
    )


def fused_k_indexer_norm_rope(
    k_input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    hadamard: bool = False,
    neox_style: bool = False,
) -> torch.Tensor:
    """Indexer K: LayerNorm + RoPE on leading dims (+ 128-pt Hadamard) -> bf16.
    CUDA only."""
    # k_input may be a non-contiguous wk slice; output is always contiguous.
    k_out = torch.empty(k_input.shape, dtype=k_input.dtype, device=k_input.device)
    module = _jit_k_indexer_norm_rope_module(k_input.dtype, hadamard, neox_style)
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
    hadamard: bool = False,
    neox_style: bool = False,
) -> None:
    """Indexer K + fused store: LayerNorm + RoPE on leading dims (+ Hadamard) +
    fp8 act-quant + paged index-k cache write, in one launch. CUDA only.

    ``cache`` is (num_pages, 132 * page_size) uint8 with one row per page; its
    rows may be strided, and the kernel follows ``cache.stride(0)``."""
    if not out_cache_loc.is_contiguous():
        out_cache_loc = out_cache_loc.contiguous()
    module = _jit_k_indexer_norm_rope_store_module(
        k_input.dtype, page_size, hadamard, neox_style
    )
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
