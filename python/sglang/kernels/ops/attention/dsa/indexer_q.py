"""DSA only. Indexer Q kernels (JIT)."""

import logging
from typing import Tuple

import torch

from sglang.kernels.arch import current_arch
from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.ops.attention.dsv4.utils import make_name

logger = logging.getLogger(__name__)


@cache_once
def _jit_kpool_q_hadamard_quant_module():
    if not current_arch().is_sm(12):
        raise RuntimeError("KPool Q Hadamard quantization requires SM12x")
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "kpool_q_hadamard_quant",
        *args,
        cuda_files=["dsa/kpool_q_hadamard_quant.cuh"],
        cuda_wrappers=[
            ("forward", f"KPoolQHadamardQuantKernel<{args}>::forward"),
        ],
    )


@cache_once
def can_use_fused_kpool_q_hadamard_quant() -> bool:
    if not current_arch().is_sm(12):
        return False
    try:
        _jit_kpool_q_hadamard_quant_module()
        return True
    except Exception as exc:
        logger.warning("Failed to load fused KPool Q Hadamard quantization: %s", exc)
        return False


def fused_kpool_q_hadamard_quant(
    query: torch.Tensor, head_gate: torch.Tensor, softmax_scale: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """SM12x KPool indexer: Hadamard + UE8M0 FP8 quant + FP32 gate scaling.

    The contiguous BF16 query has shape [tokens, heads, 128]; the contiguous
    FP32 gate [tokens, heads] already includes the head-count scale.
    """
    q_fp8 = torch.empty_like(query, dtype=torch.float8_e4m3fn)
    weights = head_gate.new_empty(*head_gate.shape, 1)
    _jit_kpool_q_hadamard_quant_module().forward(
        query, head_gate, q_fp8, weights, float(softmax_scale)
    )
    return q_fp8, weights


# Indexer q is laid out [rope | nope] (V4 is [nope | rope]) -> kRopeFirst=true.
@cache_once
def _jit_main_q_indexer_rope_first_quant_module(
    dtype: torch.dtype, hadamard: bool, neox_style: bool
):
    args = make_cpp_args(dtype, is_arch_support_pdl(), True, hadamard, neox_style)
    return load_jit(
        make_name(
            "main_q_indexer_rope_first_quant"
            + ("_hadamard" if hadamard else "")
            + ("_neox" if neox_style else "")
        ),
        *args,
        cuda_files=["deepseek_v4/main_norm_rope.cuh"],
        cuda_wrappers=[
            ("forward", f"FusedQIndexerRopeHadamardQuantKernel<{args}>::forward"),
        ],
    )


def fused_q_indexer_rope_first_quant(
    q_input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: float,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    hadamard: bool = False,
    neox_style: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Indexer Q: RoPE on the leading dims (+ 128-pt Hadamard) + fp8 act-quant.
    CUDA only."""
    q_fp8 = torch.empty(q_input.shape, dtype=torch.float8_e4m3fn, device=q_input.device)
    weights_out = torch.empty(
        (*q_input.shape[:-1], 1), dtype=torch.float32, device=q_input.device
    )
    module = _jit_main_q_indexer_rope_first_quant_module(
        q_input.dtype, hadamard, neox_style
    )
    module.forward(
        q_input,
        q_fp8,
        weight,
        weights_out,
        float(weight_scale),
        cos_sin_cache,
        positions,
    )
    return q_fp8, weights_out
