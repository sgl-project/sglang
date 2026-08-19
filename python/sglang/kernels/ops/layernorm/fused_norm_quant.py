"""Fused (residual-add +) RMSNorm + dynamic per-token FP8 quantization.

One launch produces the normalized activation in fp8 plus its per-token scale,
so an FP8 linear with dynamic per-token activation scaling can skip its own
``per_token_quant_fp8`` launch. The residual stream is updated in place; the
normalized value is never materialized in the activation dtype.

Limited to ``hidden_size <= 8192`` (one thread per 16B vector, one CTA per
token), so models above that -- e.g. Llama-3.1-405B-FP8-dynamic and
Nemotron-Ultra-253B-FP8-dynamic, both hidden 16384 -- fall back to the unfused
path.

This is the *dynamic per-token* sibling of the existing quant fusions:

===========================================  ==================  ============
path                                         scale granularity   device
===========================================  ==================  ============
flashinfer ``fused_add_rmsnorm_quant``       static per-tensor   CUDA
aiter fused AR+RMSNorm+quant                 per-group 1x128     ROCm
**this kernel**                              dynamic per-token   CUDA
===========================================  ==================  ============

Originally authored by @narutolhy in sgl-project/sglang#22005.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Tuple

import torch

from sglang.kernel_api_logging import debug_kernel_api
from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)


# 16B vectors / 8 elements per thread, one CTA per token, <=1024 threads.
_ELEMS_PER_VEC = 8
_MAX_HIDDEN_SIZE = 1024 * _ELEMS_PER_VEC


def is_supported_fused_norm_quant_hidden_size(hidden_size: int) -> bool:
    return (
        hidden_size > 0
        and hidden_size % _ELEMS_PER_VEC == 0
        and hidden_size <= _MAX_HIDDEN_SIZE
    )


@cache_once
def _jit_module(dtype: torch.dtype, has_residual: bool, add_weight_one: bool) -> Module:
    args = make_cpp_args(has_residual, add_weight_one, dtype)
    return load_jit(
        "fused_add_rmsnorm_per_token_quant",
        *args,
        cuda_files=["elementwise/fused_add_rmsnorm_per_token_quant.cuh"],
        cuda_wrappers=[
            (
                "fused_add_rmsnorm_per_token_quant",
                f"FusedAddRMSNormPerTokenQuantKernel<{args}>::run",
            )
        ],
    )


@torch.compiler.assume_constant_result
@cache_once
def can_use_fused_norm_per_token_quant(
    hidden_size: int, dtype: torch.dtype, add_weight_one: bool = False
) -> bool:
    """Whether the JIT kernel is loadable for this shape/dtype.

    ``@torch.compiler.assume_constant_result`` keeps the probe out of the
    dynamo graph: the answer only depends on shape/dtype/build, never on tensor
    data, so it is resolved once at trace time instead of becoming a guard.
    """
    if dtype not in (torch.bfloat16, torch.float16):
        return False
    if not is_supported_fused_norm_quant_hidden_size(hidden_size):
        return False
    try:
        _jit_module(dtype, True, add_weight_one)
        _jit_module(dtype, False, add_weight_one)
        return True
    except Exception:
        # Must stay non-fatal (this is a capability probe on an opt-in path),
        # but it must not be silent: swallowing a build error here turns a
        # broken kernel into a green CI run full of skips.
        logger.warning(
            "fused add+rmsnorm+per-token-fp8-quant JIT kernel failed to build "
            "for hidden_size=%d dtype=%s; falling back to the unfused path.",
            hidden_size,
            dtype,
            exc_info=True,
        )
        return False


def _alloc_outputs(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    num_tokens, hidden_size = x.shape
    return (
        torch.empty(
            (num_tokens, hidden_size), dtype=torch.float8_e4m3fn, device=x.device
        ),
        torch.empty((num_tokens, 1), dtype=torch.float32, device=x.device),
    )


@debug_kernel_api
def fused_add_rmsnorm_per_token_quant(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    add_weight_one: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``residual += input``; then normalize and quantize.

    Returns ``(normed_fp8, per_token_scale)``. ``residual`` is updated in
    place and carries the residual stream; the normalized value is emitted
    only in fp8, since every caller feeds it straight into an FP8 linear.
    ``add_weight_one`` selects Gemma semantics (``weight + 1``) without
    materializing the biased weight.
    """
    out_fp8, out_scales = _alloc_outputs(input)
    module = _jit_module(input.dtype, True, add_weight_one)
    module.fused_add_rmsnorm_per_token_quant(
        input, residual, weight, out_fp8, out_scales, eps
    )
    return out_fp8, out_scales


@debug_kernel_api
def fused_rmsnorm_per_token_quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    add_weight_one: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize and quantize without a residual add.

    Returns ``(normed_fp8, per_token_scale)``.
    """
    out_fp8, out_scales = _alloc_outputs(input)
    module = _jit_module(input.dtype, False, add_weight_one)
    # The no-residual specialization ignores the residual view; pass ``input``
    # as a shape-compatible placeholder.
    module.fused_add_rmsnorm_per_token_quant(
        input, input, weight, out_fp8, out_scales, eps
    )
    return out_fp8, out_scales


__all__ = [
    "can_use_fused_norm_per_token_quant",
    "fused_add_rmsnorm_per_token_quant",
    "fused_rmsnorm_per_token_quant",
    "is_supported_fused_norm_quant_hidden_size",
]
