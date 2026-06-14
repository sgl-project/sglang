"""Fused linear + tanh-GELU via the cublasLt GELU epilogue.

Many diffusion DiT FeedForwards compute ``gelu(linear(x))`` as a standalone GEMM
followed by a separate, bandwidth-bound GELU kernel over the ``[tokens, 4*dim]``
MLP intermediate. ``torch._addmm_activation`` folds the bias-add and GELU into
the GEMM epilogue (cublasLt), removing the extra kernel launch and the
intermediate HBM round-trip. cublasLt's GELU matches the tanh-approximate GELU
to within bf16/fp16 rounding, so the fused path is numerically equivalent for
half-precision inference.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.linear import UnquantizedLinearMethod

# ``torch._addmm_activation`` is the (private but stable) entry point to the
# cublasLt GEMM+bias+activation epilogue. Guard for builds where it is absent so
# the reference path is always available.
_HAS_ADDMM_ACTIVATION = hasattr(torch, "_addmm_activation")


def can_fuse_linear_gelu(linear: Any, x: torch.Tensor) -> bool:
    """Whether ``gelu(linear(x))`` can use the fused cublasLt epilogue.

    Requires the fused API to exist, a CUDA half-precision input, and an
    unquantized, bias'd, non-output-gathering linear layer (so the local weight
    shard is exactly what the reference forward multiplies — correct under TP).
    """
    if not (_HAS_ADDMM_ACTIVATION and x.is_cuda):
        return False
    # Plain nn.Linear has no quant_method (None); sglang linears must be
    # unquantized. Reject any real quantization method.
    quant_method = getattr(linear, "quant_method", None)
    if quant_method is not None and not isinstance(
        quant_method, UnquantizedLinearMethod
    ):
        return False
    if getattr(linear, "skip_bias_add", False):
        return False
    # A column-parallel layer that gathers its output across TP ranks would have
    # the cross-rank gather skipped by the per-shard fused path. Allow it only
    # when the gather is a no-op (single TP rank); otherwise fall back.
    if getattr(linear, "gather_output", False) and getattr(linear, "tp_size", 1) > 1:
        return False
    weight = getattr(linear, "weight", None)
    bias = getattr(linear, "bias", None)
    if weight is None or bias is None or weight.dim() != 2:
        return False
    return weight.dtype in (torch.bfloat16, torch.float16)


def linear_gelu_tanh(linear: Any, x: torch.Tensor) -> torch.Tensor:
    """Return tanh-approximate ``gelu(linear(x))``.

    Uses the fused cublasLt GELU epilogue when :func:`can_fuse_linear_gelu`
    holds; otherwise falls back to the exact reference ``linear(x)`` followed by
    ``F.gelu(approximate="tanh")``.
    """
    if can_fuse_linear_gelu(linear, x):
        x2d = x.reshape(-1, x.shape[-1])
        out = torch._addmm_activation(
            linear.bias, x2d, linear.weight.t(), use_gelu=True
        )
        return out.view(*x.shape[:-1], out.shape[-1])
    out = linear(x)
    if isinstance(out, tuple):
        out = out[0]
    return F.gelu(out, approximate="tanh")


class FusedTanhGELU(nn.Module):
    """Drop-in replacement for the diffusers ``GELU(approximate="tanh")`` proj+act.

    Holds the same ``proj`` linear (identical checkpoint keys) but fuses the
    up-projection GEMM with the tanh-GELU via :func:`linear_gelu_tanh`.
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return linear_gelu_tanh(self.proj, hidden_states)
