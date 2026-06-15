"""Fused linear + tanh-GELU kernels via the cublasLt GELU epilogue.

Many diffusion DiT FeedForwards compute ``gelu(linear(x))`` as a standalone GEMM
followed by a separate, bandwidth-bound GELU kernel over the ``[tokens, 4*dim]``
MLP intermediate. ``torch._addmm_activation`` folds the bias-add and GELU into
the GEMM epilogue (cublasLt), removing the extra kernel launch and the
intermediate HBM round-trip. cublasLt's GELU matches the tanh-approximate GELU
to within bf16/fp16 rounding (max abs diff ~5e-6 in fp32), so the fused path is
numerically equivalent for half-precision inference.

The fused GEMM is exposed as a registered custom op (``register_custom_op``)
exactly like the other diffusion jit_kernels (e.g. qknorm_rope), so it stays a
single opaque op under ``torch.compile`` -- no graph break, no fallback to the
unfused path.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.linear import UnquantizedLinearMethod
from sglang.srt.utils.custom_op import register_custom_op

# ``torch._addmm_activation`` is the (private but stable) entry point to the
# cublasLt GEMM+bias+activation epilogue. Guard for builds where it is absent so
# the reference path is always available.
_HAS_ADDMM_ACTIVATION = hasattr(torch, "_addmm_activation")


def _fused_linear_gelu_tanh_fake(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


@register_custom_op(
    op_name="diffusion_fused_linear_gelu_tanh",
    mutates_args=[],
    fake_impl=_fused_linear_gelu_tanh_fake,
)
def fused_linear_gelu_tanh(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """``gelu_tanh(x @ weight.T + bias)`` fused in the cublasLt GELU epilogue.

    ``weight`` is ``[out, in]`` (nn.Linear / sglang linear layout). Registered as
    a custom op so it is opaque under torch.compile.
    """
    x2d = x.reshape(-1, x.shape[-1])
    out = torch._addmm_activation(bias, x2d, weight.t(), use_gelu=True)
    return out.view(*x.shape[:-1], weight.shape[0])


def can_fuse_linear_gelu(linear: Any, x: torch.Tensor) -> bool:
    """Whether ``gelu(linear(x))`` can use the fused cublasLt epilogue.

    Requires the fused API to exist, a CUDA half-precision input, and an
    unquantized, bias'd, non-output-gathering linear layer (so the local weight
    shard is exactly what the reference forward multiplies -- correct under TP).
    A column-parallel layer that gathers across multiple ranks is excluded
    (the per-shard fused op would skip the cross-rank gather); a single-rank
    no-op gather is fine.
    """
    if not (_HAS_ADDMM_ACTIVATION and x.is_cuda):
        return False
    if x.dtype not in (torch.bfloat16, torch.float16):
        return False
    if getattr(linear, "_sgl_disable_fused_linear_gelu", False):
        return False
    # Quantized checkpoints can leave selected layers unquantized via their
    # exclude list. Keep those layers on the reference path because the fused
    # epilogue can move strict image-consistency metrics in mixed-precision runs.
    if getattr(linear, "quant_config", None) is not None:
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
    if getattr(linear, "gather_output", False) and getattr(linear, "tp_size", 1) > 1:
        return False
    weight = getattr(linear, "weight", None)
    bias = getattr(linear, "bias", None)
    if weight is None or bias is None or weight.dim() != 2:
        return False
    if weight.dtype != x.dtype or bias.dtype != x.dtype:
        return False
    return weight.dtype in (torch.bfloat16, torch.float16)


def linear_gelu_tanh(linear: Any, x: torch.Tensor) -> torch.Tensor:
    """Return tanh-approximate ``gelu(linear(x))``.

    Uses the fused cublasLt GELU epilogue (as a registered custom op, so it is
    compile-safe) when :func:`can_fuse_linear_gelu` holds; otherwise falls back
    to the exact reference ``linear(x)`` + ``F.gelu(approximate="tanh")``.
    """
    if can_fuse_linear_gelu(linear, x):
        return fused_linear_gelu_tanh(x, linear.weight, linear.bias)
    out = linear(x)
    if isinstance(out, tuple):
        out = out[0]
    return F.gelu(out, approximate="tanh")


class FusedTanhGELU(nn.Module):
    """Drop-in replacement for the diffusers ``GELU(approximate="tanh")`` proj+act.

    Holds the same ``proj`` linear (identical checkpoint keys) but fuses the
    up-projection GEMM with the tanh-GELU via :func:`linear_gelu_tanh`.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        bias: bool = True,
        disable_fused: bool = False,
    ):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)
        self.proj._sgl_disable_fused_linear_gelu = disable_fused

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return linear_gelu_tanh(self.proj, hidden_states)
