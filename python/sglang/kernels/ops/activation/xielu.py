"""xIELU activation (https://arxiv.org/abs/2411.13010), used by Apertus."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_xielu_module(dtype: torch.dtype) -> Module:
    if dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(f"xielu supports float16 and bfloat16, got {dtype}")
    args = make_cpp_args(dtype, is_arch_support_pdl())
    return load_jit(
        "xielu",
        *args,
        cuda_files=["elementwise/xielu.cuh"],
        cuda_wrappers=[("xielu", f"XIELUKernel<{args}>::run")],
    )


@register_custom_op(mutates_args=["out"])
def _xielu_inplace(
    input: torch.Tensor,
    out: torch.Tensor,
    alpha_p: float,
    alpha_n: float,
    beta: float,
    eps: float,
) -> None:
    _jit_xielu_module(input.dtype).xielu(
        input.view(-1), out.view(-1), alpha_p, alpha_n, beta, eps
    )


def xielu(
    input: torch.Tensor,
    alpha_p: float,
    alpha_n: float,
    beta: float,
    eps: float,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Elementwise xIELU on a contiguous CUDA tensor.

    ``alpha_p`` and ``alpha_n`` are the effective coefficients, after the
    softplus: ``softplus(alpha_p)`` and ``beta + softplus(alpha_n)``. The
    expression is evaluated in fp32 and rounded once, bit-identical to
    :func:`xielu_reference`.
    """
    if out is None:
        out = torch.empty_like(input)
    _xielu_inplace(input, out, alpha_p, alpha_n, beta, eps)
    return out


def xielu_reference(
    x: torch.Tensor, alpha_p: float, alpha_n: float, beta: float, eps: float
) -> torch.Tensor:
    """The fp32 torch expression the kernel matches, rounded once to ``x.dtype``."""
    xf = x.float()
    pos = alpha_p * xf * xf + beta * xf
    neg = (torch.expm1(torch.clamp_max(xf, eps)) - xf) * alpha_n + beta * xf
    return torch.where(xf > 0, pos, neg).to(x.dtype)
