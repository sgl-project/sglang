"""xIELU activation (https://arxiv.org/abs/2411.13010), used by Apertus."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F

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
    input: torch.Tensor, out: torch.Tensor, coeffs: torch.Tensor
) -> None:
    _jit_xielu_module(input.dtype).xielu(input.view(-1), out.view(-1), coeffs)


def xielu(
    input: torch.Tensor, coeffs: torch.Tensor, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Elementwise xIELU on a contiguous float16/bfloat16 CUDA tensor.

    ``coeffs`` is the fp32 tensor from :func:`xielu_coeffs` on the same device.
    The expression is evaluated in fp32 and rounded once, bit-identical to
    :func:`xielu_reference` in bfloat16.
    """
    if out is None:
        out = torch.empty_like(input)
    _xielu_inplace(input, out, coeffs)
    return out


def xielu_coeffs(
    alpha_p: torch.Tensor,
    alpha_n: torch.Tensor,
    beta: torch.Tensor,
    eps: torch.Tensor,
) -> torch.Tensor:
    """Effective coefficients from the checkpoint's pre-softplus parameters, as
    fp32 ``[softplus(alpha_p), beta + softplus(alpha_n), beta, eps]``."""
    beta = beta.float().reshape(1)
    return torch.cat(
        [
            F.softplus(alpha_p.float()).reshape(1),
            beta + F.softplus(alpha_n.float()).reshape(1),
            beta,
            eps.float().reshape(1),
        ]
    )


def xielu_reference(x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """The fp32 torch expression the kernel matches, rounded once to ``x.dtype``."""
    alpha_p, alpha_n, beta, eps = coeffs.to(x.device).unbind()
    xf = x.float()
    pos = alpha_p * xf * xf + beta * xf
    neg = (torch.expm1(torch.clamp_max(xf, eps)) - xf) * alpha_n + beta * xf
    return torch.where(xf > 0, pos, neg).to(x.dtype)
