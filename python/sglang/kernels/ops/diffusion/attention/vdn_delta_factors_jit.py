"""Fused VDN-H3 delta-rule factors: (I + A)^-1 folded into the transition and injection.

One CUDA kernel (block Gauss-Jordan inverse in registers + the two products) replaces the
cholesky / solve_triangular / GEMM chain of ``delta_factor_apply`` for the ``vdn_solve`` and
``vdn_scaled`` rules.  fp32, head_dim 128 only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

HEAD_DIM = 128


@cache_once
def _jit_vdn_delta_factors_module() -> Module:
    if torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError(
            "vdn_delta_factors needs SM80 or later (2 x 70 KB shared memory per SM)"
        )
    return load_jit(
        "diffusion_vdn_delta_factors",
        cuda_files=["diffusion/vdn_delta_factors.cuh"],
        cuda_wrappers=[
            ("vdn_delta_factors", "vdn_delta_factors::VdnDeltaFactorsKernel::run")
        ],
    )


def _fake_impl(
    A: torch.Tensor, B: torch.Tensor, alpha: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    del alpha
    return torch.empty_like(A), torch.empty_like(B)


@register_custom_op(
    op_name="diffusion_vdn_delta_factors",
    mutates_args=[],
    fake_impl=_fake_impl,
)
def vdn_delta_factors(
    A: torch.Tensor, B: torch.Tensor, alpha: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(alpha[..., :, None] * inv(I + A), B @ inv(I + A))`` for fp32 ``[..., 128, 128]`` SPD ``A``.

    ``B`` has the shape of ``A``; ``alpha`` is ``[..., 128]``.  Same accuracy as the eager
    cholesky path (both are dominated by cond(I + A) in fp32).
    """
    transition = torch.empty_like(A)
    injection = torch.empty_like(B)
    module = _jit_vdn_delta_factors_module()
    module.vdn_delta_factors(
        transition.view(-1, HEAD_DIM, HEAD_DIM),
        injection.view(-1, HEAD_DIM, HEAD_DIM),
        A.view(-1, HEAD_DIM, HEAD_DIM),
        B.view(-1, HEAD_DIM, HEAD_DIM),
        alpha.view(-1, HEAD_DIM),
    )
    return transition, injection


def can_use_vdn_delta_factors(
    A: torch.Tensor, B: torch.Tensor, alpha: torch.Tensor
) -> bool:
    return (
        A.is_cuda
        and A.dtype is torch.float32
        and B.dtype is torch.float32
        and alpha.dtype is torch.float32
        and A.device == B.device == alpha.device
        and A.dim() >= 2
        and A.shape[-1] == HEAD_DIM
        and A.shape[-2] == HEAD_DIM
        and B.shape == A.shape
        and alpha.shape == A.shape[:-1]
        and A.is_contiguous()
        and B.is_contiguous()
        and alpha.is_contiguous()
        and torch.cuda.get_device_capability(A.device)[0] >= 8
    )


__all__ = [
    "can_use_vdn_delta_factors",
    "vdn_delta_factors",
]
