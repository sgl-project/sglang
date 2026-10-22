# SPDX-License-Identifier: Apache-2.0
"""Beam sandwich RMSNorm, depth-scaled residual, and next RMSNorm in one launch.

The post-norm, both scaled operands, and the combined stream round to the
activation dtype. Norm reduction differences can be amplified by residual
cancellation, so the result is not guaranteed within one ULP.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.fused_op import BaseFusedOp
from sglang.kernels.spec import CapabilityRequirement, KernelBackend
from sglang.srt.utils.custom_op import register_custom_op


@triton.jit
def _fused_sandwich_residual_rmsnorm_kernel(
    out_ptr,
    combined_ptr,
    x_ptr,
    residual_ptr,
    x_stride_row,
    residual_stride_row,
    sandwich_weight_ptr,
    out_weight_ptr,
    eps,
    x_scale,
    residual_scale,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_OUT_NORM: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    row = tl.program_id(axis=0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim
    act_dtype = x_ptr.dtype.element_ty

    # The preceding kernel does not write the weights; load them before the PDL wait.
    w = tl.load(sandwich_weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    if HAS_OUT_NORM:
        w2 = tl.load(out_weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    x = tl.load(x_ptr + row * x_stride_row + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / hidden_dim + eps)
    # The standalone RMSNorm writes the activation dtype before the combine.
    x = (x * rstd * w).to(act_dtype).to(tl.float32)

    residual = tl.load(
        residual_ptr + row * residual_stride_row + offsets, mask=mask, other=0.0
    )
    # `depth_scaled_residual` casts the residual to the branch dtype first.
    residual = residual.to(act_dtype).to(tl.float32)
    xs = (x * x_scale).to(act_dtype).to(tl.float32)
    rs = (residual * residual_scale).to(act_dtype).to(tl.float32)
    combined = (xs + rs).to(act_dtype)
    out_offsets = row * hidden_dim + offsets
    tl.store(combined_ptr + out_offsets, combined, mask=mask)

    if HAS_OUT_NORM:
        c = combined.to(tl.float32)
        rstd2 = tl.rsqrt(tl.sum(c * c, axis=0) / hidden_dim + eps)
        tl.store(out_ptr + out_offsets, (c * rstd2 * w2).to(act_dtype), mask=mask)


def fused_sandwich_residual_rmsnorm(
    branch: torch.Tensor,
    residual: torch.Tensor,
    sandwich_weight: torch.Tensor,
    out_weight: Optional[torch.Tensor],
    eps: float,
    x_scale: float,
    residual_scale: float,
    *,
    enable_pdl: bool = False,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Return `(next_norm(combined), combined)`; the first is None without `out_weight`.

    `branch` and `residual` are 2-D `[tokens, hidden]` in the activation dtype.
    `sandwich_weight` is the post-norm weight applied to `branch` before the
    combine; `out_weight` is the following norm's weight (None returns only the
    combined stream, for the last layer of a non-final pipeline stage).
    `enable_pdl` launches with programmatic dependent launch; the caller must
    check that the device supports it.
    """
    if branch.ndim != 2 or residual.shape != branch.shape:
        raise ValueError(
            f"expected matching 2-D tensors; got {tuple(branch.shape)} and {tuple(residual.shape)}"
        )
    if branch.stride(1) != 1 or residual.stride(1) != 1:
        raise ValueError("branch and residual must be contiguous along hidden")
    if branch.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("branch must be fp16, bf16 or fp32")
    hidden_dim = branch.shape[1]
    for w in (sandwich_weight, out_weight):
        if w is not None and (w.shape != (hidden_dim,) or not w.is_contiguous()):
            raise ValueError("norm weights must be contiguous [hidden]")
    combined = torch.empty_like(branch, memory_format=torch.contiguous_format)
    out = torch.empty_like(combined) if out_weight is not None else None
    _fused_sandwich_residual_rmsnorm(
        branch,
        residual,
        sandwich_weight,
        out_weight,
        eps,
        x_scale,
        residual_scale,
        out,
        combined,
        enable_pdl,
    )
    return out, combined


@register_custom_op(mutates_args=["out", "combined"])
def _fused_sandwich_residual_rmsnorm(
    branch: torch.Tensor,
    residual: torch.Tensor,
    sandwich_weight: torch.Tensor,
    out_weight: Optional[torch.Tensor],
    eps: float,
    x_scale: float,
    residual_scale: float,
    out: Optional[torch.Tensor],
    combined: torch.Tensor,
    enable_pdl: bool,
) -> None:
    hidden_dim = branch.shape[1]
    rows = branch.shape[0]
    if rows == 0:
        return
    _fused_sandwich_residual_rmsnorm_kernel[(rows,)](
        out if out is not None else combined,
        combined,
        branch,
        residual,
        branch.stride(0),
        residual.stride(0),
        sandwich_weight,
        out_weight if out_weight is not None else sandwich_weight,
        eps=eps,
        x_scale=x_scale,
        residual_scale=residual_scale,
        hidden_dim=hidden_dim,
        BLOCK_SIZE=triton.next_power_of_2(hidden_dim),
        HAS_OUT_NORM=out_weight is not None,
        USE_PDL=enable_pdl,
        num_warps=8,
        enable_fp_fusion=False,
        **({"launch_pdl": True} if enable_pdl else {}),
    )


def _rmsnorm_native(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Pure-torch RMSNorm reference: ``(x / RMS(x)) * weight`` in FP32."""
    y = x.to(torch.float32)
    y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (y * weight).to(x.dtype)


class BeamSandwichResidualRMSNormOp(BaseFusedOp):
    """Post-normalize a branch, combine residuals, then normalize the next input.

    Returns ``(normed, combined)``; ``normed`` is None without ``out_weight``
    (a non-final pipeline stage).
    """

    op = "layernorm.beam_sandwich_residual_rmsnorm"
    capabilities = {KernelBackend.TRITON: {CapabilityRequirement.CUDA}}

    def _torch_compile_forward(self, num_tokens: int):
        # Native compilation can remove the required intermediate rounding.
        return None

    def forward_torch_compile(self, *args, **kwargs):
        if self._compiled_native is None:
            # Without emulated casts, Inductor drops eager's intermediate rounding.
            self._compiled_native = torch.compile(
                self.forward_native, options={"emulate_precision_casts": True}
            )
        return self._compiled_native(*args, **kwargs)

    def forward_native(
        self,
        branch: torch.Tensor,
        residual: torch.Tensor,
        sandwich_weight: torch.Tensor,
        out_weight: Optional[torch.Tensor],
        eps: float,
        scales: tuple[float, float],
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        branch = _rmsnorm_native(branch, sandwich_weight, eps)
        residual = residual.to(branch.dtype)
        combined = branch * scales[0] + residual * scales[1]
        out = (
            _rmsnorm_native(combined, out_weight, eps)
            if out_weight is not None
            else None
        )
        return out, combined

    def forward_triton(
        self,
        branch: torch.Tensor,
        residual: torch.Tensor,
        sandwich_weight: torch.Tensor,
        out_weight: Optional[torch.Tensor],
        eps: float,
        scales: tuple[float, float],
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        from sglang.kernels.jit.utils import is_arch_support_pdl

        return fused_sandwich_residual_rmsnorm(
            branch,
            residual,
            sandwich_weight,
            out_weight,
            eps,
            *scales,
            enable_pdl=is_arch_support_pdl(),
        )
