# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Beam sandwich RMSNorm, depth-scaled residual, and next RMSNorm in one launch.

The post-norm and combined stream round to the activation dtype. FP32 combine
rounds each product before adding; activation-dtype combine also rounds both
products to the activation dtype. Norm reduction differences can be amplified
by residual cancellation, so the result is not guaranteed within one ULP.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

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
    HAS_SANDWICH: tl.constexpr,
    HAS_OUT_NORM: tl.constexpr,
    COMBINE_IN_FP32: tl.constexpr,
):
    row = tl.program_id(axis=0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim
    act_dtype = x_ptr.dtype.element_ty

    x = tl.load(x_ptr + row * x_stride_row + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    if HAS_SANDWICH:
        w = tl.load(sandwich_weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(x * x, axis=0) / hidden_dim + eps)
        # The standalone RMSNorm writes the activation dtype before the combine.
        x = (x * rstd * w).to(act_dtype).to(tl.float32)

    residual = tl.load(
        residual_ptr + row * residual_stride_row + offsets, mask=mask, other=0.0
    )
    # `depth_scaled_residual` casts the residual to the branch dtype first.
    residual = residual.to(act_dtype).to(tl.float32)
    if COMBINE_IN_FP32:
        # Explicit rounding also prevents ptxas from fusing packed FP32 ops.
        xs = libdevice.mul_rn(x, x_scale)
        rs = libdevice.mul_rn(residual, residual_scale)
        combined = libdevice.add_rn(xs, rs).to(act_dtype)
    else:
        xs = (x * x_scale).to(act_dtype).to(tl.float32)
        rs = (residual * residual_scale).to(act_dtype).to(tl.float32)
        combined = (xs + rs).to(act_dtype)
    out_offsets = row * hidden_dim + offsets
    tl.store(combined_ptr + out_offsets, combined, mask=mask)

    if HAS_OUT_NORM:
        c = combined.to(tl.float32)
        w2 = tl.load(out_weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        rstd2 = tl.rsqrt(tl.sum(c * c, axis=0) / hidden_dim + eps)
        tl.store(out_ptr + out_offsets, (c * rstd2 * w2).to(act_dtype), mask=mask)


def fused_sandwich_residual_rmsnorm(
    branch: torch.Tensor,
    residual: torch.Tensor,
    sandwich_weight: Optional[torch.Tensor],
    out_weight: Optional[torch.Tensor],
    eps: float,
    x_scale: float,
    residual_scale: float,
    combine_in_fp32: bool,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Return `(next_norm(combined), combined)`; the first is None without `out_weight`.

    `branch` and `residual` are 2-D `[tokens, hidden]` in the activation dtype.
    `sandwich_weight` is the post-norm weight applied to `branch` before the
    combine (None skips it); `out_weight` is the following norm's weight (None
    returns only the combined stream, for the last layer of a non-final
    pipeline stage).
    """
    if branch.ndim != 2 or residual.shape != branch.shape:
        raise ValueError(
            f"expected matching 2-D tensors; got {tuple(branch.shape)} and {tuple(residual.shape)}"
        )
    if branch.stride(1) != 1 or residual.stride(1) != 1:
        raise ValueError("branch and residual must be contiguous along hidden")
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
        combine_in_fp32,
        out,
        combined,
    )
    return out, combined


@register_custom_op(mutates_args=["out", "combined"])
def _fused_sandwich_residual_rmsnorm(
    branch: torch.Tensor,
    residual: torch.Tensor,
    sandwich_weight: Optional[torch.Tensor],
    out_weight: Optional[torch.Tensor],
    eps: float,
    x_scale: float,
    residual_scale: float,
    combine_in_fp32: bool,
    out: Optional[torch.Tensor],
    combined: torch.Tensor,
) -> None:
    hidden_dim = branch.shape[1]
    rows = branch.shape[0]
    if rows == 0:
        return
    dummy = sandwich_weight if sandwich_weight is not None else out_weight
    if dummy is None:
        dummy = branch  # never dereferenced: both norm flags are off
    _fused_sandwich_residual_rmsnorm_kernel[(rows,)](
        out if out is not None else combined,
        combined,
        branch,
        residual,
        branch.stride(0),
        residual.stride(0),
        sandwich_weight if sandwich_weight is not None else dummy,
        out_weight if out_weight is not None else dummy,
        eps=eps,
        x_scale=x_scale,
        residual_scale=residual_scale,
        hidden_dim=hidden_dim,
        BLOCK_SIZE=triton.next_power_of_2(hidden_dim),
        HAS_SANDWICH=sandwich_weight is not None,
        HAS_OUT_NORM=out_weight is not None,
        COMBINE_IN_FP32=combine_in_fp32,
        num_warps=8,
        enable_fp_fusion=False,
    )
