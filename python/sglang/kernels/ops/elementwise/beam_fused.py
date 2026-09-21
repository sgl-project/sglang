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
"""Fused Triton kernels for Beam's sandwich-norm residual flow and output gate.

Beam's decoder layer runs, twice per layer, the eager sequence

    t        = post_norm(branch)                       # RMSNorm, bf16 out
    combined = depth_scaled_residual(t, residual)      # bf16 arithmetic
    out      = next_norm(combined)                     # RMSNorm, bf16 out

which costs eight kernel launches per transition (two RMSNorms and six
element-wise ops). `fused_sandwich_residual_rmsnorm` does it in one launch and
reproduces the eager rounding boundaries: the sandwich norm rounds to the
activation dtype before the combine, each scaled operand rounds before the add
(or, with `combine_in_fp32`, the fp32 sum rounds once), and the combined stream
is stored in the activation dtype. Only the RMS reduction order differs from the
standalone norm kernel.

`fused_attention_gate` replaces the three element-wise ops of
`sigmoid(gate + bias) * attention_output`, keeping the bf16 rounding of the
biased gate and of the sigmoid before the product.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


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
    eps: tl.constexpr,
    x_scale: tl.constexpr,
    residual_scale: tl.constexpr,
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

    x = tl.load(x_ptr + row * x_stride_row + offsets, mask=mask, other=0.0).to(tl.float32)
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
        combined = (x * x_scale + residual * residual_scale).to(act_dtype)
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
    scales: Tuple[float, float],
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
    combined = torch.empty_like(branch)
    out = torch.empty_like(branch) if out_weight is not None else None
    rows = branch.shape[0]
    if rows == 0:
        return out, combined
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
        x_scale=scales[0],
        residual_scale=scales[1],
        hidden_dim=hidden_dim,
        BLOCK_SIZE=triton.next_power_of_2(hidden_dim),
        HAS_SANDWICH=sandwich_weight is not None,
        HAS_OUT_NORM=out_weight is not None,
        COMBINE_IN_FP32=combine_in_fp32,
        num_warps=8,
    )
    return out, combined


@triton.jit
def _fused_attention_gate_kernel(
    out_ptr,
    attn_ptr,
    gate_ptr,
    bias_ptr,
    hidden_dim: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1)
    offsets = block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offsets < hidden_dim
    head = offsets // HEAD_DIM
    act_dtype = attn_ptr.dtype.element_ty

    off = row * hidden_dim + offsets
    attn = tl.load(attn_ptr + off, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(gate_ptr + off, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(bias_ptr + head, mask=mask, other=0.0).to(tl.float32)
    # Eager: (gate + bias) rounds to bf16, sigmoid rounds to bf16, product rounds.
    g = (g + b).to(act_dtype).to(tl.float32)
    m = tl.sigmoid(g).to(act_dtype).to(tl.float32)
    tl.store(out_ptr + off, (attn * m).to(act_dtype), mask=mask)


def fused_attention_gate(
    attention_output: torch.Tensor,
    gate: torch.Tensor,
    gate_bias: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """`attention_output * sigmoid(gate + gate_bias[head])` for 2-D `[tokens, heads*head_dim]` inputs."""
    if attention_output.shape != gate.shape or attention_output.ndim != 2:
        raise ValueError("attention_output and gate must be matching 2-D tensors")
    attention_output = attention_output.contiguous()
    gate = gate.contiguous()
    hidden_dim = attention_output.shape[1]
    if gate_bias.numel() * head_dim != hidden_dim or not gate_bias.is_contiguous():
        raise ValueError("gate_bias must hold one contiguous entry per local head")
    out = torch.empty_like(attention_output)
    rows = attention_output.shape[0]
    if rows == 0:
        return out
    block_h = 1024
    _fused_attention_gate_kernel[(rows, triton.cdiv(hidden_dim, block_h))](
        out,
        attention_output,
        gate,
        gate_bias,
        hidden_dim=hidden_dim,
        HEAD_DIM=head_dim,
        BLOCK_H=block_h,
        num_warps=4,
    )
    return out
