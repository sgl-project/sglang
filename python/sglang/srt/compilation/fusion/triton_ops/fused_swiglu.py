# Copyright 2024-2025 Ben Fattori and SGLang Team
#
# Adapted from https://github.com/fattorib/fusedswiglu
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the “Software”),
# to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# ==============================================================================

# Computes fused SwiGLU forward pass adapted from `GLU Variants Improve Transformer`
# <https://arxiv.org/abs/2002.05202>

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.utils import direct_register_custom_op


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)


@triton.jit
def swiglu_kernel(
    x_ptr,
    w_gate_ptr,
    w_up_ptr,
    o_ptr,
    x_scale_ptr,
    w_scale_ptr,
    o_scale_ptr,
    USE_SCALE: tl.constexpr,
    xrow_stride: tl.constexpr,
    xcol_stride: tl.constexpr,
    wrow_stride: tl.constexpr,
    wcol_stride: tl.constexpr,
    orow_stride: tl.constexpr,
    ocol_stride: tl.constexpr,
    dim_m,
    dim_n,
    dim_k,
    min_val: tl.constexpr,
    max_val: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # fmt: on
    pid = tl.program_id(0)

    num_pid_row = tl.cdiv(dim_m, BLOCK_SIZE_M)
    num_pid_col = tl.cdiv(dim_k, BLOCK_SIZE_K)

    num_pid_in_group = GROUP_SIZE_M * num_pid_col
    group_id = pid // num_pid_in_group
    first_pid_row = group_id * GROUP_SIZE_M
    group_size_row = min(num_pid_row - first_pid_row, GROUP_SIZE_M)
    pid_row = first_pid_row + (pid % group_size_row)
    pid_col = (pid % num_pid_in_group) // group_size_row

    acc_gate = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=tl.float32)
    acc_up = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=tl.float32)

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(dim_m, dim_n),
        strides=(xrow_stride, xcol_stride),
        offsets=(pid_row * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    w_gate_block_ptr = tl.make_block_ptr(
        w_gate_ptr,
        shape=(dim_n, dim_k),
        strides=(wrow_stride, wcol_stride),
        offsets=(
            0,
            pid_col * BLOCK_SIZE_K,
        ),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K),
        order=(1, 0),
    )
    w_up_block_ptr = tl.make_block_ptr(
        w_up_ptr,
        shape=(dim_n, dim_k),
        strides=(wrow_stride, wcol_stride),
        offsets=(
            0,
            pid_col * BLOCK_SIZE_K,
        ),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K),
        order=(1, 0),
    )

    for _ in range(0, tl.cdiv(dim_n, BLOCK_SIZE_N)):
        x_block = tl.load(x_block_ptr, boundary_check=(0, 1))

        w_gate_block = tl.load(w_gate_block_ptr, boundary_check=(0, 1))
        w_up_block = tl.load(w_up_block_ptr, boundary_check=(0, 1))

        acc_gate += tl.dot(x_block, w_gate_block, allow_tf32=False)
        acc_up += tl.dot(x_block, w_up_block, allow_tf32=False)

        x_block_ptr = tl.advance(x_block_ptr, offsets=(0, BLOCK_SIZE_N))
        w_gate_block_ptr = tl.advance(w_gate_block_ptr, offsets=(BLOCK_SIZE_N, 0))
        w_up_block_ptr = tl.advance(w_up_block_ptr, offsets=(BLOCK_SIZE_N, 0))

    if USE_SCALE:
        scale = tl.load(w_scale_ptr) * tl.load(x_scale_ptr)
        o_scale_inv = 1.0 / tl.load(o_scale_ptr)
        acc_up = (acc_up * scale) * silu(acc_gate * scale)
        acc_up = tl.clamp(acc_up * o_scale_inv, min_val, max_val)
    else:
        acc_up *= silu(acc_gate)

    offs_out_m = pid_row * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_out_k = pid_col * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    mask = (offs_out_m[:, None] < dim_m) & (offs_out_k[None, :] < dim_k)
    o_ptrs = (
        (o_ptr) + offs_out_m[:, None] * orow_stride + offs_out_k[None, :] * ocol_stride
    )
    tl.store(o_ptrs, acc_up.to(o_ptr.type.element_ty), mask=mask)


def fused_swiglu(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: Optional[torch.Tensor] = None,
    w_scale: Optional[torch.Tensor] = None,
    o_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    M, N, K = x.shape[0], x.shape[1], w.shape[1] // 2
    out = torch.empty((M, K), device=x.device, dtype=x.dtype)
    w_gate, w_up = torch.split(w, w.shape[1] // 2, dim=1)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(K, META["BLOCK_SIZE_K"]),
        )

    # fmt: off
    swiglu_kernel[grid](
        x,w_gate,w_up,out,x_scale,w_scale,o_scale,
        x_scale is not None and w_scale is not None and o_scale is not None,
        x.stride(0),x.stride(1),
        w_up.stride(0),w_up.stride(1),
        out.stride(0),out.stride(1),
        M,N,K,
        torch.finfo(out.dtype).min,
        torch.finfo(out.dtype).max,
        128,64,128,1,num_warps=4,num_stages=4,
    )

    # fmt: on
    return out


def fused_swiglu_fake(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: Optional[torch.Tensor] = None,
    w_scale: Optional[torch.Tensor] = None,
    o_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    M, N, K = x.shape[0], x.shape[1], w.shape[1] // 2
    return torch.empty((M, K), device=x.device, dtype=x.dtype)


direct_register_custom_op(
    op_name="fused_swiglu",
    op_func=fused_swiglu,
    mutates_args=[],
    fake_impl=fused_swiglu_fake,
)


def fused_swiglu_fwd(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: Optional[torch.Tensor] = None,
    w_scale: Optional[torch.Tensor] = None,
    o_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.ops.sglang.fused_swiglu(x, w, x_scale, w_scale, o_scale)
