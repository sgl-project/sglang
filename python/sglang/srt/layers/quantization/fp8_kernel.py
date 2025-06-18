# Copyright 2024 SGLang Team
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

import functools
import json
import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.math_utils import align
from sglang.srt.layers.quantization import deep_gemm_wrapper
from sglang.srt.utils import (
    direct_register_custom_op,
    get_device_core_count,
    get_device_name,
    is_cuda,
    is_hip,
    log_info_on_rank0,
    supports_custom_op,
)

_is_hip = is_hip()
_is_cuda = is_cuda()

if _is_cuda:
    from sgl_kernel import (
        sgl_per_tensor_quant_fp8,
        sgl_per_token_group_quant_fp8,
        sgl_per_token_quant_fp8,
    )

logger = logging.getLogger(__name__)


@lru_cache()
def is_fp8_fnuz() -> bool:
    if _is_hip:
        # only device 0 is checked, this assumes MI300 platforms are homogeneous
        return "gfx94" in torch.cuda.get_device_properties(0).gcnArchName
    return False


if is_fp8_fnuz():
    fp8_dtype = torch.float8_e4m3fnuz
    fp8_max = 224.0
else:
    fp8_dtype = torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max
fp8_min = -fp8_max

if supports_custom_op():

    def deep_gemm_fp8_fp8_bf16_nt(
        A: torch.Tensor,
        As: torch.Tensor,
        B: torch.Tensor,
        Bs: torch.Tensor,
        C: torch.Tensor,
    ) -> None:
        deep_gemm_wrapper.gemm_nt_f8f8bf16((A, As), (B, Bs), C)

    def deep_gemm_fp8_fp8_bf16_nt_fake(
        A: torch.Tensor,
        As: torch.Tensor,
        B: torch.Tensor,
        Bs: torch.Tensor,
        C: torch.Tensor,
    ) -> None:
        return

    direct_register_custom_op(
        op_name="deep_gemm_fp8_fp8_bf16_nt",
        op_func=deep_gemm_fp8_fp8_bf16_nt,
        mutates_args=["C"],
        fake_impl=deep_gemm_fp8_fp8_bf16_nt_fake,
    )


@triton.jit
def _per_token_group_quant_fp8(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    # Stride of input
    y_stride,
    # Columns of input
    N,
    # Avoid to divide zero
    eps,
    # Information for float8
    fp8_min,
    fp8_max,
    # Meta-parameters
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group quantization on a
    tensor.

    This function converts the tensor values into float8 values.
    """
    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    y_ptr += g_id * y_stride
    y_q_ptr += g_id * y_stride
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)  # N <= BLOCK
    mask = cols < N

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / fp8_max
    y_s_inv = 1.0 / y_s
    y_q = tl.clamp(y * y_s_inv, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


@triton.jit
def _per_token_group_quant_fp8_colmajor(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    # Num columns of y
    y_num_columns,
    # Stride from one column to the next of y_s
    y_s_col_stride,
    # Avoid to divide zero
    eps,
    # Information for float8
    fp8_min,
    fp8_max,
    # Meta-parameters
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor.
    This function converts the tensor values into float8 values.
    """
    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    y_ptr += g_id * group_size
    y_q_ptr += g_id * group_size

    # Convert g_id the flattened block coordinate to 2D so we can index
    # into the output y_scales matrix
    blocks_per_row = y_num_columns // group_size
    scale_col = g_id % blocks_per_row
    scale_row = g_id // blocks_per_row
    y_s_ptr += scale_col * y_s_col_stride + scale_row

    cols = tl.arange(0, BLOCK)  # group_size <= BLOCK
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / fp8_max
    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    column_major_scales: bool = False,
    scale_tma_aligned: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Function to perform per-token-group quantization on an input tensor `x`.

    It converts the tensor values into signed float8 values and returns the
    quantized tensor along with the scaling factor used for quantization.

    Args:
        x: The input tenosr with ndim >= 2.
        group_size: The group size used for quantization.
        eps: The minimum to avoid dividing zero.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The quantized tensor and the scaling factor for quantization.
    """
    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    x_q = torch.empty_like(x, device=x.device, dtype=fp8_dtype)
    M = x.numel() // group_size
    N = group_size
    if column_major_scales:
        if scale_tma_aligned:
            # aligned to 4 * sizeof(float)
            aligned_size = (x.shape[-2] + 3) // 4 * 4
            x_s = torch.empty(
                x.shape[:-2] + (x.shape[-1] // group_size, aligned_size),
                device=x.device,
                dtype=torch.float32,
            ).permute(-1, -2)[: x.shape[-2], :]
        else:
            x_s = torch.empty(
                (x.shape[-1] // group_size,) + x.shape[:-1],
                device=x.device,
                dtype=torch.float32,
            ).permute(-1, -2)
    else:
        x_s = torch.empty(
            x.shape[:-1] + (x.shape[-1] // group_size,),
            device=x.device,
            dtype=torch.float32,
        )

    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1
    if column_major_scales:
        _per_token_group_quant_fp8_colmajor[(M,)](
            x,
            x_q,
            x_s,
            group_size,
            x.shape[1],
            x_s.stride(1),
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        _per_token_group_quant_fp8[(M,)](
            x,
            x_q,
            x_s,
            group_size,
            N,
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return x_q, x_s


def sglang_per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    column_major_scales: bool = False,
    scale_tma_aligned: bool = False,
    scale_ue8m0: bool = False,
):
    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    x_q = torch.empty_like(x, device=x.device, dtype=fp8_dtype)
    if scale_ue8m0:
        assert column_major_scales and scale_tma_aligned
        x_q_mn, x_q_k = x.shape
        x_s_mn, x_s_k = x_q_mn, x_q_k // 128
        aligned_mn = align(x_s_mn, 4)
        aligned_k = align(x_s_k, 4)
        # TODO(FIXME): Fix cuda kernel and recover here to empty.
        x_s = torch.zeros(
            (aligned_k // 4, aligned_mn),
            device=x.device,
            dtype=torch.int,
        ).transpose(0, 1)[:x_s_mn, :]
    elif column_major_scales:
        if scale_tma_aligned:
            # TODO extract "align" function
            # aligned to 4 * sizeof(float)
            aligned_size = (x.shape[-2] + 3) // 4 * 4
            x_s = torch.empty(
                x.shape[:-2] + (x.shape[-1] // group_size, aligned_size),
                device=x.device,
                dtype=torch.float32,
            ).permute(-1, -2)[: x.shape[-2], :]
        else:
            x_s = torch.empty(
                (x.shape[-1] // group_size,) + x.shape[:-1],
                device=x.device,
                dtype=torch.float32,
            ).permute(-1, -2)
    else:
        x_s = torch.empty(
            x.shape[:-1] + (x.shape[-1] // group_size,),
            device=x.device,
            dtype=torch.float32,
        )
    if x.shape[0] > 0:
        sgl_per_token_group_quant_fp8(
            x, x_q, x_s, group_size, eps, fp8_min, fp8_max, scale_ue8m0
        )

    return x_q, x_s


def sglang_per_token_quant_fp8(
    x: torch.Tensor,
    dtype: torch.dtype = fp8_dtype,
):
    assert x.is_contiguous(), "`x` is not contiguous"

    x_q = torch.empty_like(x, device=x.device, dtype=dtype)
    x_s = torch.empty(
        x.shape[0],
        1,
        device=x.device,
        dtype=torch.float32,
    )

    sgl_per_token_quant_fp8(x, x_q, x_s)

    return x_q, x_s


@triton.jit
def _static_quant_fp8(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    y_s_repeat_ptr,
    # Stride of input
    y_stride,
    # Columns of input
    N,
    # Information for float8
    fp8_min,
    fp8_max,
    # Meta-parameters
    BLOCK: tl.constexpr,
    REPEAT_SCALE: tl.constexpr,
):
    """A Triton-accelerated function to perform quantization using the given scale on a
    tensor

    This function converts the tensor values into float8 values.
    """
    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    y_ptr += g_id * y_stride
    y_q_ptr += g_id * y_stride
    if REPEAT_SCALE:
        y_s_repeat_ptr += g_id

    cols = tl.arange(0, BLOCK)  # N <= BLOCK
    mask = cols < N

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y_s = tl.load(y_s_ptr).to(tl.float32)
    y_s_inv = 1.0 / y_s
    y_q = tl.clamp(y * y_s_inv, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    if REPEAT_SCALE:
        tl.store(y_s_repeat_ptr, y_s)


def static_quant_fp8(
    x: torch.Tensor,
    x_s: torch.Tensor,
    repeat_scale: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Function to perform static quantization using the given scale on an input tensor `x`.

    It converts the tensor values into signed float8 values and returns the
    quantized tensor along with the scaling factor used for quantization.

    Args:
        x: The input tenosr with ndim >= 2.
        x_s: The quantization scale.
        repeat_scale: Whether to broadcast per-tensor scale to per-channel scale.
        dtype: The dype of output tensor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The quantized tensor and the scaling factor for quantization.
    """
    assert x.is_contiguous(), "`x` is not contiguous"
    assert x_s.numel() == 1, "only supports per-tensor scale"

    x_q = torch.empty_like(x, device=x.device, dtype=fp8_dtype)
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    if repeat_scale:
        x_s_repeat = torch.empty(
            (M, 1),
            device=x.device,
            dtype=torch.float32,
        )
    else:
        x_s_repeat = None

    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1
    _static_quant_fp8[(M,)](
        x,
        x_q,
        x_s,
        x_s_repeat,
        N,
        N,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        BLOCK=BLOCK,
        REPEAT_SCALE=repeat_scale,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    x_s = x_s_repeat if repeat_scale else x_s
    return x_q, x_s


@triton.jit
def _w8a8_block_fp8_matmul(
    # Pointers to inputs and output
    A,
    B,
    C,
    As,
    Bs,
    # Shape for matmul
    M,
    N,
    K,
    # Block size for block-wise quantization
    group_n,
    group_k,
    # Stride for inputs and output
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Triton-accelerated function used to perform linear operations (dot
    product) on input tensors `A` and `B` with block-wise quantization, and store the result in output
    tensor `C`.
    """

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    As_ptrs = As + offs_am * stride_As_m
    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        k_start = k * BLOCK_SIZE_K
        offs_ks = k_start // group_k
        a_s = tl.load(As_ptrs + offs_ks * stride_As_k)
        b_s = tl.load(Bs_ptrs + offs_ks * stride_Bs_k)

        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def _w8a8_block_fp8_matmul_unrolledx4(
    # Pointers to inputs and output
    A,
    B,
    C,
    As,
    Bs,
    # Shape for matmul
    M,
    N,
    K,
    # Block size for block-wise quantization
    group_n,
    group_k,
    # Stride for inputs and output
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Triton-accelerated function used to perform linear operations (dot
    product) on input tensors `A` and `B` with block-wise quantization, and store the result in output
    tensor `C`.
    """

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    As_ptrs = As + offs_am * stride_As_m
    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # manually unroll to 4 iterations
    UNROLL_FACTOR = 4
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * UNROLL_FACTOR)):
        # 1st iteration
        a = tl.load(
            a_ptrs,
            mask=offs_k[None, :] < K - (k * UNROLL_FACTOR) * BLOCK_SIZE_K,
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=offs_k[:, None] < K - (k * UNROLL_FACTOR) * BLOCK_SIZE_K,
            other=0.0,
        )

        k_start = (k * UNROLL_FACTOR) * BLOCK_SIZE_K
        offs_ks = k_start // group_k
        a_s = tl.load(As_ptrs + offs_ks * stride_As_k)
        b_s = tl.load(Bs_ptrs + offs_ks * stride_Bs_k)

        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

        # 2nd iteration
        a = tl.load(
            a_ptrs,
            mask=offs_k[None, :] < K - (k * UNROLL_FACTOR + 1) * BLOCK_SIZE_K,
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=offs_k[:, None] < K - (k * UNROLL_FACTOR + 1) * BLOCK_SIZE_K,
            other=0.0,
        )

        k_start = k_start + BLOCK_SIZE_K
        offs_ks = k_start // group_k
        a_s = tl.load(As_ptrs + offs_ks * stride_As_k)
        b_s = tl.load(Bs_ptrs + offs_ks * stride_Bs_k)

        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

        # 3rd iteration
        a = tl.load(
            a_ptrs,
            mask=offs_k[None, :] < K - (k * UNROLL_FACTOR + 2) * BLOCK_SIZE_K,
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=offs_k[:, None] < K - (k * UNROLL_FACTOR + 2) * BLOCK_SIZE_K,
            other=0.0,
        )

        k_start = k_start + BLOCK_SIZE_K
        offs_ks = k_start // group_k
        a_s = tl.load(As_ptrs + offs_ks * stride_As_k)
        b_s = tl.load(Bs_ptrs + offs_ks * stride_Bs_k)

        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

        # 4th iteration
        a = tl.load(
            a_ptrs,
            mask=offs_k[None, :] < K - (k * UNROLL_FACTOR + 3) * BLOCK_SIZE_K,
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=offs_k[:, None] < K - (k * UNROLL_FACTOR + 3) * BLOCK_SIZE_K,
            other=0.0,
        )

        k_start = k_start + BLOCK_SIZE_K
        offs_ks = k_start // group_k
        a_s = tl.load(As_ptrs + offs_ks * stride_As_k)
        b_s = tl.load(Bs_ptrs + offs_ks * stride_Bs_k)

        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@functools.lru_cache
def get_w8a8_block_fp8_configs(
    N: int, K: int, block_n: int, block_k: int
) -> Optional[Dict[int, Any]]:
    """
    Return optimized configurations for the w8a8 block fp8 kernel.

    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the w8a8 block fp8 kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """

    # First look up if an optimized configuration is available in the configs
    # directory
    device_name = get_device_name().replace(" ", "_")
    json_file_name = f"N={N},K={K},device_name={device_name},dtype=fp8_w8a8,block_shape=[{block_n}, {block_k}].json"

    config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name
    )
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            log_info_on_rank0(
                logger,
                f"Using configuration from {config_file_path} for W8A8 Block FP8 kernel.",
            )
            # If a configuration has been found, return it
            return {int(key): val for key, val in json.load(f).items()}

    # If no optimized configuration is available, we will use the default
    # configuration
    logger.warning(
        (
            "Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal! "
            "Config file not found at %s"
        ),
        config_file_path,
    )
    return None


def select_w8a8_block_fp8_matmul_kernel(M, N, META):
    return _w8a8_block_fp8_matmul


if _is_hip:

    def use_w8a8_block_fp8_matmul_unrolledx4(M, N, META):
        # Use manually unrolledx4 kernel on AMD GPU when the grid size is small.
        # Empirical testing shows the sweet spot lies when it's less than the # of
        # compute units available on the device.
        num_workgroups = triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(
            N, META["BLOCK_SIZE_N"]
        )
        num_workgroups <= get_device_core_count()

    def select_w8a8_block_fp8_matmul_kernel(M, N, META):
        if use_w8a8_block_fp8_matmul_unrolledx4(M, N, META):
            return _w8a8_block_fp8_matmul_unrolledx4
        else:
            return _w8a8_block_fp8_matmul


def prepare_block_fp8_matmul_inputs(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: List[int],
    output_dtype: torch.dtype = torch.float16,
) -> Tuple[int, int, int]:
    assert len(block_size) == 2
    block_n, block_k = block_size[0], block_size[1]

    assert A.shape[-1] == B.shape[-1]
    assert A.shape[:-1] == As.shape[:-1]
    assert A.is_contiguous()

    if As.dtype == torch.float:
        assert triton.cdiv(A.shape[-1], block_k) == As.shape[-1]
    elif As.dtype == torch.int:
        assert (
            triton.cdiv(triton.cdiv(A.shape[-1], block_k), 4) == As.shape[-1]
        ), f"{A.shape=} {As.shape=} {block_size=}"
    else:
        raise NotImplementedError

    M = A.numel() // A.shape[-1]

    assert B.ndim == 2
    assert B.is_contiguous()
    assert Bs.ndim == 2
    N, K = B.shape

    if Bs.dtype == torch.float:
        assert triton.cdiv(N, block_n) == Bs.shape[0]
        assert triton.cdiv(K, block_k) == Bs.shape[1]
    elif Bs.dtype == torch.int:
        assert N == Bs.shape[0], f"{B.shape=} {Bs.shape=} {block_size=}"
        assert (
            triton.cdiv(triton.cdiv(K, block_k), 4) == Bs.shape[1]
        ), f"{B.shape=} {Bs.shape=} {block_size=}"
    else:
        raise NotImplementedError

    C_shape = A.shape[:-1] + (N,)
    C = A.new_empty(C_shape, dtype=output_dtype)

    return M, N, K, C


def w8a8_block_fp8_matmul_deepgemm(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: List[int],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    M, N, K, C = prepare_block_fp8_matmul_inputs(A, B, As, Bs, block_size, output_dtype)

    # Deepgemm only supports output tensor type as bfloat16
    assert C.dtype == torch.bfloat16 and deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM

    if supports_custom_op():
        torch.ops.sglang.deep_gemm_fp8_fp8_bf16_nt(A, As, B, Bs, C)
    else:
        deep_gemm_wrapper.gemm_nt_f8f8bf16((A, As), (B, Bs), C)

    return C


def w8a8_block_fp8_matmul_triton(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: List[int],
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """This function performs matrix multiplication with block-wise quantization.

    It takes two input tensors `A` and `B` with scales `As` and `Bs`.
    The output is returned in the specified `output_dtype`.

    Args:
        A: The input tensor, e.g., activation.
        B: The input tensor, e.g., weight.
        As: The per-token-group quantization scale for `A`.
        Bs: The per-block quantization scale for `B`.
        block_size: The block size for per-block quantization. It should be 2-dim, e.g., [128, 128].
        output_dytpe: The dtype of the returned tensor.

    Returns:
        torch.Tensor: The result of matmul.
    """

    M, N, K, C = prepare_block_fp8_matmul_inputs(A, B, As, Bs, block_size, output_dtype)

    block_n, block_k = block_size

    configs = get_w8a8_block_fp8_configs(N, K, block_size[0], block_size[1])
    if configs:
        # If an optimal configuration map has been found, look up the
        # optimal config
        config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
    else:
        # Default config
        # Block-wise quant: BLOCK_SIZE_K must be divisible by block_size[1]
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": block_size[0],
            "BLOCK_SIZE_K": block_size[1],
            "GROUP_SIZE_M": 32,
            "num_warps": 4,
            "num_stages": 3,
        }

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    kernel = select_w8a8_block_fp8_matmul_kernel(M, N, config)

    kernel[grid](
        A,
        B,
        C,
        As,
        Bs,
        M,
        N,
        K,
        block_n,
        block_k,
        A.stride(-2),
        A.stride(-1),
        B.stride(1),
        B.stride(0),
        C.stride(-2),
        C.stride(-1),
        As.stride(-2),
        As.stride(-1),
        Bs.stride(1),
        Bs.stride(0),
        **config,
    )

    return C


# universal entry point, for testing purposes
def w8a8_block_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: List[int],
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    if output_dtype == torch.bfloat16 and deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
        return w8a8_block_fp8_matmul_deepgemm(
            A, B, As, Bs, block_size, output_dtype=output_dtype
        )

    return w8a8_block_fp8_matmul_triton(
        A, B, As, Bs, block_size, output_dtype=output_dtype
    )


@triton.jit
def _per_tensor_quant_mla_fp8_stage1(
    x_ptr,
    x_s_ptr,
    head_size,
    x_stride_h,
    x_stride_s,
    eps,
    fp8_max,
    BLOCK_SIZE: tl.constexpr,
):
    seq_id = tl.program_id(0)
    head_id = tl.program_id(1)
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < head_size

    x_ptr += head_id * x_stride_h + seq_id * x_stride_s
    x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    _absmax = tl.maximum(tl.max(tl.abs(x)), eps)

    tl.atomic_max(x_s_ptr, _absmax / fp8_max)


@triton.jit
def _per_tensor_quant_mla_fp8_stage2(
    x_ptr,
    x_s_ptr,
    x_q_ptr,
    num_seq,
    head_size,
    x_stride_h,
    x_stride_s,
    fp8_min,
    fp8_max,
    BLOCK_SIZE: tl.constexpr,
):
    seq_id = tl.program_id(0)
    head_id = tl.program_id(1)
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < head_size

    x_s = tl.load(x_s_ptr)
    x_s_inv = 1.0 / x_s

    x_ptr += head_id * x_stride_h + seq_id * x_stride_s
    x_q_ptr += head_id * num_seq * head_size + seq_id * head_size

    x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    x_q = tl.clamp(x * x_s_inv, fp8_min, fp8_max).to(x_q_ptr.dtype.element_ty)
    tl.store(x_q_ptr + offset, x_q, mask=mask)


def per_tensor_quant_mla_fp8(
    x: torch.Tensor, x_s_out: torch.Tensor, eps: float = 1e-12
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This function quantizes input values to float8 values with tensor-wise quantization
    and specialized for mla absorbed case.
    """
    assert x.dim() == 3, "`x` is not a 3d-tensor"
    assert (
        x_s_out.shape == (1,)
        and x_s_out.dtype == torch.float32
        and x_s_out.device == x.device
    )

    x_q = x.new_empty(x.size(), dtype=fp8_dtype)

    num_head, num_seq, head_size = x.shape
    BLOCK_SIZE = triton.next_power_of_2(head_size)
    grid = (num_seq, num_head)

    _per_tensor_quant_mla_fp8_stage1[grid](
        x,
        x_s_out,
        head_size,
        x.stride(0),
        x.stride(1),
        eps,
        fp8_max,
        BLOCK_SIZE,
    )
    _per_tensor_quant_mla_fp8_stage2[grid](
        x,
        x_s_out,
        x_q,
        num_seq,
        head_size,
        x.stride(0),
        x.stride(1),
        fp8_min,
        fp8_max,
        BLOCK_SIZE,
    )

    return x_q, x_s_out


@triton.jit
def _per_token_group_quant_mla_deep_gemm_masked_fp8(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    masked_m_ptr,
    group_size,
    y_stride_b,
    y_stride_t,
    y_q_stride_b,
    y_q_stride_t,
    y_s_stride_b,
    y_s_stride_g,
    eps,
    fp8_min,
    fp8_max,
    NUM_GROUP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor for deep_gemm grouped_gemm_masked.
    This function converts the tensor values into float8 values.
    y and y_q: (b, t, k)
    y_s: (b, k//group_size, t)
    """
    t_id = tl.program_id(0)
    b_id = tl.program_id(1)

    y_ptr += b_id * y_stride_b + t_id * y_stride_t
    y_q_ptr += b_id * y_q_stride_b + t_id * y_q_stride_t
    y_s_ptr += b_id * y_s_stride_b + t_id

    if t_id == 0:
        tl.store(masked_m_ptr + b_id, tl.num_programs(0))

    cols = tl.arange(0, BLOCK)  # group_size <= BLOCK
    mask = cols < group_size

    for gid in range(NUM_GROUP):
        y = tl.load(y_ptr + gid * group_size + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
        y_s = _absmax / fp8_max
        y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

        tl.store(y_q_ptr + gid * group_size + cols, y_q, mask=mask)
        tl.store(y_s_ptr + gid * y_s_stride_g, y_s)


def per_token_group_quant_mla_deep_gemm_masked_fp8(
    x: torch.Tensor,
    group_size: int = 128,
    eps: float = 1e-12,
    dtype: torch.dtype = fp8_dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This function quantizes input values to float8 values with per-token-group-quantization
    for deep_gemm grouped_gemm_masked and specialized for mla absorbed case.
    """
    assert x.dim() == 3, "`x` is not a 3d-tensor"

    b, m, k = x.shape
    aligned_m = (m + 255) // 256 * 256  # 256 is the max block_m of the gemm kernel
    num_tiles_k = k // group_size
    assert num_tiles_k * group_size == k, f"k % {group_size} must be zero"

    x_q = x.new_empty((b, aligned_m, k), dtype=dtype)
    x_s = x.new_empty((b, num_tiles_k, aligned_m), dtype=torch.float32)
    masked_m = x.new_empty((b,), dtype=torch.int32)

    BLOCK_SIZE = triton.next_power_of_2(group_size)
    grid = (m, b)

    _per_token_group_quant_mla_deep_gemm_masked_fp8[grid](
        x,
        x_q,
        x_s,
        masked_m,
        group_size,
        x.stride(0),
        x.stride(1),
        x_q.stride(0),
        x_q.stride(1),
        x_s.stride(0),
        x_s.stride(1),
        eps,
        -fp8_max,
        fp8_max,
        num_tiles_k,
        BLOCK_SIZE,
    )

    return x_q, x_s.transpose(1, 2), masked_m, m, aligned_m


def scaled_fp8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    num_token_padding: Optional[int] = None,
    use_per_token_if_dynamic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to FP8 (8-bit floating point) format.

    Args:
        input (torch.Tensor): Input tensor to be quantized
        scale (Optional[torch.Tensor]): Pre-computed scaling factor for static quantization.
            If None, scales will be computed dynamically.
        num_token_padding (Optional[int]): If specified, pad the first dimension
            of the output to at least this value.
        use_per_token_if_dynamic (bool): When using dynamic scaling (scale=None),
            determines the quantization granularity:
            - True: compute scale per token
            - False: compute single scale per tensor

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - quantized_tensor: The FP8 quantized version of input
            - scale_tensor: The scaling factors used for quantization

    Raises:
        AssertionError: If input is not 2D or if static scale's numel != 1
    """
    assert input.ndim == 2, f"Expected 2D input tensor, got {input.ndim}D"
    shape = input.shape
    if num_token_padding:
        shape = (max(num_token_padding, input.shape[0]), shape[1])
    output = torch.empty(shape, device=input.device, dtype=fp8_dtype)

    if scale is None:
        # Dynamic scaling
        if use_per_token_if_dynamic:
            scale = torch.empty((shape[0], 1), device=input.device, dtype=torch.float32)
            sgl_per_token_quant_fp8(input, output, scale)
        else:
            scale = torch.zeros(1, device=input.device, dtype=torch.float32)
            sgl_per_tensor_quant_fp8(
                input, output, scale, is_static=False
            )  # False for dynamic
    else:
        # Static scaling
        assert scale.numel() == 1, f"Expected scalar scale, got numel={scale.numel()}"
        sgl_per_tensor_quant_fp8(
            input, output, scale, is_static=True
        )  # True for static

    return output, scale
