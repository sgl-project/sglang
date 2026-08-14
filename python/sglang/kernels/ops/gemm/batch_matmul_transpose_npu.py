# Copyright 2023-2026 SGLang Team
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
from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.npu import get_npu_ai_core_count

_GENERIC_CONFIGS = [
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 128, "BLOCK_K": 32}),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 128, "BLOCK_K": 32}),
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 128, "BLOCK_K": 64}),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 128, "BLOCK_K": 64}),
]
_FAST_CONFIGS = [
    triton.Config({"BLOCK_B": 8, "BLOCK_N": 128, "BLOCK_K": 64}),
    triton.Config({"BLOCK_B": 8, "BLOCK_N": 128, "BLOCK_K": 128}),
]
_B1_FAST_CONFIGS = [
    triton.Config({"BLOCK_N": 128, "BLOCK_K": 64}),
    triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}),
]


@triton.autotune(configs=_GENERIC_CONFIGS, key=["B", "M", "K", "N"])
@triton.jit
def _batch_matmul_transpose_generic_kernel(
    a_ptr,
    w_ptr,
    c_ptr,
    stride_ab,
    stride_am,
    stride_ak,
    stride_wm,
    stride_wk,
    stride_wn,
    stride_cb,
    stride_cm,
    stride_cn,
    B,
    M,
    K,
    N,
    IS_BF16: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_cores = tl.num_programs(0)
    num_b_tiles = tl.cdiv(B, BLOCK_B)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    num_m_groups = tl.cdiv(M, GROUP_M)
    num_blocks = num_m_groups * num_b_tiles * num_n_tiles
    mn_tiles = num_b_tiles * num_n_tiles

    for block_idx in range(pid, num_blocks, num_cores):
        m_group_idx = block_idx // mn_tiles
        remainder = block_idx % mn_tiles
        b_tile_idx = remainder // num_n_tiles
        n_tile_idx = remainder % num_n_tiles
        offs_b = b_tile_idx * BLOCK_B + tl.arange(0, BLOCK_B)
        offs_n = n_tile_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        m_base = m_group_idx * GROUP_M
        for group_idx in range(GROUP_M):
            m_idx = m_base + group_idx
            if m_idx < M:
                accumulator = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)
                for k_base in range(0, K, BLOCK_K):
                    offs_k = k_base + tl.arange(0, BLOCK_K)
                    a_ptrs = (
                        a_ptr
                        + offs_b[:, None] * stride_ab
                        + m_idx * stride_am
                        + offs_k[None, :] * stride_ak
                    )
                    w_ptrs = (
                        w_ptr
                        + m_idx * stride_wm
                        + offs_k[:, None] * stride_wk
                        + offs_n[None, :] * stride_wn
                    )
                    a = tl.load(
                        a_ptrs,
                        mask=(offs_b[:, None] < B) & (offs_k[None, :] < K),
                        other=0.0,
                    )
                    w = tl.load(
                        w_ptrs,
                        mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
                        other=0.0,
                    )
                    accumulator += tl.dot(a, w)
                c_ptrs = (
                    c_ptr
                    + offs_b[:, None] * stride_cb
                    + m_idx * stride_cm
                    + offs_n[None, :] * stride_cn
                )
                output = (
                    accumulator.to(tl.bfloat16)
                    if IS_BF16
                    else accumulator.to(tl.float16)
                )
                tl.store(
                    c_ptrs,
                    output,
                    mask=(offs_b[:, None] < B) & (offs_n[None, :] < N),
                )


@triton.autotune(configs=_FAST_CONFIGS, key=["B", "M", "K", "N"])
@triton.jit
def _batch_matmul_transpose_fast_kernel(
    a_ptr,
    w_ptr,
    c_ptr,
    stride_ab,
    stride_am,
    stride_ak,
    stride_wm,
    stride_wk,
    stride_wn,
    stride_cb,
    stride_cm,
    stride_cn,
    B,
    M,
    K,
    N,
    IS_BF16: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_cores = tl.num_programs(0)
    num_b_tiles = B // BLOCK_B
    num_n_tiles = N // BLOCK_N
    num_m_groups = M // GROUP_M
    num_blocks = num_m_groups * num_b_tiles * num_n_tiles
    mn_tiles = num_b_tiles * num_n_tiles
    for block_idx in range(pid, num_blocks, num_cores):
        m_group_idx = block_idx // mn_tiles
        remainder = block_idx % mn_tiles
        b_tile_idx = remainder // num_n_tiles
        n_tile_idx = remainder % num_n_tiles
        offs_b = b_tile_idx * BLOCK_B + tl.arange(0, BLOCK_B)
        offs_n = n_tile_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        m_base = m_group_idx * GROUP_M
        for group_idx in range(GROUP_M):
            m_idx = m_base + group_idx
            accumulator = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)
            for k_base in range(0, K, BLOCK_K):
                offs_k = k_base + tl.arange(0, BLOCK_K)
                a = tl.load(
                    a_ptr
                    + offs_b[:, None] * stride_ab
                    + m_idx * stride_am
                    + offs_k[None, :] * stride_ak
                )
                w = tl.load(
                    w_ptr
                    + m_idx * stride_wm
                    + offs_k[:, None] * stride_wk
                    + offs_n[None, :] * stride_wn
                )
                accumulator += tl.dot(a, w)
            c_ptrs = (
                c_ptr
                + offs_b[:, None] * stride_cb
                + m_idx * stride_cm
                + offs_n[None, :] * stride_cn
            )
            output = (
                accumulator.to(tl.bfloat16) if IS_BF16 else accumulator.to(tl.float16)
            )
            tl.store(c_ptrs, output)


@triton.autotune(configs=_B1_FAST_CONFIGS, key=["M", "K", "N"])
@triton.jit
def _batch_matmul_transpose_b1_fast_kernel(
    a_ptr,
    w_ptr,
    c_ptr,
    stride_am,
    stride_ak,
    stride_wm,
    stride_wk,
    stride_wn,
    stride_cm,
    stride_cn,
    M,
    K,
    N,
    IS_BF16: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_cores = tl.num_programs(0)
    num_n_tiles = N // BLOCK_N
    num_blocks = (M // GROUP_M) * num_n_tiles
    offs_n = tl.arange(0, BLOCK_N)
    for block_idx in range(pid, num_blocks, num_cores):
        m_group_idx = block_idx // num_n_tiles
        n_tile_idx = block_idx % num_n_tiles
        n_base = n_tile_idx * BLOCK_N
        m_base = m_group_idx * GROUP_M
        for group_idx in range(GROUP_M):
            m_idx = m_base + group_idx
            accumulator = tl.zeros((1, BLOCK_N), dtype=tl.float32)
            for k_base in range(0, K, BLOCK_K):
                offs_k = k_base + tl.arange(0, BLOCK_K)
                a = tl.load(a_ptr + m_idx * stride_am + offs_k[None, :] * stride_ak)
                w = tl.load(
                    w_ptr
                    + m_idx * stride_wm
                    + offs_k[:, None] * stride_wk
                    + (n_base + offs_n)[None, :] * stride_wn
                )
                accumulator += tl.dot(a, w)
            c_ptrs = c_ptr + m_idx * stride_cm + (n_base + offs_n)[None, :] * stride_cn
            output = (
                accumulator.to(tl.bfloat16) if IS_BF16 else accumulator.to(tl.float16)
            )
            tl.store(c_ptrs, output)


def npu_nz_to_nd(tensor_b: torch.Tensor) -> torch.Tensor:
    if tensor_b.ndim != 4 or tensor_b.shape[-1] != 16:
        raise ValueError("NZ weights must have shape [M, N/16, K, 16]")
    m, n16, k, inner = tensor_b.shape
    return tensor_b.permute(0, 2, 1, 3).contiguous().view(m, k, n16 * inner)


def batch_matmul_transpose_npu(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    tensor_c: torch.Tensor,
    format_mode: str | None = None,
    quant_mode: str | None = None,
) -> None:
    """Compute C[b,m,n] = A[b,m,k] @ B[m,k,n] on Ascend A5."""
    del quant_mode
    mode = "ND" if format_mode is None else str(format_mode).upper()
    if mode not in ("ND", "NZ"):
        raise ValueError(f"Unsupported NPU weight format {format_mode!r}")
    weight = npu_nz_to_nd(tensor_b) if mode == "NZ" else tensor_b
    if tensor_a.ndim != 3 or weight.ndim != 3 or tensor_c.ndim != 3:
        raise ValueError("batch_matmul_transpose_npu requires rank-3 ND tensors")
    batch, num_rows, inner = tensor_a.shape
    if weight.shape[:2] != (num_rows, inner):
        raise ValueError("A and B dimensions are incompatible")
    output_width = weight.shape[2]
    if tensor_c.shape != (batch, num_rows, output_width):
        raise ValueError("output tensor has the wrong shape")
    a = tensor_a.contiguous()
    weight = weight.contiguous()
    group_m = 1 if num_rows < 16 or num_rows % 2 else 2
    grid = (get_npu_ai_core_count(),)
    common = (
        a,
        weight,
        tensor_c,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        tensor_c.stride(0),
        tensor_c.stride(1),
        tensor_c.stride(2),
    )
    is_bf16 = a.dtype == torch.bfloat16
    if batch == 1 and output_width % 128 == 0 and inner % 64 == 0:
        _batch_matmul_transpose_b1_fast_kernel[grid](
            a,
            weight,
            tensor_c,
            a.stride(1),
            a.stride(2),
            weight.stride(0),
            weight.stride(1),
            weight.stride(2),
            tensor_c.stride(1),
            tensor_c.stride(2),
            num_rows,
            inner,
            output_width,
            IS_BF16=is_bf16,
            GROUP_M=group_m,
        )
    elif batch % 8 == 0 and output_width % 128 == 0 and inner % 128 == 0:
        _batch_matmul_transpose_fast_kernel[grid](
            *common,
            batch,
            num_rows,
            inner,
            output_width,
            IS_BF16=is_bf16,
            GROUP_M=group_m,
        )
    else:
        _batch_matmul_transpose_generic_kernel[grid](
            *common,
            batch,
            num_rows,
            inner,
            output_width,
            IS_BF16=is_bf16,
            GROUP_M=group_m,
        )
