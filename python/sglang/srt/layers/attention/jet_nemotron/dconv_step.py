# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
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
#
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl


def ensure_contiguous(t: torch.Tensor) -> torch.Tensor:
    return t if t.is_contiguous() else t.contiguous()


@triton.jit
def _causal_conv_step_kernel(
    X_ptr,
    Cache_ptr,
    Kernels_ptr,
    Out_ptr,
    D,
    X_stride_b,
    X_stride_d,
    Cache_stride_b,
    Cache_stride_d,
    Cache_stride_w,
    Kernels_stride_b,
    Kernels_stride_d,
    Kernels_stride_w,
    Out_stride_b,
    Out_stride_d,
    W: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """
    Triton kernel for a single step (T=1) of causal dynamic convolution.
    Updates the cache in-place and computes the output (without activation).
    Optimized for small W (1 < W <= 4) by manually unrolling the W dimension.
    Does NOT handle separate static bias.

    Grid: (B, cdiv(D, BLOCK_SIZE_D))
    Updates Cache[b, d, :] and computes Out[b, d].
    """
    pid_b = tl.program_id(0)
    pid_d_block = tl.program_id(1)

    offs_d = pid_d_block * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    d_mask = offs_d < D

    x_ptrs = X_ptr + pid_b * X_stride_b + offs_d * X_stride_d
    x_curr = tl.load(x_ptrs, mask=d_mask, other=0.0)

    accumulator = tl.zeros((BLOCK_SIZE_D,), dtype=x_curr.dtype)

    if tl.constexpr(W > 1):
        k_ptr_0 = (
            Kernels_ptr
            + pid_b * Kernels_stride_b
            + offs_d * Kernels_stride_d
            + 0 * Kernels_stride_w
        )
        k_val_0 = tl.load(k_ptr_0, mask=d_mask, other=0.0)

        cache_ptr_1 = (
            Cache_ptr
            + pid_b * Cache_stride_b
            + offs_d * Cache_stride_d
            + 1 * Cache_stride_w
        )
        cache_val_1 = tl.load(cache_ptr_1, mask=d_mask, other=0.0)

        accumulator += cache_val_1 * k_val_0

    if tl.constexpr(W > 2):
        k_ptr_1 = (
            Kernels_ptr
            + pid_b * Kernels_stride_b
            + offs_d * Kernels_stride_d
            + 1 * Kernels_stride_w
        )
        k_val_1 = tl.load(k_ptr_1, mask=d_mask, other=0.0)

        cache_ptr_2 = (
            Cache_ptr
            + pid_b * Cache_stride_b
            + offs_d * Cache_stride_d
            + 2 * Cache_stride_w
        )
        cache_val_2 = tl.load(cache_ptr_2, mask=d_mask, other=0.0)

        accumulator += cache_val_2 * k_val_1

        cache_ptr_1 = (
            Cache_ptr
            + pid_b * Cache_stride_b
            + offs_d * Cache_stride_d
            + 1 * Cache_stride_w
        )
        tl.store(cache_ptr_1, cache_val_2, mask=d_mask)

    if tl.constexpr(W > 3):
        k_ptr_2 = (
            Kernels_ptr
            + pid_b * Kernels_stride_b
            + offs_d * Kernels_stride_d
            + 2 * Kernels_stride_w
        )
        k_val_2 = tl.load(k_ptr_2, mask=d_mask, other=0.0)

        cache_ptr_3 = (
            Cache_ptr
            + pid_b * Cache_stride_b
            + offs_d * Cache_stride_d
            + 3 * Cache_stride_w
        )
        cache_val_3 = tl.load(cache_ptr_3, mask=d_mask, other=0.0)

        accumulator += cache_val_3 * k_val_2

        cache_ptr_2 = (
            Cache_ptr
            + pid_b * Cache_stride_b
            + offs_d * Cache_stride_d
            + 2 * Cache_stride_w
        )
        tl.store(cache_ptr_2, cache_val_3, mask=d_mask)

    k_ptr_last = (
        Kernels_ptr
        + pid_b * Kernels_stride_b
        + offs_d * Kernels_stride_d
        + (W - 1) * Kernels_stride_w
    )
    k_val_last = tl.load(k_ptr_last, mask=d_mask, other=0.0)

    accumulator += x_curr * k_val_last

    cache_ptr_last = (
        Cache_ptr
        + pid_b * Cache_stride_b
        + offs_d * Cache_stride_d
        + (W - 1) * Cache_stride_w
    )
    tl.store(cache_ptr_last, x_curr, mask=d_mask)

    out_ptrs = Out_ptr + pid_b * Out_stride_b + offs_d * Out_stride_d
    tl.store(out_ptrs, accumulator, mask=d_mask)


def causal_conv_step_triton(
    x: torch.Tensor,  # Input tensor [B, 1, D]
    cache: torch.Tensor,  # Cache tensor [B, D, W-1], modified in-place
    kernels: torch.Tensor,  # Kernels tensor [B, D, W]
) -> torch.Tensor:  # Returns output tensor [B, D] (before activation)
    """
    Performs one step of causal dynamic convolution using Triton.
    Updates the cache in-place. Does NOT fuse activation. Assumes 1 < W <= 4.
    Uses manually unrolled kernel for W dimension.

    Args:
        x: Current input token tensor of shape [B, 1, D].
        cache: Cache tensor of shape [B, D, W]. Will be updated in-place.
        kernels: Dynamically generated kernels tensor of shape [B, D, W].

    Returns:
        Output tensor of shape [B, D] for the current step (before activation).
    """
    assert x.dim() == 3 and x.shape[1] == 1, "Input x must have shape [B, 1, D]"
    assert cache.dim() == 3, "Cache must have shape [B, D, W]"
    assert kernels.dim() == 3, "Kernels must have shape [B, D, W]"
    B, _, D = x.shape
    W = cache.shape[2]
    assert 1 < W <= 4, f"Kernel W={W}, this optimized version assumes 1 < W <= 4"

    x_squeezed = x.squeeze(1)
    x_squeezed = ensure_contiguous(x_squeezed)
    cache = ensure_contiguous(cache)
    kernels = ensure_contiguous(kernels)

    out = torch.empty_like(x_squeezed)
    grid = lambda meta: (B, triton.cdiv(D, meta["BLOCK_SIZE_D"]))
    BLOCK_SIZE_D = 64

    _causal_conv_step_kernel[grid](
        x_squeezed,
        cache,
        kernels,
        out,
        D,
        x_squeezed.stride(0),
        x_squeezed.stride(1),
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        kernels.stride(0),
        kernels.stride(1),
        kernels.stride(2),
        out.stride(0),
        out.stride(1),
        W=W,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )

    return out
