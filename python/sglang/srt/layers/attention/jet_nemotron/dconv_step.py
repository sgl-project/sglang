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


# Helper function to ensure tensors are contiguous for Triton
def ensure_contiguous(t: torch.Tensor) -> torch.Tensor:
    # Ensure tensor is contiguous in memory.
    return t if t.is_contiguous() else t.contiguous()


# Removed _apply_activation helper function


@triton.jit
def _causal_conv_step_kernel(
    # --- Input/Output Pointers ---
    X_ptr,  # Pointer to current input x [B, D] (after squeeze)
    Cache_ptr,  # Pointer to cache [B, D, W], updated IN-PLACE
    Kernels_ptr,  # Pointer to generated kernels [B, D, W]
    Out_ptr,  # Pointer to output tensor [B, D]
    # --- Tensor Dimensions ---
    B,
    D,  # Batch size, Feature dimension
    # --- Tensor Strides ---
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
    # --- Kernel Meta-Parameters ---
    W: tl.constexpr,  # Kernel width (Cache size), passed as compile-time constant (1 < W <= 4)
    BLOCK_SIZE_D: tl.constexpr,  # Block size for D dimension (tuning parameter)
    # Removed ACTIVATION: tl.constexpr
):
    """
    Triton kernel for a single step (T=1) of causal dynamic convolution.
    Updates the cache in-place and computes the output (without activation).
    Optimized for small W (1 < W <= 4) by manually unrolling the W dimension.
    Does NOT handle separate static bias.

    Grid: (B, cdiv(D, BLOCK_SIZE_D))
    Updates Cache[b, d, :] and computes Out[b, d].
    """
    # 1. --- Get Program IDs and Calculate Indices ---
    pid_b = tl.program_id(0)  # Program ID for batch dimension
    pid_d_block = tl.program_id(1)  # Program ID for dimension block

    offs_d = pid_d_block * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    d_mask = offs_d < D  # Shape: [BLOCK_SIZE_D]

    # 2. --- Load Current Input X ---
    x_ptrs = X_ptr + pid_b * X_stride_b + offs_d * X_stride_d
    x_curr = tl.load(x_ptrs, mask=d_mask, other=0.0)  # Shape: [BLOCK_SIZE_D]

    # --- Initialize Accumulator ---
    accumulator = tl.zeros((BLOCK_SIZE_D,), dtype=x_curr.dtype)  # Use input dtype

    # --- Manually Unroll Operations for W ---
    # We will load kernel values and cache values step-by-step
    # and perform the calculation and cache update.

    # --- Step w = 0 ---
    # Compute: cache_val_1 * k_val_0 (part 1)
    # Cache Update: store cache_val_1 at index 0
    if tl.constexpr(W > 1):
        # Load k_val_0
        k_ptr_0 = (
            Kernels_ptr
            + pid_b * Kernels_stride_b
            + offs_d * Kernels_stride_d
            + 0 * Kernels_stride_w
        )
        k_val_0 = tl.load(k_ptr_0, mask=d_mask, other=0.0)

        # Load cache_val_1 (needed for computation and storing at index 0)
        cache_ptr_1 = (
            Cache_ptr
            + pid_b * Cache_stride_b
            + offs_d * Cache_stride_d
            + 1 * Cache_stride_w
        )
        cache_val_1 = tl.load(cache_ptr_1, mask=d_mask, other=0.0)

        # Accumulate Part 1
        accumulator += cache_val_1 * k_val_0

        # Cache Update: Store cache_val_1 -> cache_ptr_0
        cache_ptr_0 = (
            Cache_ptr
            + pid_b * Cache_stride_b
            + offs_d * Cache_stride_d
            + 0 * Cache_stride_w
        )
        tl.store(cache_ptr_0, cache_val_1, mask=d_mask)

    # --- Step w = 1 ---
    # Compute: cache_val_2 * k_val_1 (part 1)
    # Cache Update: store cache_val_2 at index 1
    if tl.constexpr(W > 2):
        # Load k_val_1
        k_ptr_1 = (
            Kernels_ptr
            + pid_b * Kernels_stride_b
            + offs_d * Kernels_stride_d
            + 1 * Kernels_stride_w
        )
        k_val_1 = tl.load(k_ptr_1, mask=d_mask, other=0.0)

        # Load cache_val_2
        cache_ptr_2 = (
            Cache_ptr
            + pid_b * Cache_stride_b
            + offs_d * Cache_stride_d
            + 2 * Cache_stride_w
        )
        cache_val_2 = tl.load(cache_ptr_2, mask=d_mask, other=0.0)

        # Accumulate Part 1
        accumulator += cache_val_2 * k_val_1

        # Cache Update: Store cache_val_2 -> cache_ptr_1
        cache_ptr_1 = (
            Cache_ptr
            + pid_b * Cache_stride_b
            + offs_d * Cache_stride_d
            + 1 * Cache_stride_w
        )
        tl.store(cache_ptr_1, cache_val_2, mask=d_mask)

    # --- Step w = 2 ---
    # Compute: cache_val_3 * k_val_2 (part 1)
    # Cache Update: store cache_val_3 at index 2
    if tl.constexpr(W > 3):
        # Load k_val_2
        k_ptr_2 = (
            Kernels_ptr
            + pid_b * Kernels_stride_b
            + offs_d * Kernels_stride_d
            + 2 * Kernels_stride_w
        )
        k_val_2 = tl.load(k_ptr_2, mask=d_mask, other=0.0)

        # Load cache_val_3
        cache_ptr_3 = (
            Cache_ptr
            + pid_b * Cache_stride_b
            + offs_d * Cache_stride_d
            + 3 * Cache_stride_w
        )
        cache_val_3 = tl.load(cache_ptr_3, mask=d_mask, other=0.0)

        # Accumulate Part 1
        accumulator += cache_val_3 * k_val_2

        # Cache Update: Store cache_val_3 -> cache_ptr_2
        cache_ptr_2 = (
            Cache_ptr
            + pid_b * Cache_stride_b
            + offs_d * Cache_stride_d
            + 2 * Cache_stride_w
        )
        tl.store(cache_ptr_2, cache_val_3, mask=d_mask)

    # --- Final Step (Part 2 and Final Cache Update) ---
    # Compute: x_curr * k_val_{W-1} (part 2)
    # Cache Update: store x_curr at index W-1

    # Load k_val_{W-1}
    k_ptr_last = (
        Kernels_ptr
        + pid_b * Kernels_stride_b
        + offs_d * Kernels_stride_d
        + (W - 1) * Kernels_stride_w
    )
    k_val_last = tl.load(k_ptr_last, mask=d_mask, other=0.0)

    # Accumulate Part 2
    accumulator += x_curr * k_val_last

    # Final Cache Update: Store x_curr -> cache_ptr_{W-1}
    cache_ptr_last = (
        Cache_ptr
        + pid_b * Cache_stride_b
        + offs_d * Cache_stride_d
        + (W - 1) * Cache_stride_w
    )
    tl.store(cache_ptr_last, x_curr, mask=d_mask)

    # Removed activation application: accumulator = _apply_activation(accumulator, ACTIVATION)

    # 6. --- Store Output ---
    out_ptrs = Out_ptr + pid_b * Out_stride_b + offs_d * Out_stride_d
    tl.store(out_ptrs, accumulator, mask=d_mask)  # Store result without activation

    # Cache update is now fully handled within the unrolled steps.


# --- Python Wrapper Function ---
def causal_conv_step_triton(
    x: torch.Tensor,  # Input tensor [B, 1, D]
    cache: torch.Tensor,  # Cache tensor [B, D, W-1], modified in-place
    kernels: torch.Tensor,  # Kernels tensor [B, D, W]
    # Removed activation parameter
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
    # --- Input Validation and Preparation ---
    assert x.dim() == 3 and x.shape[1] == 1, "Input x must have shape [B, 1, D]"
    assert cache.dim() == 3, "Cache must have shape [B, D, W]"
    assert kernels.dim() == 3, "Kernels must have shape [B, D, W]"
    B, _, D = x.shape
    W = cache.shape[2]
    # Updated assertion: W must be > 1 and <= 4
    assert 1 < W <= 4, f"Kernel W={W}, this optimized version assumes 1 < W <= 4"
    assert (
        cache.shape[0] == B and cache.shape[1] == D
    ), f"Cache shape mismatch: {cache.shape}"
    assert kernels.shape == cache.shape, f"Kernels shape mismatch: {kernels.shape}"
    assert (
        x.is_cuda and cache.is_cuda and kernels.is_cuda
    ), "Inputs must be CUDA tensors"
    # Allow different input dtypes, but ensure they are compatible or handled
    # assert x.dtype == cache.dtype == kernels.dtype, "Input dtypes must match"

    # Squeeze the time dimension from input x
    x_squeezed = x.squeeze(1)  # Shape [B, D]

    # Ensure tensors are contiguous for correct stride calculations in Triton
    x_squeezed = ensure_contiguous(x_squeezed)
    # Cache MUST be contiguous for in-place updates and loads/stores to work reliably
    cache = ensure_contiguous(cache)
    kernels = ensure_contiguous(kernels)

    # Create output tensor with the same dtype as input x
    out = torch.empty_like(x_squeezed)  # Shape [B, D]

    # --- Triton Kernel Launch ---
    grid = lambda meta: (B, triton.cdiv(D, meta["BLOCK_SIZE_D"]))
    BLOCK_SIZE_D = 64  # Example, tune this value

    # Launch the kernel
    _causal_conv_step_kernel[grid](
        x_squeezed,
        cache,
        kernels,
        out,  # Tensor pointers
        B,
        D,  # Dimensions
        x_squeezed.stride(0),
        x_squeezed.stride(1),  # x strides
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),  # cache strides
        kernels.stride(0),
        kernels.stride(1),
        kernels.stride(2),  # kernels strides
        out.stride(0),
        out.stride(1),  # out strides
        # --- Meta-parameters ---
        W=W,  # Pass W as constexpr
        BLOCK_SIZE_D=BLOCK_SIZE_D,  # Pass BLOCK_SIZE_D as constexpr
        # Removed ACTIVATION=activation
    )

    return out  # Return the computed output [B, D] (before activation)
