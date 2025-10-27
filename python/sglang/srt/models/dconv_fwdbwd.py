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
from einops import rearrange
import torch.nn.functional as F
from torch.autograd import Function

# Helper function to ensure tensors are contiguous for Triton
def ensure_contiguous(t: torch.Tensor) -> torch.Tensor:
    return t if t.is_contiguous() else t.contiguous()

# --- Forward Kernel (from previous step, slight modifications for autograd context) ---
@triton.jit
def _dynamic_conv_fwd_kernel(
    X_ptr, K_ptr, Out_ptr,
    B, T, D,
    X_stride_b, X_stride_t, X_stride_d,
    K_stride_b, K_stride_t, K_stride_d, K_stride_w,
    Out_stride_b, Out_stride_t, Out_stride_d,
    W: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_batch_time = tl.program_id(0)
    pid_d_block = tl.program_id(1)

    batch_idx = tl.cast(pid_batch_time // T, tl.int64)
    time_idx = pid_batch_time % T

    offs_d = pid_d_block * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    d_mask = offs_d < D

    accumulator = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32)
    offs_w = tl.arange(0, W)

    # Load Kernels
    k_ptrs = K_ptr + (batch_idx * K_stride_b + time_idx * K_stride_t +
                      offs_d[:, None] * K_stride_d + offs_w[None, :] * K_stride_w)
    k_vals = tl.load(k_ptrs, mask=d_mask[:, None], other=0.0)

    # Load Input X with implicit padding
    t_in_offs = time_idx + offs_w - W + 1
    t_in_mask = (t_in_offs >= 0) & (t_in_offs < T)
    x_ptrs = X_ptr + (batch_idx * X_stride_b + t_in_offs[None, :] * X_stride_t +
                      offs_d[:, None] * X_stride_d)
    x_load_mask = d_mask[:, None] & t_in_mask[None, :]
    x_vals = tl.load(x_ptrs, mask=x_load_mask, other=0.0)

    # Compute and Accumulate
    product = k_vals * x_vals
    accumulator += tl.sum(product, axis=1)

    # Store Result
    out_ptrs = Out_ptr + (batch_idx * Out_stride_b + time_idx * Out_stride_t +
                          offs_d * Out_stride_d)
    tl.store(out_ptrs, accumulator, mask=d_mask)

# --- Backward Kernel for Input Gradient (dX) ---
@triton.jit
def _dynamic_conv_bwd_dx_kernel(
    GradOut_ptr, K_ptr, GradX_ptr, # Note: GradX is accumulated into
    B, T, D,
    GradOut_stride_b, GradOut_stride_t, GradOut_stride_d,
    K_stride_b, K_stride_t, K_stride_d, K_stride_w,
    GradX_stride_b, GradX_stride_t, GradX_stride_d,
    W: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """
    Computes gradient w.r.t. input X.
    Grid: (B * T, cdiv(D, BLOCK_SIZE_D)) - covering GradX output
    GradX[b, t_x, d] = sum_{w=0}^{W-1} GradOut[b, t, d] * K[b, t, d, w]
                       where t = t_x + W - 1 - w
    """
    pid_batch_time_x = tl.program_id(0) # Covers B * T for output GradX
    pid_d_block = tl.program_id(1)

    batch_idx = pid_batch_time_x // T
    time_idx_x = pid_batch_time_x % T # This is t_x

    offs_d = pid_d_block * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    d_mask = offs_d < D

    # Accumulator for GradX elements
    accumulator = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32)
    offs_w = tl.arange(0, W) # [W]

    # Loop over W to accumulate contributions
    # Calculate the 't' index needed for GradOut and K based on t_x and w
    # t = t_x + W - 1 - w
    t_k_gradout_offs = time_idx_x + W - 1 - offs_w # Shape [W]

    # Mask for valid 't' indices [0, T)
    t_k_gradout_mask = (t_k_gradout_offs >= 0) & (t_k_gradout_offs < T) # Shape [W]

    # --- Load GradOut ---
    # Pointers shape: [BLOCK_SIZE_D, W]
    gradout_ptrs = GradOut_ptr + (batch_idx * GradOut_stride_b +
                                  t_k_gradout_offs[None, :] * GradOut_stride_t +
                                  offs_d[:, None] * GradOut_stride_d)
    # Combined mask for loading GradOut (valid D and valid t)
    gradout_load_mask = d_mask[:, None] & t_k_gradout_mask[None, :]
    # Shape: [BLOCK_SIZE_D, W]
    gradout_vals = tl.load(gradout_ptrs, mask=gradout_load_mask, other=0.0)

    # --- Load Kernels ---
    # Pointers shape: [BLOCK_SIZE_D, W]
    k_ptrs = K_ptr + (batch_idx * K_stride_b +
                      t_k_gradout_offs[None, :] * K_stride_t +
                      offs_d[:, None] * K_stride_d +
                      offs_w[None, :] * K_stride_w) # Index K with 't' and 'w'
    # Combined mask for loading K (valid D and valid t)
    k_load_mask = d_mask[:, None] & t_k_gradout_mask[None, :]
    # Shape: [BLOCK_SIZE_D, W]
    k_vals = tl.load(k_ptrs, mask=k_load_mask, other=0.0)

    # --- Compute product and accumulate ---
    # Shape: [BLOCK_SIZE_D, W]
    product = gradout_vals * k_vals
    # Sum contributions over the W dimension
    accumulator += tl.sum(product, axis=1) # Shape: [BLOCK_SIZE_D]

    # --- Store accumulated gradients ---
    # Note: This kernel computes the *entire* gradient value for GradX[b, t_x, d_block].
    # If this kernel could potentially be called multiple times for the same GradX elements
    # (e.g., in complex graphs), atomic adds would be needed. Here, it seems direct store is fine.
    gradx_ptrs = GradX_ptr + (batch_idx * GradX_stride_b +
                              time_idx_x * GradX_stride_t +
                              offs_d * GradX_stride_d)
    tl.store(gradx_ptrs, accumulator, mask=d_mask)


# --- Backward Kernel for Kernel Gradient (dK) ---
@triton.jit
def _dynamic_conv_bwd_dk_kernel(
    GradOut_ptr, X_ptr, GradK_ptr, # Note: GradK is written directly
    B, T, D,
    GradOut_stride_b, GradOut_stride_t, GradOut_stride_d,
    X_stride_b, X_stride_t, X_stride_d,
    GradK_stride_b, GradK_stride_t, GradK_stride_d, GradK_stride_w,
    W: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """
    Computes gradient w.r.t. kernels K.
    Grid: (B * T, cdiv(D, BLOCK_SIZE_D)) - covering GradK output dims B, T, D
    GradK[b, t, d, w] = GradOut[b, t, d] * X[b, t + w - W + 1, d]
    """
    pid_batch_time = tl.program_id(0) # Covers B * T for output GradK
    pid_d_block = tl.program_id(1)

    batch_idx = pid_batch_time // T
    time_idx = pid_batch_time % T # This is 't' for GradK and GradOut

    offs_d = pid_d_block * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    d_mask = offs_d < D

    offs_w = tl.arange(0, W) # [W]

    # --- Load GradOut ---
    # Pointers shape: [BLOCK_SIZE_D] (only depends on b, t, d)
    gradout_ptrs = GradOut_ptr + (batch_idx * GradOut_stride_b +
                                  time_idx * GradOut_stride_t +
                                  offs_d * GradOut_stride_d)
    # Shape: [BLOCK_SIZE_D]
    gradout_vals = tl.load(gradout_ptrs, mask=d_mask, other=0.0)

    # --- Load Input X with implicit padding ---
    # Calculate X's time index: t_x = t + w - W + 1
    t_in_offs = time_idx + offs_w - W + 1 # Shape [W]
    # Mask for valid t_x index [0, T)
    t_in_mask = (t_in_offs >= 0) & (t_in_offs < T) # Shape [W]

    # Pointers shape: [BLOCK_SIZE_D, W]
    x_ptrs = X_ptr + (batch_idx * X_stride_b +
                      t_in_offs[None, :] * X_stride_t +
                      offs_d[:, None] * X_stride_d)
    # Combined mask for loading X (valid D and valid t_x)
    x_load_mask = d_mask[:, None] & t_in_mask[None, :] # Shape [BLOCK_SIZE_D, W]
    # Shape: [BLOCK_SIZE_D, W]
    x_vals = tl.load(x_ptrs, mask=x_load_mask, other=0.0)

    # --- Compute GradK = GradOut * X ---
    # Broadcast gradout_vals: [BLOCK_SIZE_D, 1] * [BLOCK_SIZE_D, W] -> [BLOCK_SIZE_D, W]
    gradk_vals = gradout_vals[:, None] * x_vals # Shape [BLOCK_SIZE_D, W]

    # --- Store gradients for Kernels ---
    # Pointers shape: [BLOCK_SIZE_D, W]
    gradk_ptrs = GradK_ptr + (batch_idx * GradK_stride_b +
                              time_idx * GradK_stride_t +
                              offs_d[:, None] * GradK_stride_d +
                              offs_w[None, :] * GradK_stride_w)
    # Mask only needed for D dimension (W is fully computed)
    # Store computed gradient values.
    tl.store(gradk_ptrs, gradk_vals, mask=d_mask[:, None])


# --- Autograd Function ---
class DynamicConvTritonFunc(Function):

    @staticmethod
    def forward(ctx, x, kernels):
        """
        Args:
            x: Input tensor [B, T, D]
            kernels: Kernels tensor [B, T, D, W]
        """
        x = ensure_contiguous(x)
        kernels = ensure_contiguous(kernels)

        B, T, D = x.shape
        W = kernels.shape[3]
        assert W <= 4, "Kernel W > 4 not expected for this version"

        out = torch.empty_like(x) # Output shape [B, T, D]

        grid = lambda meta: (B * T, triton.cdiv(D, meta['BLOCK_SIZE_D']))
        BLOCK_SIZE_D = 128 # Consider tuning

        _dynamic_conv_fwd_kernel[grid](
            x, kernels, out,
            B, T, D,
            x.stride(0), x.stride(1), x.stride(2),
            kernels.stride(0), kernels.stride(1), kernels.stride(2), kernels.stride(3),
            out.stride(0), out.stride(1), out.stride(2),
            W=W,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
        )

        # Save tensors needed for backward
        # Need x for dK, need kernels for dX
        ctx.save_for_backward(x, kernels)
        # Store W and BLOCK_SIZE_D needed for backward kernel calls
        ctx.W = W
        ctx.BLOCK_SIZE_D = BLOCK_SIZE_D

        return out

    @staticmethod
    def backward(ctx, grad_out):
        """
        Args:
            grad_out: Gradient w.r.t. the output tensor [B, T, D]
        Returns:
            grad_x: Gradient w.r.t. input x [B, T, D]
            grad_kernels: Gradient w.r.t. kernels [B, T, D, W]
        """
        grad_out = ensure_contiguous(grad_out)
        x, kernels = ctx.saved_tensors
        W = ctx.W
        BLOCK_SIZE_D = ctx.BLOCK_SIZE_D

        B, T, D = x.shape

        # Initialize gradients
        # grad_x needs accumulation, start with zeros.
        grad_x = torch.zeros_like(x)
        # grad_kernels is computed directly, can use empty_like if kernel handles all writes.
        # Using empty and relying on kernel writing zeros via masking/other=0.0.
        grad_kernels = torch.empty_like(kernels)

        # Define grid (can often be the same as forward or similar)
        grid = lambda meta: (B * T, triton.cdiv(D, meta['BLOCK_SIZE_D']))

        # Kernel call for grad_x
        _dynamic_conv_bwd_dx_kernel[grid](
            grad_out, kernels, grad_x,
            B, T, D,
            grad_out.stride(0), grad_out.stride(1), grad_out.stride(2),
            kernels.stride(0), kernels.stride(1), kernels.stride(2), kernels.stride(3),
            grad_x.stride(0), grad_x.stride(1), grad_x.stride(2),
            W=W,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
        )

        # Kernel call for grad_kernels
        _dynamic_conv_bwd_dk_kernel[grid](
            grad_out, x, grad_kernels,
            B, T, D,
            grad_out.stride(0), grad_out.stride(1), grad_out.stride(2),
            x.stride(0), x.stride(1), x.stride(2),
            grad_kernels.stride(0), grad_kernels.stride(1), grad_kernels.stride(2), grad_kernels.stride(3),
            W=W,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
        )

        # Return gradients in the order inputs were received by forward
        return grad_x, grad_kernels

# --- User-facing function ---
def dynamic_conv_triton_autograd(x: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
    """
    Fused dynamic convolution with autograd support using Triton kernels.
    Assumes W <= 4.

    Args:
        x: Input tensor of shape [B, T, D].
        kernels: Dynamic kernels of shape [B, T, D, W].

    Returns:
        Output tensor of shape [B, T, D].
    """
    return DynamicConvTritonFunc.apply(x, kernels)