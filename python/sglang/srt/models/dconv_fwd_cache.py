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

# --- Forward Kernel (modified for optional cache) ---
@triton.jit
def _dynamic_conv_fwd_kernel(
    X_ptr, K_ptr, Out_ptr,
    Cache_ptr, # New: Pointer to cache tensor
    B, T, D, T_CACHE: tl.constexpr, # New: T is shape of x, T_CACHE is shape of cache
    X_stride_b, X_stride_t, X_stride_d,
    K_stride_b, K_stride_t, K_stride_d, K_stride_w,
    Out_stride_b, Out_stride_t, Out_stride_d,
    Cache_stride_b, Cache_stride_t, Cache_stride_d, # New: Strides for cache tensor
    W: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_batch_time = tl.program_id(0) # Covers B * T_out
    pid_d_block = tl.program_id(1)

    # T here is the time dimension of x and Out
    batch_idx = tl.cast(pid_batch_time // T, tl.int64)
    time_idx = pid_batch_time % T # Current output time step for x (0 to T-1)

    offs_d = pid_d_block * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    d_mask = offs_d < D

    accumulator = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32)
    offs_w = tl.arange(0, W) # Kernel window offsets [0, 1, ..., W-1]

    # Load Kernels (kernels are aligned with x's T dimension)
    # K_ptr is indexed by time_idx which is the output time relative to x's start
    k_ptrs = K_ptr + (batch_idx * K_stride_b + time_idx * K_stride_t +
                      offs_d[:, None] * K_stride_d + offs_w[None, :] * K_stride_w)
    k_vals = tl.load(k_ptrs, mask=d_mask[:, None], other=0.0) # Shape: [BLOCK_SIZE_D, W]

    # --- Load Input from conceptual [Cache, X] tensor ---
    # `time_idx` is the current output time step (0 to T-1, where T is x.shape[1])
    # `offs_w` is [0, ..., W-1]
    # Convolution input time indices relative to the *start of x*:
    # e.g., for W=3, offs_w - W + 1 gives [-2, -1, 0]
    # so input_time_indices_rel_to_x_start are [time_idx-2, time_idx-1, time_idx]
    input_time_indices_rel_to_x_start = time_idx + offs_w - W + 1 # Shape: [W]

    # Effective input time indices in the conceptual [Cache, X] sequence:
    # These indices range from 0 (start of cache) to T_CACHE + T - 1 (end of x)
    eff_t_indices = input_time_indices_rel_to_x_start + T_CACHE # Shape: [W]

    # Overall mask for valid time indices within the conceptual [Cache, X] tensor
    # Total effective length is T_CACHE (for cache) + T (for x)
    eff_t_valid_mask = (eff_t_indices >= 0) & (eff_t_indices < (T_CACHE + T)) # Shape: [W]

    # --- Load from Cache ---
    # Condition for loading from cache: index is valid AND index < T_CACHE
    # (eff_t_indices are 0-indexed from the start of the cache)
    cache_load_time_mask = eff_t_valid_mask & (eff_t_indices < T_CACHE) # Shape: [W]
    cache_ptr_indices = eff_t_indices # Use directly if in cache range

    cache_ptrs = Cache_ptr + (batch_idx * Cache_stride_b +
                              cache_ptr_indices[None, :] * Cache_stride_t +
                              offs_d[:, None] * Cache_stride_d)
    cache_final_load_mask = d_mask[:, None] & cache_load_time_mask[None, :] # Shape: [BLOCK_SIZE_D, W]
    vals_from_cache = tl.load(cache_ptrs, mask=cache_final_load_mask, other=0.0) # Shape: [BLOCK_SIZE_D, W]

    # --- Load from X ---
    # Condition for loading from X: index is valid AND index >= T_CACHE
    x_load_time_mask = eff_t_valid_mask & (eff_t_indices >= T_CACHE) # Shape: [W]
    # Adjust indices for X_ptr: X_ptr expects indices from 0 to T-1 (relative to start of x)
    x_ptr_indices = eff_t_indices - T_CACHE # Shape: [W]

    x_ptrs = X_ptr + (batch_idx * X_stride_b +
                      x_ptr_indices[None, :] * X_stride_t +
                      offs_d[:, None] * X_stride_d)
    x_final_load_mask = d_mask[:, None] & x_load_time_mask[None, :] # Shape: [BLOCK_SIZE_D, W]
    vals_from_x = tl.load(x_ptrs, mask=x_final_load_mask, other=0.0) # Shape: [BLOCK_SIZE_D, W]

    # Combine values. Masks ensure only one source contributes non-zero per element.
    # If T_CACHE == 0, cache_load_time_mask is all False, so vals_from_cache is 0.0.
    x_input_vals = vals_from_cache + vals_from_x # Shape: [BLOCK_SIZE_D, W]

    # Compute and Accumulate
    product = k_vals * x_input_vals # Element-wise product
    accumulator += tl.sum(product, axis=1) # Sum over W dimension

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
    def forward(ctx, x, kernels, cache=None): # Added cache argument
        """
        Args:
            x: Input tensor [B, T, D]
            kernels: Kernels tensor [B, T, D, W]
            cache: Optional past context tensor [B, T_cache, D]
        """
        x = ensure_contiguous(x)
        kernels = ensure_contiguous(kernels)

        B, T, D = x.shape # T is the time dimension of the current input x
        _B_k, _T_k, _D_k, W = kernels.shape # Kernels are [B, T_x, D, W]
        assert B == _B_k and T == _T_k and D == _D_k, \
            f"Shape mismatch between x ({x.shape}) and kernels ({kernels.shape}) on B, T, or D dims"
        assert W <= 4, "Kernel W > 4 not expected for this version"

        out = torch.empty_like(x) # Output shape [B, T, D], corresponds to x

        T_cache_val = 0
        # Use x's data pointer and zero strides as placeholders if cache is None.
        # These won't be used by the kernel if T_CACHE_VAL is 0 due to masking.
        cache_ptr_val = x 
        cache_s_b, cache_s_t, cache_s_d = 0, 0, 0

        if cache is not None:
            cache = ensure_contiguous(cache)
            B_c, T_c, D_c = cache.shape
            assert B_c == B, f"Batch size mismatch: x ({B}) vs cache ({B_c})"
            assert D_c == D, f"Dimension mismatch: x ({D}) vs cache ({D_c})"
            T_cache_val = T_c
            cache_ptr_val = cache
            cache_s_b, cache_s_t, cache_s_d = cache.stride(0), cache.stride(1), cache.stride(2)

        grid = lambda meta: (B * T, triton.cdiv(D, meta['BLOCK_SIZE_D']))
        BLOCK_SIZE_D = 128 # Consider tuning

        _dynamic_conv_fwd_kernel[grid](
            x, kernels, out,                                                           # X, K, Out pointers
            cache_ptr_val,                                                              # Cache pointer
            B, T, D, T_cache_val,                                                       # Shapes: B, T_x, D, T_cache
            x.stride(0), x.stride(1), x.stride(2),                                      # X strides
            kernels.stride(0), kernels.stride(1), kernels.stride(2), kernels.stride(3), # K strides
            out.stride(0), out.stride(1), out.stride(2),                                # Out strides
            cache_s_b, cache_s_t, cache_s_d,                                            # Cache strides
            W=W,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
        )

        # Save tensors needed for backward (cache is not needed for current backward)
        ctx.save_for_backward(x, kernels)
        ctx.W = W
        ctx.BLOCK_SIZE_D = BLOCK_SIZE_D
        # ctx.T_cache = T_cache_val # Not needed for current backward

        return out

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError("Backward of cached fwdbwd is not implemented")


# --- User-facing function ---
def dynamic_conv_triton_cache(x: torch.Tensor, kernels: torch.Tensor, cache: torch.Tensor = None) -> torch.Tensor:
    """
    Fused dynamic convolution with autograd support using Triton kernels.
    Assumes W <= 4.

    Args:
        x: Input tensor of shape [B, T, D].
        kernels: Dynamic kernels of shape [B, T, D, W].
        cache: Optional past context tensor of shape [B, T_cache, D].
               If provided, treated as concatenated before x for convolution input.

    Returns:
        Output tensor of shape [B, T, D].
    """
    return DynamicConvTritonFunc.apply(x, kernels, cache)