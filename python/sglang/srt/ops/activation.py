# Copyright 2023-2024 SGLang Team
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
"""
Activation Operations - Reference Implementation
=================================================

This file demonstrates **RFC Feature #3: Platform-specific Op / Kernel Resolution**.

DESIGN PATTERN:
---------------
Each operation in this module follows a three-layer design:

    1. Public API (silu_and_mul, gelu_and_mul, etc.)
       - Full docstrings for IDE hover/autocomplete
       - Full type hints for static analysis
       - This is what users import and use

    2. Platform Dispatch (_get_impl)
       - Queries current_platform for the optimized implementation
       - Caches the result to avoid repeated lookups
       - Transparent to users

    3. Native Fallback (_silu_and_mul_native, etc.)
       - Pure PyTorch implementation
       - Used when no platform-specific kernel is available
       - Ensures code works on any device, just slower

HOW IT WORKS:
-------------
When you call silu_and_mul(x, out):

    1. First call: _get_impl("silu_and_mul", fallback) is invoked
       - Imports current_platform (lazy, avoids circular imports)
       - Calls current_platform.get_op_by_name("silu_and_mul")
       - Platform returns sgl_kernel.silu_and_mul (on CUDA)
         or torch_npu.npu_swiglu wrapper (on NPU)
         or None (on unsupported platform → use fallback)
       - Caches the result in _impl_cache

    2. Subsequent calls: Returns cached implementation immediately

    3. The implementation (CUDA kernel, NPU kernel, or fallback) is called

This achieves:
- Zero runtime overhead after first call (cached)
- No import-time platform checks (lazy)
- No if-else scattered in model code
- Full IDE support (docstrings, type hints)

COMPARISON TO OLD APPROACH:
---------------------------
OLD (scattered checks):
    _is_cuda = is_cuda()
    _is_npu = is_npu()
    if _is_cuda:
        from sgl_kernel import silu_and_mul
    # ...must handle every platform at import time

NEW (platform dispatch):
    from sglang.srt.ops import silu_and_mul
    silu_and_mul(x, out)  # Just works on any platform!
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# =============================================================================
# Native Fallback Implementations
# =============================================================================
# These pure-PyTorch implementations ensure ops work on any device.
# They're used when no platform-specific kernel is registered.
#
# NOTE: Platform-specific kernels (sgl_kernel, torch_npu) are NOT imported here.
# They are registered in each platform's _init_ops() method and looked up
# at runtime via current_platform.get_op_by_name().


def _silu_and_mul_native(x: torch.Tensor, out: torch.Tensor) -> None:
    """Native PyTorch fallback for silu_and_mul.

    This is used when no optimized kernel is available for the current platform.
    Equivalent to: out = F.silu(x[..., :d]) * x[..., d:]
    """
    d = x.shape[-1] // 2
    out.copy_(F.silu(x[..., :d]) * x[..., d:])


def _gelu_and_mul_native(x: torch.Tensor, out: torch.Tensor) -> None:
    """Native PyTorch fallback for gelu_and_mul.

    Uses exact GELU (not tanh approximation).
    """
    d = x.shape[-1] // 2
    out.copy_(F.gelu(x[..., :d], approximate="none") * x[..., d:])


def _gelu_tanh_and_mul_native(x: torch.Tensor, out: torch.Tensor) -> None:
    """Native PyTorch fallback for gelu_tanh_and_mul.

    Uses tanh-approximated GELU for faster computation.
    """
    d = x.shape[-1] // 2
    out.copy_(F.gelu(x[..., :d], approximate="tanh") * x[..., d:])


# =============================================================================
# Op Implementations Cache - The Core of Platform Dispatch
# =============================================================================
#
# This is where the magic happens. The _impl_cache stores resolved implementations
# so we only do platform lookup once per op, not on every call.
#
# Flow on first call to silu_and_mul(x, out):
#   1. Check if "silu_and_mul" is in _impl_cache → No
#   2. Import current_platform (lazy import avoids circular deps)
#   3. Call current_platform.get_op_by_name("silu_and_mul")
#   4. Platform returns registered kernel or None
#   5. Cache the result (kernel or fallback)
#   6. Return cached implementation
#
# Flow on subsequent calls:
#   1. Check if "silu_and_mul" is in _impl_cache → Yes
#   2. Return cached implementation (zero overhead!)

_impl_cache: dict[str, callable] = {}


def _get_impl(op_name: str, fallback: callable) -> callable:
    """Get the platform-specific implementation, with caching.

    This function is the bridge between the user-facing op functions
    and the platform-specific implementations registered in Platform classes.

    Args:
        op_name: Canonical name of the operation (e.g., "silu_and_mul").
                 This must match the key in Platform._ops registry.
        fallback: Native PyTorch implementation to use if no platform kernel exists.

    Returns:
        The cached implementation (platform kernel or fallback).

    Note:
        The import of current_platform is inside this function (not at module level)
        to avoid circular imports. This is safe because _get_impl is only called
        when an op is actually invoked, not at import time.
    """
    if op_name not in _impl_cache:
        # Lazy import to avoid circular dependency:
        # ops.activation → platforms → ops.base → ...
        from sglang.srt.platforms import current_platform

        # Ask the current platform for its implementation
        # Each platform registers ops in _init_ops() (see platforms/cuda.py, etc.)
        impl = current_platform.get_op_by_name(op_name)

        if impl is not None:
            # Platform has an optimized kernel - use it!
            _impl_cache[op_name] = impl
        else:
            # No platform kernel available - fall back to pure PyTorch
            _impl_cache[op_name] = fallback

    return _impl_cache[op_name]


# =============================================================================
# Public API: Activation Operations
# =============================================================================
#
# Each function below is the user-facing API. They have:
# - Complete docstrings (visible in IDE hover)
# - Full type hints (for IDE autocompletion and static analysis)
# - Consistent signature with platform kernels
#
# Users import these and use them directly:
#     from sglang.srt.ops import silu_and_mul
#     silu_and_mul(x, out)  # Works on any platform!


def silu_and_mul(x: torch.Tensor, out: torch.Tensor) -> None:
    """Fused SiLU activation and element-wise multiply.

    Computes: ``out = silu(x[..., :d]) * x[..., d:]`` where ``d = x.shape[-1] // 2``

    This is a fused kernel that combines the SiLU (Swish) activation with
    element-wise multiplication, commonly used in SwiGLU-based feed-forward
    networks (e.g., LLaMA, Mistral, DeepSeek).

    The fused implementation avoids materializing intermediate tensors,
    reducing memory bandwidth and improving performance.

    Args:
        x: Input tensor of shape ``(..., 2*d)`` where the last dimension
            will be split in half. The first half goes through SiLU activation.
        out: Output tensor of shape ``(..., d)``, will be written in-place.
            Must be pre-allocated with the correct shape and dtype.

    Example:
        >>> x = torch.randn(batch_size, seq_len, 2 * hidden_dim, device="cuda")
        >>> out = torch.empty(batch_size, seq_len, hidden_dim, device="cuda")
        >>> silu_and_mul(x, out)
    """
    impl = _get_impl("silu_and_mul", _silu_and_mul_native)
    impl(x, out)


def gelu_and_mul(x: torch.Tensor, out: torch.Tensor) -> None:
    """Fused GELU activation and element-wise multiply.

    Computes: ``out = gelu(x[..., :d]) * x[..., d:]`` where ``d = x.shape[-1] // 2``

    Uses the exact GELU activation (not the tanh approximation).
    For the tanh-approximated version, use :func:`gelu_tanh_and_mul`.

    This is commonly used in GEGLU-based feed-forward networks.

    Args:
        x: Input tensor of shape ``(..., 2*d)`` where the last dimension
            will be split in half. The first half goes through GELU activation.
        out: Output tensor of shape ``(..., d)``, will be written in-place.
            Must be pre-allocated with the correct shape and dtype.

    Example:
        >>> x = torch.randn(batch_size, seq_len, 2 * hidden_dim, device="cuda")
        >>> out = torch.empty(batch_size, seq_len, hidden_dim, device="cuda")
        >>> gelu_and_mul(x, out)
    """
    impl = _get_impl("gelu_and_mul", _gelu_and_mul_native)
    impl(x, out)


def gelu_tanh_and_mul(x: torch.Tensor, out: torch.Tensor) -> None:
    """Fused GELU (tanh approximation) activation and element-wise multiply.

    Computes: ``out = gelu_tanh(x[..., :d]) * x[..., d:]`` where ``d = x.shape[-1] // 2``

    Uses the tanh approximation of GELU for faster computation:
    ``gelu_tanh(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))``

    This is faster than exact GELU but slightly less accurate.
    For the exact version, use :func:`gelu_and_mul`.

    Args:
        x: Input tensor of shape ``(..., 2*d)`` where the last dimension
            will be split in half. The first half goes through GELU activation.
        out: Output tensor of shape ``(..., d)``, will be written in-place.
            Must be pre-allocated with the correct shape and dtype.

    Example:
        >>> x = torch.randn(batch_size, seq_len, 2 * hidden_dim, device="cuda")
        >>> out = torch.empty(batch_size, seq_len, hidden_dim, device="cuda")
        >>> gelu_tanh_and_mul(x, out)
    """
    impl = _get_impl("gelu_tanh_and_mul", _gelu_tanh_and_mul_native)
    impl(x, out)


__all__ = [
    "silu_and_mul",
    "gelu_and_mul",
    "gelu_tanh_and_mul",
]
