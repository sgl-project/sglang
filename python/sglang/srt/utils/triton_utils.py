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
"""Triton availability and kernel-caching helpers."""

from __future__ import annotations

import functools
import inspect
import logging

import triton

from sglang.srt.environ import envs
from sglang.srt.platforms import current_platform
from sglang.srt.utils.device import is_cuda

logger = logging.getLogger(__name__)

__all__ = [
    "support_triton",
    "next_power_of_2",
    "CachedKernel",
    "cached_triton_kernel",
]


def support_triton(backend: str) -> bool:
    return current_platform.capabilities.supports_triton and backend not in [
        "torch_native",
        "intel_amx",
    ]


def next_power_of_2(n: int):
    return 1 << (n - 1).bit_length() if n > 0 else 1


setattr(triton, "next_power_of_2", next_power_of_2)


class CachedKernel:
    """
    Wrapper that allows kernel[grid](...) syntax with caching based on a key function.

    This wrapper caches compiled Triton kernels based on keys extracted by a
    user-provided key function to avoid redundant compilations.
    """

    def __init__(self, fn, key_fn=None):
        self.fn = fn
        assert isinstance(fn, triton.runtime.jit.JITFunction)

        original_fn = fn.fn
        self.signature = inspect.signature(original_fn)
        self.param_names = tuple(self.signature.parameters.keys())
        self.num_args = len(self.param_names)

        # Check that no parameters have default values
        for name, param in self.signature.parameters.items():
            assert param.default is inspect.Parameter.empty, (
                f"Parameter '{name}' has a default value. Default parameters are not supported in cached kernels."
            )

        functools.update_wrapper(self, original_fn)
        self.kernel_cache = {}

        # Store the key function
        self.key_fn = key_fn

    def __getitem__(self, grid):
        """
        Index with grid to get a launcher function.
        Returns a launcher that will handle caching based on the key function.
        """
        assert isinstance(grid, tuple) and len(grid) <= 3, (
            "Grid must be a tuple with at most 3 dimensions."
        )

        # Normalize grid once
        if len(grid) < 3:
            grid = grid + (1,) * (3 - len(grid))

        def launcher(*args, **kwargs):
            cache_key = self.key_fn(args, kwargs)

            cached_kernel = self.kernel_cache.get(cache_key)

            if cached_kernel is None:
                # First time: compile and cache the kernel
                cached_kernel = self.fn[grid](*args, **kwargs)
                self.kernel_cache[cache_key] = cached_kernel
                return cached_kernel
            else:
                # Use cached kernel
                all_args = self._build_args(args, kwargs)
                cached_kernel[grid](*all_args)
                return cached_kernel

        return launcher

    def _build_args(self, args, kwargs):
        """
        Build the complete argument list for kernel invocation.
        """
        complete_args = list(args)

        for i in range(len(args), self.num_args):
            name = self.param_names[i]
            value = kwargs.get(name, inspect.Parameter.empty)
            if value is not inspect.Parameter.empty:
                complete_args.append(value)
            else:
                raise ValueError(f"Missing argument: {name}")

        return complete_args

    def _clear_cache(self):
        """
        Clear the kernel cache for testing purposes.
        """
        self.kernel_cache.clear()


def cached_triton_kernel(key_fn=None):
    """
    Decorator that enables key-based caching for Triton kernels using a key function.

    It essentially bypasses Triton's built-in caching mechanism, allowing users to
    define their own caching strategy based on kernel parameters. This helps reduce
    the heavy overheads of Triton kernel launch when the kernel specialization dispatch
    is simple.

    Usage:
        @cached_triton_kernel(key_fn=lambda args, kwargs: kwargs.get('BLOCK_SIZE', 1024))
        @triton.jit
        def my_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
            ...

        # Invoke normally
        my_kernel[grid](x, y, BLOCK_SIZE=1024)

    Args:
        key_fn: A function that takes (args, kwargs) and returns the cache key(s).
                The key can be a single value or a tuple of values.

    Returns:
        A decorator that wraps the kernel with caching functionality.

    Note: Kernels with default parameter values are not supported and will raise an assertion error.
    """

    def decorator(fn):
        # Auto-enable the custom kernel cache for CUDA, where it is
        # known to be compatible.
        if is_cuda() and not envs.SGLANG_USE_CUSTOM_TRITON_KERNEL_CACHE.is_set():
            logger.debug("Detected platform CUDA, using custom triton kernel cache.")
            return CachedKernel(fn, key_fn)

        if envs.SGLANG_USE_CUSTOM_TRITON_KERNEL_CACHE.get():
            logger.debug(
                f"{envs.SGLANG_USE_CUSTOM_TRITON_KERNEL_CACHE.name} = True. Using custom triton kernel cache."
            )
            return CachedKernel(fn, key_fn)
        else:
            # Fallback to the native triton cache.
            logger.debug(
                f"{envs.SGLANG_USE_CUSTOM_TRITON_KERNEL_CACHE.name} = False. Using native triton kernel cache."
            )
            return fn

    return decorator
