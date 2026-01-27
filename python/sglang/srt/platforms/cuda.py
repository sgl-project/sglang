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
NVIDIA CUDA Platform Implementation
====================================

This file demonstrates the standard pattern for implementing a Platform.
Use this as a reference when adding support for new hardware vendors.

THE OP REGISTRY PATTERN:
------------------------
Each platform maintains a lazy-loaded dictionary (_ops) that maps operation
names (strings) to their platform-specific implementations (callables).

    _ops = {
        "silu_and_mul": sgl_kernel.silu_and_mul,  # CUDA kernel
        "gelu_and_mul": sgl_kernel.gelu_and_mul,  # CUDA kernel
        ...
    }

The registry is populated on first access via _init_ops(), which:
1. Imports the platform-specific kernel library (sgl_kernel for CUDA)
2. Returns a dict mapping op names to kernel functions

WHY LAZY LOADING?
-----------------
Kernel imports can be expensive (loading .so files, GPU initialization).
By deferring _init_ops() until an op is actually requested:
- Faster startup for code that doesn't use ops
- No import errors if the kernel library isn't installed

ADDING A NEW OP:
----------------
1. Add the op to _init_ops():
       from sgl_kernel import my_new_op as cuda_my_new_op
       return {
           ...existing ops...,
           "my_new_op": cuda_my_new_op,
       }

2. Define the user-facing function in sglang.srt.ops:
       def my_new_op(x: torch.Tensor, out: torch.Tensor) -> None:
           impl = _get_impl("my_new_op", _my_new_op_fallback)
           impl(x, out)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from sglang.srt.platforms.interface import Platform, PlatformEnum

if TYPE_CHECKING:
    from sglang.srt.ops.base import OpProxy


class CudaPlatform(Platform):
    """NVIDIA CUDA platform implementation.

    This is the most common platform. sgl_kernel provides optimized CUDA
    kernels for activation ops, attention, and other compute-intensive ops.
    """

    _enum = PlatformEnum.CUDA
    device_name = "cuda"
    device_type = "cuda"

    # Op registry - lazily initialized on first access
    # Maps operation names (str) → implementations (Callable)
    _ops: dict[str, Callable] | None = None

    @classmethod
    def _init_ops(cls) -> dict[str, Callable]:
        """Initialize the op registry with CUDA kernels.

        This method is called once, on the first op lookup.
        All sgl_kernel imports happen here to keep startup fast.

        Returns:
            Dict mapping op names to sgl_kernel functions.
        """
        # Import CUDA kernels from sgl_kernel
        # These are compiled CUDA extensions providing optimized implementations
        from sgl_kernel import gelu_and_mul as cuda_gelu_and_mul
        from sgl_kernel import gelu_tanh_and_mul as cuda_gelu_tanh_and_mul
        from sgl_kernel import silu_and_mul as cuda_silu_and_mul

        return {
            # Activation ops - used in feed-forward networks (SwiGLU, GEGLU)
            "silu_and_mul": cuda_silu_and_mul,
            "gelu_and_mul": cuda_gelu_and_mul,
            "gelu_tanh_and_mul": cuda_gelu_tanh_and_mul,
            # Add more ops here as needed:
            # "my_new_op": cuda_my_new_op,
        }

    def get_op(self, op: "OpProxy") -> Callable | None:
        """Get the CUDA implementation of an operation.

        Args:
            op: The OpProxy representing the operation to look up.

        Returns:
            The CUDA kernel function, or None if not registered.
        """
        if CudaPlatform._ops is None:
            CudaPlatform._ops = self._init_ops()
        return CudaPlatform._ops.get(op.name)
