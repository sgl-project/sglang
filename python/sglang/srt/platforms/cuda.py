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

THE OP REGISTRY PATTERN (with Lazy Imports):
---------------------------------------------
Each platform maintains a lazy-loaded dictionary (_ops) that maps operation
names (strings) to OpSpec instances for lazy imports:

    _ops = {
        "silu_and_mul": OpSpec("sgl_kernel", "silu_and_mul"),
        "gelu_and_mul": OpSpec("sgl_kernel", "gelu_and_mul"),
        ...
    }

The registry is populated on first access via _init_ops(), which returns
a dict mapping op names to OpSpec instances. The actual kernel imports
happen only when an op is first called (via OpSpec.resolve()).

WHY LAZY IMPORTS (OpSpec)?
--------------------------
1. Faster startup - No kernel imports until an op is actually used
2. No import errors - If sgl_kernel isn't installed, no error until op is called
3. Graceful fallback - If import fails, the op falls back to native PyTorch

ADDING A NEW OP:
----------------
1. Add the op to _init_ops():
       return {
           ...existing ops...,
           "my_new_op": OpSpec("sgl_kernel", "my_new_op"),
       }

2. Define the user-facing function in sglang.srt.ops:
       def my_new_op(x: torch.Tensor, out: torch.Tensor) -> None:
           impl = _get_impl("my_new_op", _my_new_op_fallback)
           impl(x, out)

OUT-OF-CLASS REGISTRATION:
--------------------------
Ops can also be registered from outside this file using register_op():

    from sglang.srt.platforms.interface import register_op, OpSpec, PlatformEnum
    register_op(PlatformEnum.CUDA, "custom_op", OpSpec("my_lib", "custom_op"))
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from sglang.srt.platforms.interface import OpSpec, Platform, PlatformEnum

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
    # Maps operation names (str) → OpSpec for lazy imports
    _ops: dict[str, OpSpec | Callable] | None = None

    @classmethod
    def _init_ops(cls) -> dict[str, OpSpec | Callable]:
        """Initialize the op registry with CUDA kernel specs.

        This method is called once, on the first op lookup.
        Returns OpSpec instances for lazy imports - the actual sgl_kernel
        imports happen only when an op is first called.

        Returns:
            Dict mapping op names to OpSpec instances.
        """
        return {
            # Activation ops - used in feed-forward networks (SwiGLU, GEGLU)
            # OpSpec defers the import until the op is actually called
            "silu_and_mul": OpSpec("sgl_kernel", "silu_and_mul"),
            "gelu_and_mul": OpSpec("sgl_kernel", "gelu_and_mul"),
            "gelu_tanh_and_mul": OpSpec("sgl_kernel", "gelu_tanh_and_mul"),
            # Add more ops here as needed:
            # "my_new_op": OpSpec("sgl_kernel", "my_new_op"),
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

        impl = CudaPlatform._ops.get(op.name)
        if impl is None:
            return None

        # Resolve OpSpec to actual callable if needed
        if isinstance(impl, OpSpec):
            try:
                resolved = impl.resolve()
                # Cache the resolved callable for future calls
                CudaPlatform._ops[op.name] = resolved
                return resolved
            except (ImportError, AttributeError):
                return None

        return impl
