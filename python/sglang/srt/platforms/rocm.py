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
AMD ROCm/HIP Platform Implementation
=====================================

This platform supports AMD GPUs via the ROCm (Radeon Open Compute) stack.

KEY DIFFERENCES FROM CUDA:
--------------------------
1. device_type = "cuda" - ROCm uses PyTorch's CUDA device type, but device_name
   is "rocm" to indicate the underlying AMD ROCm platform.

2. Detection quirk - ROCm sets torch.cuda.is_available() = True, so we must
   check torch.version.hip to distinguish from real NVIDIA CUDA.

3. Same sgl_kernel interface - ROCm uses the same sgl_kernel library, which
   provides HIP-compiled versions of the same kernels.

PLATFORM-SPECIFIC MODULES:
--------------------------
This platform provides access to ROCm-specific functions via PlatformModules:
    current_platform.modules.gelu_quick(x, out)  # Only on ROCm

The _get_module_attr() method provides access to sgl_kernel functions that
may be ROCm-specific (e.g., gelu_quick for fast GELU on HIP).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from sglang.srt.platforms.interface import Platform, PlatformEnum

if TYPE_CHECKING:
    from sglang.srt.ops.base import OpProxy


class RocmPlatform(Platform):
    """AMD ROCm/HIP platform implementation.

    Supports AMD GPUs via the ROCm stack. Uses sgl_kernel with HIP backend.
    """

    _enum = PlatformEnum.ROCM
    device_name = "rocm"  # Human-readable name
    device_type = "cuda"  # ROCm uses CUDA device type in PyTorch!

    # Op registry - lazily initialized on first access
    _ops: dict[str, Callable] | None = None

    # Cache for sgl_kernel attributes (for PlatformModules)
    _sgl_kernel_cache: dict[str, Any] | None = None

    @classmethod
    def _get_sgl_kernel_attr(cls, name: str) -> Any | None:
        """Lazy load and cache sgl_kernel attributes.

        This helper is used by _get_module_attr to provide access to
        ROCm-specific functions like gelu_quick via PlatformModules.
        """
        if cls._sgl_kernel_cache is None:
            cls._sgl_kernel_cache = {}

        if name not in cls._sgl_kernel_cache:
            try:
                import sgl_kernel

                if hasattr(sgl_kernel, name):
                    cls._sgl_kernel_cache[name] = getattr(sgl_kernel, name)
                else:
                    return None
            except ImportError:
                return None

        return cls._sgl_kernel_cache.get(name)

    def _get_module_attr(self, name: str) -> Any | None:
        """Get ROCm-specific module attributes.

        This enables PlatformModules to expose ROCm-specific functions:
            current_platform.modules.gelu_quick(x, out)

        Available attributes (from sgl_kernel):
        - gelu_quick: Fast approximate GELU for HIP
        - Other HIP-specific kernels as needed
        """
        return self._get_sgl_kernel_attr(name)

    @classmethod
    def _init_ops(cls) -> dict[str, Callable]:
        """Initialize the op registry with HIP kernels.

        sgl_kernel provides HIP-compiled versions of the same kernels
        used on CUDA. The API is identical.
        """
        from sgl_kernel import gelu_and_mul as hip_gelu_and_mul
        from sgl_kernel import gelu_tanh_and_mul as hip_gelu_tanh_and_mul
        from sgl_kernel import silu_and_mul as hip_silu_and_mul

        return {
            "silu_and_mul": hip_silu_and_mul,
            "gelu_and_mul": hip_gelu_and_mul,
            "gelu_tanh_and_mul": hip_gelu_tanh_and_mul,
        }

    def get_op(self, op: "OpProxy") -> Callable | None:
        """Get the ROCm/HIP implementation of an operation."""
        if RocmPlatform._ops is None:
            RocmPlatform._ops = self._init_ops()
        return RocmPlatform._ops.get(op.name)
