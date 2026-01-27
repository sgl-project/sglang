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
Intel XPU Platform Implementation
==================================

This platform supports Intel GPUs (Arc, Data Center GPUs) via Intel's oneAPI
and SYCL-based XPU stack.

XPU-SPECIFIC CONSIDERATIONS:
----------------------------
1. device_type = "xpu" - Intel XPU has its own device namespace (torch.xpu)
2. Detection via torch.xpu.is_available()
3. sgl_kernel may use SYCL-based kernels for XPU

Unlike MUSA and ROCm, XPU does NOT set torch.cuda.is_available() = True,
so detection order relative to CUDA doesn't matter. However, we check
XPU after CUDA for consistency with the general priority order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from sglang.srt.platforms.interface import Platform, PlatformEnum

if TYPE_CHECKING:
    from sglang.srt.ops.base import OpProxy


class XpuPlatform(Platform):
    """Intel XPU platform implementation.

    Supports Intel GPUs (Arc, Data Center GPUs) via oneAPI/SYCL.
    """

    _enum = PlatformEnum.XPU
    device_name = "xpu"
    device_type = "xpu"

    # Op registry - lazily initialized on first access
    _ops: dict[str, Callable] | None = None

    @classmethod
    def _init_ops(cls) -> dict[str, Callable]:
        """Initialize the op registry with XPU kernels.

        sgl_kernel provides XPU-compiled versions of the same kernels
        used on CUDA. The API is identical.
        """
        from sgl_kernel import gelu_and_mul as xpu_gelu_and_mul
        from sgl_kernel import gelu_tanh_and_mul as xpu_gelu_tanh_and_mul
        from sgl_kernel import silu_and_mul as xpu_silu_and_mul

        return {
            "silu_and_mul": xpu_silu_and_mul,
            "gelu_and_mul": xpu_gelu_and_mul,
            "gelu_tanh_and_mul": xpu_gelu_tanh_and_mul,
        }

    def get_op(self, op: "OpProxy") -> Callable | None:
        """Get the XPU implementation of an operation."""
        if XpuPlatform._ops is None:
            XpuPlatform._ops = self._init_ops()
        return XpuPlatform._ops.get(op.name)
