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
Moore Threads MUSA Platform Implementation
===========================================

This platform supports Moore Threads GPUs via the MUSA (Moore Threads Unified
System Architecture) stack.

CRITICAL: DETECTION ORDER
-------------------------
MUSA MUST be detected BEFORE CUDA because:
1. MUSA sets torch.cuda.is_available() = True (for compatibility)
2. But MUSA is NOT NVIDIA CUDA - it requires different kernels
3. Detection uses torch.musa.is_available() to distinguish

If MUSA detection came after CUDA, MUSA devices would be misidentified
as CUDA devices, leading to crashes or incorrect behavior.

MUSA-SPECIFIC CONSIDERATIONS:
-----------------------------
1. device_type = "musa" - Unlike ROCm, MUSA has its own device type
2. sgl_kernel is built with MUSA support (via torch_musa extension system)
3. MUSA kernels may have different optimization patterns than CUDA

The sgl_kernel library provides MUSA-specific implementations that are
compatible with the same API as CUDA kernels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from sglang.srt.platforms.interface import Platform, PlatformEnum

if TYPE_CHECKING:
    from sglang.srt.ops.base import OpProxy


class MusaPlatform(Platform):
    """Moore Threads MUSA platform implementation.

    Supports Moore Threads GPUs via the MUSA stack. Uses sgl_kernel with
    MUSA backend (built via torch_musa extension system).
    """

    _enum = PlatformEnum.MUSA
    device_name = "musa"
    device_type = "musa"  # MUSA has its own device type (unlike ROCm)

    # Op registry - lazily initialized on first access
    _ops: dict[str, Callable] | None = None

    @classmethod
    def _init_ops(cls) -> dict[str, Callable]:
        """Initialize the op registry with MUSA kernels.

        sgl_kernel provides MUSA-compiled versions of the same kernels
        used on CUDA. The API is identical.

        Note: sgl_kernel must be built with MUSA support via torch_musa.
        """
        from sgl_kernel import gelu_and_mul as musa_gelu_and_mul
        from sgl_kernel import gelu_tanh_and_mul as musa_gelu_tanh_and_mul
        from sgl_kernel import silu_and_mul as musa_silu_and_mul

        return {
            "silu_and_mul": musa_silu_and_mul,
            "gelu_and_mul": musa_gelu_and_mul,
            "gelu_tanh_and_mul": musa_gelu_tanh_and_mul,
        }

    def get_op(self, op: "OpProxy") -> Callable | None:
        """Get the MUSA implementation of an operation."""
        if MusaPlatform._ops is None:
            MusaPlatform._ops = self._init_ops()
        return MusaPlatform._ops.get(op.name)
