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
"""CPU platform implementation."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Callable

from sglang.srt.platforms.interface import Platform, PlatformEnum

if TYPE_CHECKING:
    from sglang.srt.ops.base import OpProxy
    from sglang.srt.server_args import ServerArgs


class CpuPlatform(Platform):
    """CPU platform implementation."""

    _enum = PlatformEnum.CPU
    device_name = "cpu"
    device_type = "cpu"

    # Lazy-loaded op registry
    _ops: dict[str, Callable] | None = None
    _has_amx: bool | None = None

    @staticmethod
    @lru_cache(maxsize=1)
    def _check_amx_support() -> bool:
        """Check if CPU has AMX support."""
        from sglang.srt.utils import cpu_has_amx_support

        return cpu_has_amx_support()

    @property
    def has_amx(self) -> bool:
        """Check if this CPU has AMX (Advanced Matrix Extensions) support.

        AMX provides hardware acceleration for matrix operations and is
        used to enable optimized kernels on compatible CPUs.

        Returns:
            True if AMX is supported, False otherwise.
        """
        if CpuPlatform._has_amx is None:
            CpuPlatform._has_amx = self._check_amx_support()
        return CpuPlatform._has_amx

    @classmethod
    def _init_ops(cls) -> dict[str, Callable]:
        """Initialize op registry. Called once on first op access."""
        import torch

        result = {}

        # Check for AMX support for optimized kernels
        if cls._has_amx is None:
            cls._has_amx = cls._check_amx_support()

        if cls._has_amx:
            # Use AMX-optimized kernels from sgl_kernel
            result["silu_and_mul"] = torch.ops.sgl_kernel.silu_and_mul_cpu
            result["gelu_and_mul"] = torch.ops.sgl_kernel.gelu_and_mul_cpu
            result["gelu_tanh_and_mul"] = torch.ops.sgl_kernel.gelu_tanh_and_mul_cpu

        # If no AMX, will fall back to native implementations in ops/activation.py

        return result

    def get_op(self, op: "OpProxy") -> Callable | None:
        """Get the CPU implementation of an operation."""
        if CpuPlatform._ops is None:
            CpuPlatform._ops = self._init_ops()
        return CpuPlatform._ops.get(op.name)

    def postprocess_server_args(self, args: "ServerArgs") -> None:
        """Apply CPU-specific server argument defaults.

        Sets attention backend to intel_amx if not specified, and
        sampling backend to pytorch for CPU compatibility.
        """
        if args.attention_backend is None:
            args.attention_backend = "intel_amx"
        args.sampling_backend = "pytorch"
