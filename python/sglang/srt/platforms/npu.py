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
Huawei Ascend NPU Platform Implementation
==========================================

This platform supports Huawei Ascend NPUs (Neural Processing Units) via the
torch_npu library.

NPU-SPECIFIC CONSIDERATIONS:
----------------------------
1. device_type = "npu" - Ascend has its own device namespace (torch.npu)

2. Different API than CUDA - NPU uses vendor-specific functions like:
   - torch_npu.npu_swiglu() instead of sgl_kernel.silu_and_mul()
   - torch_npu.npu_geglu() instead of sgl_kernel.gelu_and_mul()

3. Wrapper functions needed - The NPU API differs from CUDA kernels, so we
   wrap the NPU functions in _init_ops() to match the standard op signature.

WHY WRAPPER FUNCTIONS?
----------------------
The standard op signature is: op(x, out) -> None (in-place into out)
But NPU functions return a new tensor: result = torch_npu.npu_swiglu(x)

To bridge this, we create wrapper functions:
    def npu_silu_and_mul(x, out):
        result = torch_npu.npu_swiglu(x)
        out.copy_(result)  # Copy to output tensor

This keeps the op interface consistent across all platforms.

PLATFORM MODULES:
-----------------
NPU exposes torch_npu functions via PlatformModules for direct access:
    current_platform.modules.npu_swiglu(x)
    current_platform.modules.npu_fast_gelu(x)

SERVER ARG POST-PROCESSING:
---------------------------
NPU has platform-specific defaults (attention backend, memory config, etc.)
applied via postprocess_server_args(). See hardware_backend/npu/utils.py.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

from sglang.srt.platforms.interface import Platform, PlatformEnum

if TYPE_CHECKING:
    from sglang.srt.ops.base import OpProxy
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class NpuPlatform(Platform):
    """Huawei Ascend NPU platform implementation.

    Supports Huawei Ascend NPUs via torch_npu. Uses vendor-specific
    functions (npu_swiglu, npu_geglu) wrapped to match the standard API.
    """

    _enum = PlatformEnum.NPU
    device_name = "npu"
    device_type = "npu"

    # Op registry - lazily initialized on first access
    _ops: dict[str, Callable] | None = None

    # Lazy-loaded torch_npu module (for PlatformModules)
    _torch_npu: Any = None

    @classmethod
    def _get_torch_npu(cls) -> Any:
        """Lazy load and cache torch_npu module."""
        if cls._torch_npu is None:
            import torch_npu

            cls._torch_npu = torch_npu
        return cls._torch_npu

    def _get_module_attr(self, name: str) -> Any | None:
        """Get NPU-specific module attributes.

        This enables PlatformModules to expose NPU-specific functions:
            current_platform.modules.npu_swiglu(x)
            current_platform.modules.npu_fast_gelu(x)

        Available attributes (from torch_npu):
        - npu_swiglu: Fused SiLU with gating
        - npu_geglu: Fused GELU with gating
        - npu_fast_gelu: Fast approximate GELU
        """
        torch_npu = self._get_torch_npu()

        # Expose torch_npu functions directly
        if hasattr(torch_npu, name):
            return getattr(torch_npu, name)

        return None

    @classmethod
    def _init_ops(cls) -> dict[str, Callable]:
        """Initialize the op registry with NPU-wrapped kernels.

        NPU uses torch_npu functions with a different API than CUDA kernels:
        - CUDA: silu_and_mul(x, out) modifies out in-place
        - NPU: npu_swiglu(x) returns a new tensor

        We wrap NPU functions to match the standard signature.
        """
        import torch_npu

        # Wrapper: npu_swiglu() -> silu_and_mul(x, out)
        def npu_silu_and_mul(x, out):
            result = torch_npu.npu_swiglu(x)
            out.copy_(result)

        # Wrapper: npu_geglu(approximate=0) -> gelu_and_mul(x, out)
        def npu_gelu_and_mul(x, out):
            result, _ = torch_npu.npu_geglu(
                x, dim=-1, approximate=0, activate_left=True
            )
            out.copy_(result)

        # Wrapper: npu_geglu(approximate=1) -> gelu_tanh_and_mul(x, out)
        def npu_gelu_tanh_and_mul(x, out):
            result, _ = torch_npu.npu_geglu(
                x, dim=-1, approximate=1, activate_left=True
            )
            out.copy_(result)

        return {
            "silu_and_mul": npu_silu_and_mul,
            "gelu_and_mul": npu_gelu_and_mul,
            "gelu_tanh_and_mul": npu_gelu_tanh_and_mul,
        }

    def get_op(self, op: "OpProxy") -> Callable | None:
        """Get the NPU implementation of an operation."""
        if NpuPlatform._ops is None:
            NpuPlatform._ops = self._init_ops()
        return NpuPlatform._ops.get(op.name)

    def postprocess_server_args(self, args: "ServerArgs") -> None:
        """Apply NPU-specific server argument defaults.

        This is an example of RFC Feature #2: Platform-specific Server
        Argument Post-processing. Called during ServerArgs initialization.

        Sets attention backends, memory configurations, and other
        NPU-specific settings for Huawei Ascend devices.
        """
        from sglang.srt.hardware_backend.npu.utils import set_default_server_args

        set_default_server_args(args)

        if args.piecewise_cuda_graph_compiler != "eager":
            logger.warning(
                "At this moment Ascend platform only support prefill graph compilation with "
                "piecewise_cuda_graph_compiler='eager', change piecewise_cuda_graph_compiler to 'eager'."
            )
            args.piecewise_cuda_graph_compiler = "eager"
