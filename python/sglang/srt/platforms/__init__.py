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
SGLang Platform Detection and Singleton Management
===================================================

This module is the entry point for the Platform Abstraction Layer.
It provides the `current_platform` singleton that the rest of SGLang uses.

USAGE:
------
    from sglang.srt.platforms import current_platform

    # Check platform type
    if current_platform.is_cuda:
        print("Running on NVIDIA CUDA")

    # Get platform-specific op implementation
    impl = current_platform.get_op_by_name("silu_and_mul")

    # Access platform-specific modules
    result = current_platform.modules.npu_swiglu(x)  # Only on NPU

    # Apply platform-specific server defaults (called automatically)
    current_platform.postprocess_server_args(args)

THE SINGLETON PATTERN:
----------------------
`current_platform` is a lazily-initialized singleton:

    1. On first access, _detect_platform() is called
    2. Platform detection runs (see priority order below)
    3. The detected Platform instance is cached in _current_platform
    4. Subsequent accesses return the cached instance

This ensures:
- No platform detection overhead if not used
- Consistent platform instance across the codebase
- Platform imports only happen when needed (lazy)

PLATFORM DETECTION PRIORITY:
----------------------------
The detection order is critical for correctness:

    MUSA > NPU > ROCm > CUDA > XPU > HPU > CPU

Why this order?
- MUSA: Must be checked first because it sets torch.cuda.is_available() = True
        but is NOT actually NVIDIA CUDA. Checking MUSA first prevents misdetection.
- NPU:  Has its own torch.npu namespace, easy to detect early.
- ROCm: Also sets torch.cuda.is_available() = True, so we check for
        torch.version.hip to distinguish from real CUDA.
- CUDA: Only reached after ruling out MUSA and ROCm.
- XPU:  Has its own torch.xpu namespace.
- HPU:  Detected via habana_frameworks import.
- CPU:  Final fallback if nothing else matches.

ADDING A NEW PLATFORM:
----------------------
1. Create platforms/myplatform.py with MyPlatform class
2. Add PlatformEnum.MYPLATFORM to interface.py
3. Add detection logic here in _detect_platform()
4. Follow the detection priority guidelines above
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.platforms.interface import Platform

from sglang.srt.platforms.interface import PlatformEnum

# Singleton instance - lazily initialized on first access
_current_platform: "Platform | None" = None


def _detect_platform() -> "Platform":
    """Detect and return the current platform instance.

    This function implements the platform detection priority.
    The order is critical - see module docstring for explanation.

    Returns:
        A Platform instance for the detected hardware.
    """
    import torch

    # -------------------------------------------------------------------------
    # Priority 1: MUSA (Moore Threads)
    # -------------------------------------------------------------------------
    # CRITICAL: Must check MUSA before CUDA because MUSA sets torch.cuda = True
    # but requires different kernels and has torch.musa namespace
    if hasattr(torch, "musa") and torch.musa.is_available():
        from sglang.srt.platforms.musa import MusaPlatform

        return MusaPlatform()

    # -------------------------------------------------------------------------
    # Priority 2: NPU (Huawei Ascend)
    # -------------------------------------------------------------------------
    # NPU has its own torch.npu namespace
    if hasattr(torch, "npu") and torch.npu.is_available():
        from sglang.srt.platforms.npu import NpuPlatform

        return NpuPlatform()

    # -------------------------------------------------------------------------
    # Priority 3: ROCm (AMD)
    # -------------------------------------------------------------------------
    # CRITICAL: ROCm also sets torch.cuda.is_available() = True
    # Distinguish by checking torch.version.hip is not None
    if torch.cuda.is_available() and torch.version.hip is not None:
        from sglang.srt.platforms.rocm import RocmPlatform

        return RocmPlatform()

    # -------------------------------------------------------------------------
    # Priority 4: CUDA (NVIDIA)
    # -------------------------------------------------------------------------
    # Only reached after ruling out MUSA and ROCm
    # Both of which set torch.cuda.is_available() = True
    if torch.cuda.is_available() and torch.version.cuda is not None:
        from sglang.srt.platforms.cuda import CudaPlatform

        return CudaPlatform()

    # -------------------------------------------------------------------------
    # Priority 5: XPU (Intel)
    # -------------------------------------------------------------------------
    # Intel XPU has its own torch.xpu namespace
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        from sglang.srt.platforms.xpu import XpuPlatform

        return XpuPlatform()

    # -------------------------------------------------------------------------
    # Priority 6: HPU (Intel Habana)
    # -------------------------------------------------------------------------
    # Habana Gaudi is detected via the habana_frameworks import
    try:
        import habana_frameworks.torch  # noqa: F401

        from sglang.srt.platforms.hpu import HpuPlatform

        return HpuPlatform()
    except ImportError:
        pass

    # -------------------------------------------------------------------------
    # Priority 7: CPU (Fallback)
    # -------------------------------------------------------------------------
    # If no accelerator is detected, fall back to CPU
    from sglang.srt.platforms.cpu import CpuPlatform

    return CpuPlatform()


def _get_current_platform() -> "Platform":
    """Get or create the current platform singleton.

    This function ensures only one Platform instance exists.
    Thread-safety note: In SGLang's single-process-per-worker model,
    this is safe. For multi-threaded scenarios, add locking if needed.
    """
    global _current_platform
    if _current_platform is None:
        _current_platform = _detect_platform()
    return _current_platform


# =============================================================================
# Module-level lazy attribute access
# =============================================================================
# This uses Python's __getattr__ for modules (PEP 562) to make
# `current_platform` act like a module attribute while being lazily initialized.


def __getattr__(name: str):
    if name == "current_platform":
        return _get_current_platform()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PlatformEnum",
    "current_platform",
]
