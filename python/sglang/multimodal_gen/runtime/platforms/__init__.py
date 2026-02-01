# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/platforms/__init__.py
"""
Multimodal Gen Platform Re-exports.

This module re-exports the unified platform abstraction from sglang.platforms
for backward compatibility. Existing imports from this module will continue
to work, but new code should import from sglang.platforms directly.

Usage:
    # Old import (still works):
    from sglang.multimodal_gen.runtime.platforms import current_platform

    # New preferred import:
    from sglang.platforms import current_platform

For CUDA platforms, the unified sglang.platforms.cuda.CudaPlatform is used.
For other platforms (ROCm, MPS, MUSA, CPU), the local implementations are
still used until they are migrated to the top-level module.
"""

import traceback
from typing import TYPE_CHECKING

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import resolve_obj_by_qualname

# Re-export core classes from unified platform module
from sglang.platforms.interface import (
    AttentionBackendEnum,
    CpuArchEnum,
    DeviceCapability,
    Platform,
    PlatformEnum,
    UnspecifiedPlatform,
)

logger = init_logger(__name__)


def cuda_platform_plugin() -> str | None:
    """Detect if CUDA is available - now uses unified platform."""
    is_cuda = False

    try:
        from sglang.multimodal_gen.utils import import_pynvml

        pynvml = import_pynvml()  # type: ignore[no-untyped-call]
        pynvml.nvmlInit()
        try:
            is_cuda = pynvml.nvmlDeviceGetCount() > 0
        finally:
            pynvml.nvmlShutdown()
    except Exception as e:
        if "nvml" not in e.__class__.__name__.lower():
            raise e

        import os

        def cuda_is_jetson() -> bool:
            return os.path.isfile("/etc/nv_tegra_release") or os.path.exists(
                "/sys/class/tegra-firmware"
            )

        if cuda_is_jetson():
            is_cuda = True
    if is_cuda:
        logger.info("CUDA is available")

    # Use the unified CUDA platform
    return "sglang.platforms.cuda.CudaPlatform" if is_cuda else None


def mps_platform_plugin() -> str | None:
    """Detect if MPS (Metal Performance Shaders) is available on macOS."""
    is_mps = False

    try:
        import torch

        if torch.backends.mps.is_available():
            is_mps = True
            logger.info("MPS (Metal Performance Shaders) is available")
    except Exception as e:
        logger.info("MPS detection failed: %s", e)

    return "sglang.multimodal_gen.runtime.platforms.mps.MpsPlatform" if is_mps else None


def cpu_platform_plugin() -> str | None:
    """Detect if CPU platform should be used."""
    return "sglang.multimodal_gen.runtime.platforms.cpu.CpuPlatform"


def rocm_platform_plugin() -> str | None:
    is_rocm = False

    try:
        import amdsmi

        amdsmi.amdsmi_init()
        try:
            if len(amdsmi.amdsmi_get_processor_handles()) > 0:
                is_rocm = True
                logger.info("ROCm platform is available")
        finally:
            amdsmi.amdsmi_shut_down()
    except Exception as e:
        logger.info("ROCm platform is unavailable: %s", e)

    return (
        "sglang.multimodal_gen.runtime.platforms.rocm.RocmPlatform" if is_rocm else None
    )


def npu_platform_plugin() -> str | None:
    is_npu = False

    try:
        import torch

        if torch.npu.is_available():
            is_npu = True
            logger.info("NPU is available")
    except Exception as e:
        logger.info("NPU detection failed: %s", e)
    return (
        "sglang.multimodal_gen.runtime.platforms.npu.NPUPlatformBase"
        if is_npu
        else None
    )


def musa_platform_plugin() -> str | None:
    is_musa = False

    try:
        import pymtml

        pymtml.mtmlLibraryInit()
        try:
            is_musa = pymtml.mtmlLibraryCountDevice() > 0
        finally:
            pymtml.mtmlLibraryShutDown()
    except Exception as e:
        logger.info("MUSA platform is unavailable: %s", e)

    return (
        "sglang.multimodal_gen.runtime.platforms.musa.MusaPlatform" if is_musa else None
    )


builtin_platform_plugins = {
    "cuda": cuda_platform_plugin,
    "rocm": rocm_platform_plugin,
    "mps": mps_platform_plugin,
    "cpu": cpu_platform_plugin,
    "npu": npu_platform_plugin,
    "musa": musa_platform_plugin,
}


def resolve_current_platform_cls_qualname() -> str:
    """Resolve the platform class to use."""
    # Try MPS first on macOS
    platform_cls_qualname = mps_platform_plugin()
    if platform_cls_qualname is not None:
        return platform_cls_qualname

    # Fall back to ROCm
    platform_cls_qualname = rocm_platform_plugin()
    if platform_cls_qualname is not None:
        return platform_cls_qualname

    # Fall back to CUDA (now uses unified platform)
    platform_cls_qualname = cuda_platform_plugin()
    if platform_cls_qualname is not None:
        return platform_cls_qualname

    # Fall back to NPU
    platform_cls_qualname = npu_platform_plugin()
    if platform_cls_qualname is not None:
        return platform_cls_qualname

    # Fall back to MUSA
    platform_cls_qualname = musa_platform_plugin()
    if platform_cls_qualname is not None:
        return platform_cls_qualname

    # Fall back to CPU as last resort
    platform_cls_qualname = cpu_platform_plugin()
    if platform_cls_qualname is not None:
        return platform_cls_qualname

    raise RuntimeError("No platform plugin found. Please check your installation.")


_current_platform: Platform | None = None
_init_trace: str = ""

current_platform: Platform


def __getattr__(name: str):
    if name == "current_platform":
        global _current_platform
        if _current_platform is None:
            platform_cls_qualname = resolve_current_platform_cls_qualname()
            _current_platform = resolve_obj_by_qualname(platform_cls_qualname)()
            global _init_trace
            _init_trace = "".join(traceback.format_stack())
            # Call init_platform for one-time setup
            if hasattr(_current_platform, "init_platform"):
                _current_platform.init_platform()
        return _current_platform
    elif name in globals():
        return globals()[name]
    else:
        raise AttributeError(f"No attribute named '{name}' exists in {__name__}.")


__all__ = [
    "Platform",
    "PlatformEnum",
    "AttentionBackendEnum",
    "CpuArchEnum",
    "DeviceCapability",
    "UnspecifiedPlatform",
    "current_platform",
    "_init_trace",
]
