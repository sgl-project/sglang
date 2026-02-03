# SPDX-License-Identifier: Apache-2.0
"""
Unified Platform Abstraction for SGLang.

This module provides a unified platform abstraction layer that consolidates
platform-specific logic from both SRT (LLM serving) and multimodal_gen
(diffusion models) into a single, top-level module.

Usage:
    from sglang.platforms import current_platform

    # Platform detection (cached, fast)
    if current_platform.is_cuda():
        # CUDA-specific code
        ...

    # Op dispatch via property access (lazy loading)
    silu_and_mul = current_platform.silu_and_mul

    # Backend selection
    backend = current_platform.get_default_attention_backend()

For IDE type hints, developers have several options:
1. Default: current_platform is typed as Platform (base class)
2. Platform-specific stubs: Symlink the appropriate .pyi file for your platform
3. Type ignore: Use # type: ignore[attr-defined] for platform-specific ops
"""
from __future__ import annotations

import logging
import traceback
from typing import TYPE_CHECKING

from sglang.platforms.interface import (
    AttentionBackendEnum,
    CpuArchEnum,
    DeviceCapability,
    Platform,
    PlatformEnum,
    UnspecifiedPlatform,
    resolve_obj_by_qualname,
)

if TYPE_CHECKING:
    from sglang.platforms.cuda import CudaPlatform

    # For type checking, we use Union to allow IDE to see all platform methods
    _PlatformType = CudaPlatform | Platform
    current_platform: _PlatformType

logger = logging.getLogger(__name__)

_current_platform: Platform | None = None
_init_trace: str = ""


def _cuda_platform_plugin() -> str | None:
    """
    Detect if CUDA is available.

    Returns:
        Platform class path if CUDA available, None otherwise.

    Note: Falls back to Jetson detection if NVML fails.
    """
    is_cuda = False

    try:
        from sglang.multimodal_gen.utils import import_pynvml

        pynvml = import_pynvml()
        pynvml.nvmlInit()
        try:
            is_cuda = pynvml.nvmlDeviceGetCount() > 0
        finally:
            pynvml.nvmlShutdown()
    except ImportError as e:
        # pynvml not installed - check if this is expected or a broken installation
        error_str = str(e).lower()
        if "no module named" not in error_str:
            # Partial/broken installation - log warning
            logger.warning(
                "pynvml import failed with unexpected error: %s. "
                "If you have NVIDIA hardware, check your pynvml installation.",
                e,
            )
    except Exception as e:
        # Check if this is an NVML-specific error
        is_nvml_error = (
            "nvml" in str(type(e).__module__).lower()
            or "nvml" in type(e).__name__.lower()
            or "nvml" in str(e).lower()
        )

        if not is_nvml_error:
            # Re-raise non-NVML errors
            raise

        # Log NVML errors with actionable guidance
        logger.warning(
            "NVML error during CUDA detection: %s (%s). "
            "If you have NVIDIA hardware, check: (1) nvidia-smi works, "
            "(2) your user has GPU access permissions, "
            "(3) NVIDIA driver is properly installed. "
            "Checking for Jetson platform as fallback.",
            type(e).__name__,
            e,
        )

        # NVML not available - check for Jetson (supports CUDA without NVML)
        import os

        def cuda_is_jetson() -> bool:
            return os.path.isfile("/etc/nv_tegra_release") or os.path.exists(
                "/sys/class/tegra-firmware"
            )

        if cuda_is_jetson():
            is_cuda = True
            logger.info("Detected Jetson platform (CUDA without NVML)")

    if is_cuda:
        logger.info("CUDA platform detected")

    return "sglang.platforms.cuda.CudaPlatform" if is_cuda else None


def _rocm_platform_plugin() -> str | None:
    """
    Detect if ROCm is available.

    Returns:
        Platform class path if ROCm available, None otherwise.
    """
    is_rocm = False

    try:
        import amdsmi

        amdsmi.amdsmi_init()
        try:
            if len(amdsmi.amdsmi_get_processor_handles()) > 0:
                is_rocm = True
                logger.info("ROCm platform detected")
        finally:
            amdsmi.amdsmi_shut_down()
    except ImportError:
        # amdsmi not installed - expected on non-ROCm systems
        pass
    except Exception as e:
        # Log unexpected errors instead of silently ignoring
        logger.warning(
            "ROCm detection failed with unexpected error: %s. "
            "If you have AMD hardware, check driver permissions and amdsmi installation.",
            e,
        )

    # Use unified RocmPlatform implementation
    return "sglang.platforms.rocm.RocmPlatform" if is_rocm else None


def _musa_platform_plugin() -> str | None:
    """
    Detect if MUSA (Moore Threads) is available.

    Returns:
        Platform class path if MUSA available, None otherwise.
    """
    is_musa = False

    try:
        import pymtml

        pymtml.mtmlLibraryInit()
        try:
            is_musa = pymtml.mtmlLibraryCountDevice() > 0
            if is_musa:
                logger.info("MUSA platform detected")
        finally:
            pymtml.mtmlLibraryShutDown()
    except ImportError:
        # pymtml not installed - expected on non-MUSA systems
        pass
    except Exception as e:
        # Log unexpected errors instead of silently ignoring
        logger.warning(
            "MUSA detection failed with unexpected error: %s. "
            "If you have Moore Threads hardware, check driver and pymtml installation.",
            e,
        )

    # For now, MUSA falls back to multimodal_gen implementation
    return (
        "sglang.multimodal_gen.runtime.platforms.musa.MusaPlatform" if is_musa else None
    )


def _xpu_platform_plugin() -> str | None:
    """
    Detect if Intel XPU is available.

    Returns:
        Platform class path if XPU available, None otherwise.
    """
    is_xpu = False

    try:
        import torch

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            is_xpu = torch.xpu.device_count() > 0
            if is_xpu:
                logger.info("Intel XPU platform detected")
    except ImportError:
        # torch not installed
        pass
    except Exception as e:
        logger.warning(
            "XPU detection failed with unexpected error: %s. "
            "If you have Intel GPU hardware, check driver and oneAPI installation.",
            e,
        )

    # Use unified XpuPlatform implementation
    return "sglang.platforms.xpu.XpuPlatform" if is_xpu else None


def _mps_platform_plugin() -> str | None:
    """
    Detect if MPS (Metal Performance Shaders) is available on macOS.

    Returns:
        Platform class path if MPS available, None otherwise.
    """
    is_mps = False

    try:
        import torch
    except ImportError:
        # torch not installed - cannot check for MPS
        logger.debug("PyTorch not installed, cannot detect MPS")
        return None

    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            is_mps = True
            logger.info("MPS (Metal Performance Shaders) platform detected")
    except AttributeError as e:
        # Older PyTorch version without MPS support
        logger.debug("MPS backend not available (likely older PyTorch): %s", e)
    except Exception as e:
        logger.warning(
            "MPS detection failed: %s. If you're on Apple Silicon, "
            "ensure you have a recent version of PyTorch with Metal support.",
            e,
        )

    # For now, MPS falls back to multimodal_gen implementation
    return "sglang.multimodal_gen.runtime.platforms.mps.MpsPlatform" if is_mps else None


def _cpu_platform_plugin() -> str | None:
    """CPU is always available as a fallback."""
    # For now, CPU falls back to multimodal_gen implementation
    return "sglang.multimodal_gen.runtime.platforms.cpu.CpuPlatform"


def _detect_platform() -> Platform:
    """
    Detect and return the current platform instance.

    Detection order is critical:
    1. MPS - macOS Metal (check first on macOS)
    2. ROCm - AMD (sets torch.cuda.is_available() = True)
    3. CUDA - NVIDIA (only after ruling out ROCm)
    4. MUSA - Moore Threads
    5. XPU - Intel GPUs
    6. CPU - Final fallback

    Returns:
        Platform instance for the detected platform.

    Raises:
        RuntimeError: If no platform can be detected (should not happen).
    """
    # Try MPS first on macOS
    platform_cls_qualname = _mps_platform_plugin()
    if platform_cls_qualname is not None:
        try:
            return resolve_obj_by_qualname(platform_cls_qualname)()
        except (ImportError, AttributeError) as e:
            logger.error("Failed to load MPS platform: %s", e)

    # Fall back to ROCm
    platform_cls_qualname = _rocm_platform_plugin()
    if platform_cls_qualname is not None:
        try:
            return resolve_obj_by_qualname(platform_cls_qualname)()
        except (ImportError, AttributeError) as e:
            logger.error("Failed to load ROCm platform: %s", e)

    # Fall back to CUDA
    platform_cls_qualname = _cuda_platform_plugin()
    if platform_cls_qualname is not None:
        try:
            return resolve_obj_by_qualname(platform_cls_qualname)()
        except (ImportError, AttributeError) as e:
            logger.error("Failed to load CUDA platform: %s", e)

    # Fall back to MUSA
    platform_cls_qualname = _musa_platform_plugin()
    if platform_cls_qualname is not None:
        try:
            return resolve_obj_by_qualname(platform_cls_qualname)()
        except (ImportError, AttributeError) as e:
            logger.error("Failed to load MUSA platform: %s", e)

    # Fall back to XPU (Intel GPUs)
    platform_cls_qualname = _xpu_platform_plugin()
    if platform_cls_qualname is not None:
        try:
            return resolve_obj_by_qualname(platform_cls_qualname)()
        except (ImportError, AttributeError) as e:
            logger.error("Failed to load XPU platform: %s", e)

    # Fall back to CPU as last resort
    platform_cls_qualname = _cpu_platform_plugin()
    if platform_cls_qualname is not None:
        try:
            return resolve_obj_by_qualname(platform_cls_qualname)()
        except (ImportError, AttributeError) as e:
            logger.error("Failed to load CPU platform: %s", e)
            raise RuntimeError(
                f"Failed to load CPU platform (last resort): {e}. "
                "Please check your SGLang installation."
            ) from e

    raise RuntimeError("No platform plugin found. Please check your installation.")


def _get_current_platform() -> Platform:
    """Get or create the current platform singleton."""
    global _current_platform, _init_trace
    if _current_platform is None:
        _current_platform = _detect_platform()
        _init_trace = "".join(traceback.format_stack())
        _current_platform.init_platform()  # One-time initialization
    return _current_platform


# Lazy attribute access via PEP 562
def __getattr__(name: str):
    if name == "current_platform":
        return _get_current_platform()
    elif name in globals():
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Core classes and enums
    "Platform",
    "PlatformEnum",
    "DeviceCapability",
    "CpuArchEnum",
    "UnspecifiedPlatform",
    "AttentionBackendEnum",
    # Utility
    "resolve_obj_by_qualname",
    # Singleton
    "current_platform",
    # For backward compatibility
    "_init_trace",
]
