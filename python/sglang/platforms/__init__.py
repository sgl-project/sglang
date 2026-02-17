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
from collections.abc import Callable
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


def _try_import_pynvml():
    """Import pynvml, preferring the bundled multimodal_gen copy.

    Falls back to the pip-installed pynvml package if multimodal_gen is not
    available.  Returns None when neither source provides pynvml.

    Note: KeyError is caught because importlib can raise it when namespace
    packages or __init__.py files have resolution issues.
    """
    try:
        from sglang.multimodal_gen.utils import import_pynvml

        return import_pynvml()
    except (ImportError, KeyError):
        pass

    try:
        import pynvml

        return pynvml
    except ImportError:
        return None


def _cuda_platform_plugin() -> str | None:
    """
    Detect if CUDA is available.

    Returns:
        Platform class path if CUDA available, None otherwise.

    Note: Falls back to Jetson detection if NVML fails.
    """
    is_cuda = False

    pynvml = _try_import_pynvml()
    if pynvml is not None:
        try:
            pynvml.nvmlInit()
            try:
                is_cuda = pynvml.nvmlDeviceGetCount() > 0
            finally:
                pynvml.nvmlShutdown()
        except Exception as e:
            # Check if this is an NVML-specific error
            is_nvml_error = False
            if (
                pynvml
                and hasattr(pynvml, "NVMLError")
                and isinstance(e, pynvml.NVMLError)
            ):
                is_nvml_error = True
            else:
                # Fallback to string matching if type check is not possible
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
    return "sglang.platforms.cpu.CpuPlatform"


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
    # Ordered list of (platform name, detector function) pairs.
    # Each detector returns a qualname string if the platform is available,
    # or None if it is not.
    _PLATFORM_DETECTORS: list[tuple[str, Callable[[], str | None]]] = [
        ("MPS", _mps_platform_plugin),
        ("ROCm", _rocm_platform_plugin),
        ("CUDA", _cuda_platform_plugin),
        ("MUSA", _musa_platform_plugin),
        ("XPU", _xpu_platform_plugin),
        ("CPU", _cpu_platform_plugin),
    ]

    for name, detect_fn in _PLATFORM_DETECTORS:
        qualname = detect_fn()
        if qualname is None:
            continue
        try:
            return resolve_obj_by_qualname(qualname)()
        except (ImportError, AttributeError, KeyError) as e:
            logger.error("Failed to load %s platform: %s", name, e, exc_info=True)
            # CPU is the last-resort fallback; if it fails, nothing else to try.
            if name == "CPU":
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
