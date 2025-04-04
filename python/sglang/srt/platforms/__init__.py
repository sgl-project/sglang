# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.8.2/vllm/platforms/__init__.py

import logging
import traceback
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.platforms.interface import CpuArchEnum, Platform, PlatformEnum
from sglang.srt.utils import resolve_obj_by_qualname

logger = logging.getLogger(__name__)


def cuda_platform_plugin() -> Optional[str]:
    is_cuda = False
    logger.debug("Checking if CUDA platform is available.")
    try:
        if torch.cuda.is_available() and torch.version.cuda:
            is_cuda = True
            logger.debug("Confirmed CUDA platform is available.")
    except Exception as e:
        logger.debug("Exception happens when checking CUDA platform: %s", str(e))
        # CUDA is supported on Jetson, but NVML may not be.
        import os

        def cuda_is_jetson() -> bool:
            return os.path.isfile("/etc/nv_tegra_release") or os.path.exists(
                "/sys/class/tegra-firmware"
            )

        if cuda_is_jetson():
            logger.debug("Confirmed CUDA platform is available on Jetson.")
            is_cuda = True
        else:
            logger.debug("CUDA platform is not available because: %s", str(e))

    return "sglang.srt.platforms.cuda.CudaPlatform" if is_cuda else None


def hip_platform_plugin() -> Optional[str]:
    is_hip = False
    logger.debug("Checking if HIP platform is available.")
    try:
        if torch.cuda.is_available() and torch.version.hip:
            is_hip = True
            logger.debug("Confirmed HIP platform is available.")
    except Exception as e:
        logger.debug("HIP platform is not available because: %s", str(e))
        pass

    return "sglang.srt.platforms.hip.HipPlatform" if is_hip else None


def hpu_platform_plugin() -> Optional[str]:
    is_hpu = False
    logger.debug("Checking if HPU platform is available.")
    try:
        import habana_frameworks.torch.hpu

        if hasattr(torch, "hpu") and torch.hpu.is_available():
            is_hpu = True
            logger.debug("Confirmed HPU platform is available.")
    except Exception as e:
        logger.debug("HPU platform is not available because: %s", str(e))
        pass

    return "sglang.srt.platforms.hpu.HpuPlatform" if is_hpu else None


def xpu_platform_plugin() -> Optional[str]:
    is_xpu = False
    logger.debug("Checking if XPU platform is available.")
    try:
        # installed IPEX if the machine has XPUs.
        import intel_extension_for_pytorch  # noqa: F401
        import oneccl_bindings_for_pytorch  # noqa: F401

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            is_xpu = True
            logger.debug("Confirmed XPU platform is available.")
    except Exception as e:
        logger.debug("XPU platform is not available because: %s", str(e))
        pass

    return "sglang.srt.platforms.xpu.XpuPlatform" if is_xpu else None


def cpu_platform_plugin() -> Optional[str]:
    is_cpu = False
    logger.debug("Checking if CPU platform is available.")
    try:
        if torch.cpu.is_available and (
            torch.cpu._is_avx512_bf16_supported or torch.cpu._is_arm_sve_supported
        ):
            is_cpu = True
            logger.debug("Confirmed CPU platform is available.")
    except Exception as e:
        logger.debug("CPU platform is not available because: %s", str(e))
        pass

    return "sglang.srt.platforms.cpu.CpuPlatform" if is_cpu else None


builtin_platform_plugins = {
    "cuda": cuda_platform_plugin,
    "hip": hip_platform_plugin,
    "hpu": hpu_platform_plugin,
    "xpu": xpu_platform_plugin,
    "cpu": cpu_platform_plugin,
}
# Let platform detection order be controllable
builtin_platform_names = [
    "cuda",
    "hip",
    "hpu",
    "xpu",
    "cpu",
]

_available_platforms = None
_current_platform = None
_init_trace: str = ""


def resolve_available_platforms() -> list[str]:
    global _available_platforms
    if _available_platforms is not None:
        return _available_platforms
    # TODO: Do not support external plugin now
    _available_platforms = []
    for name in builtin_platform_names:
        func = builtin_platform_plugins[name]
        try:
            assert callable(func)
            platform_cls_qualname = func()
            if platform_cls_qualname is not None:
                _available_platforms.append(name)
        except Exception:
            pass

    return _available_platforms


def set_current_platform(name: str) -> None:
    global _current_platform
    global _init_trace

    if name is None:
        platform_cls_name = "sglang.srt.platforms.interface.UnspecifiedPlatform"
        name = "unspecified"
    else:
        avail_platforms = resolve_available_platforms()
        if name not in avail_platforms:
            logger.error(
                f'Platform "{name}" is not available. Currently available platforms are {avail_platforms}'
            )
            raise NotImplementedError()
        platform_cls_name = builtin_platform_plugins[name]()
    logger.info(f"SGLang is running on {name} platform")
    _current_platform = resolve_obj_by_qualname(platform_cls_name)()
    _current_platform.init_environments()
    _init_trace = "".join(traceback.format_stack())


def recommended_platform() -> Optional[str]:
    avail_platforms = resolve_available_platforms()
    if len(avail_platforms) == 0:
        platform_name = None
    else:
        platform_name = avail_platforms[0]
    return platform_name


if TYPE_CHECKING:
    current_platform: Platform


def __getattr__(name: str):
    if name == "current_platform":
        # lazy init current_platform.
        # 1. out-of-tree platform plugins need `from sglang.srt.platforms import
        #    Platform` so that they can inherit `Platform` class. Therefore,
        #    we cannot resolve `current_platform` during the import of
        #    `sglang.srt..platforms`.
        # 2. when users use out-of-tree platform plugins, they might run
        #    `import sglang.srt`, some sglang internal code might access
        #    `current_platform` during the import, and we need to make sure
        #    `current_platform` is only resolved after the plugins are loaded
        #    (we have tests for this, if any developer violate this, they will
        #    see the test failures).
        global _current_platform
        if _current_platform is None:
            set_current_platform(recommended_platform())
        return _current_platform
    elif name in globals():
        return globals()[name]
    else:
        raise AttributeError(f"No attribute named '{name}' exists in {__name__}.")


__all__ = [
    "CpuArchEnum",
    "Platform",
    "PlatformEnum",
    "current_platform",
    "resolve_available_platforms",
    "recommended_platform",
    "set_current_platform",
    "_init_trace",
]
