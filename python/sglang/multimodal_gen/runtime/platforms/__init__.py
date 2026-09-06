# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/platforms/__init__.py

import pkgutil
import traceback
from importlib.metadata import entry_points

from sglang.multimodal_gen.plugins import (
    DIFFUSION_PLATFORM_PLUGINS_GROUP,
    discover_diffusion_plugins,
)

# imported by other files, do not remove
from sglang.multimodal_gen.runtime.platforms.interface import (  # noqa: F401
    AttentionBackendEnum,
    Platform,
    PlatformEnum,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.environ import envs

logger = init_logger(__name__)

_BUILTIN_PLATFORM_QUALNAMES = {
    "cpu": "sglang.multimodal_gen.runtime.platforms.cpu.CpuPlatform",
    "cuda": "sglang.multimodal_gen.runtime.platforms.cuda.CudaPlatform",
    "rocm": "sglang.multimodal_gen.runtime.platforms.rocm.RocmPlatform",
    "mps": "sglang.multimodal_gen.runtime.platforms.mps.MpsPlatform",
    "npu": "sglang.multimodal_gen.runtime.platforms.npu.NPUPlatformBase",
    "musa": "sglang.multimodal_gen.runtime.platforms.musa.MusaPlatform",
    "xpu": "sglang.multimodal_gen.runtime.platforms.xpu.XpuPlatform",
}


def cuda_platform_plugin() -> str | None:
    is_cuda = False

    try:
        from sglang.multimodal_gen.utils import import_pynvml

        pynvml = import_pynvml()  # type: ignore[no-untyped-call]
        pynvml.nvmlInit()
        try:
            # NOTE: Edge case: sgl_diffusion cpu build on a GPU machine.
            # Third-party pynvml can be imported in cpu build,
            # we need to check if sgl_diffusion is built with cpu too.
            # Otherwise, sgl_diffusion will always activate cuda plugin
            # on a GPU machine, even if in a cpu build.
            is_cuda = pynvml.nvmlDeviceGetCount() > 0
        finally:
            pynvml.nvmlShutdown()
    except Exception as e:
        if "nvml" not in e.__class__.__name__.lower():
            # If the error is not related to NVML, re-raise it.
            raise e

        # CUDA is supported on Jetson, but NVML may not be.
        import os

        def cuda_is_jetson() -> bool:
            return os.path.isfile("/etc/nv_tegra_release") or os.path.exists(
                "/sys/class/tegra-firmware"
            )

        if cuda_is_jetson():
            is_cuda = True
        else:
            # NVML is NVIDIA-specific. CUDA-compatible stacks (e.g. Iluvatar
            # CoreX) expose devices through torch's CUDA API without shipping
            # libnvidia-ml. Only fall back when NVML itself is unavailable;
            # a successful NVML init that reports zero devices must keep the
            # CPU-build-on-GPU-machine edge case above.
            try:
                import torch

                # ROCm also exposes devices through torch.cuda, so keep this
                # non-NVML fallback limited to non-HIP runtimes.
                is_cuda = (
                    getattr(torch.version, "hip", None) is None
                    and torch.cuda.is_available()
                    and torch.cuda.device_count() > 0
                )
                if is_cuda:
                    logger.debug("CUDA detected via torch (NVML unavailable)")
            except Exception as exc:
                logger.debug("torch CUDA detection failed: %s", exc)

    if is_cuda:
        logger.debug("CUDA is available")

    return (
        "sglang.multimodal_gen.runtime.platforms.cuda.CudaPlatform" if is_cuda else None
    )


def mps_platform_plugin() -> str | None:
    """Detect if MPS (Metal Performance Shaders) is available on macOS."""
    is_mps = False

    try:
        import torch

        if torch.backends.mps.is_available():
            is_mps = True
            logger.debug("MPS (Metal Performance Shaders) is available")
    except Exception as e:
        logger.debug("MPS detection failed: %s", e)

    return "sglang.multimodal_gen.runtime.platforms.mps.MpsPlatform" if is_mps else None


def cpu_platform_plugin() -> str | None:
    """Detect if CPU platform should be used."""
    # CPU is always available as a fallback
    return "sglang.multimodal_gen.runtime.platforms.cpu.CpuPlatform"


def rocm_platform_plugin() -> str | None:
    is_rocm = False

    try:
        import amdsmi

        amdsmi.amdsmi_init()
        try:
            if len(amdsmi.amdsmi_get_processor_handles()) > 0:
                is_rocm = True
                logger.debug("ROCm platform is available")
        finally:
            amdsmi.amdsmi_shut_down()
    except Exception as e:
        logger.debug("ROCm platform is unavailable: %s", e)

    return (
        "sglang.multimodal_gen.runtime.platforms.rocm.RocmPlatform" if is_rocm else None
    )


def npu_platform_plugin() -> str | None:
    is_npu = False

    try:
        import torch

        if torch.npu.is_available():
            is_npu = True
            logger.debug("NPU is available")
    except Exception as e:
        logger.debug("NPU detection failed: %s", e)
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
        logger.debug("MUSA platform is unavailable: %s", e)

    return (
        "sglang.multimodal_gen.runtime.platforms.musa.MusaPlatform" if is_musa else None
    )


def xpu_platform_plugin() -> str | None:
    """Detect if Intel XPU platform is available."""
    is_xpu = False

    try:
        import torch

        # Check if Intel Extension for PyTorch is available and XPU devices exist
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            device_count = torch.xpu.device_count()
            if device_count > 0:
                is_xpu = True
                logger.info(
                    "Intel XPU platform is available with %d device(s)", device_count
                )
    except Exception as e:
        logger.info("Intel XPU platform is unavailable: %s", e)

    return "sglang.multimodal_gen.runtime.platforms.xpu.XpuPlatform" if is_xpu else None


builtin_platform_plugins = {
    "cuda": cuda_platform_plugin,
    "rocm": rocm_platform_plugin,
    "xpu": xpu_platform_plugin,
    "mps": mps_platform_plugin,
    "cpu": cpu_platform_plugin,
    "npu": npu_platform_plugin,
    "musa": musa_platform_plugin,
}


def resolve_current_platform_cls_qualname() -> str:
    """Resolve the qualname of the platform class to instantiate.

    Resolution order:

    1. ``SGLANG_DIFFUSION_PLATFORM_OVERRIDE`` — a built-in short name
       (``cpu``, ``cuda``, ...) or an explicit ``module.Class`` /
       ``module:Class`` qualname. Bypasses detection and plugins entirely.
    2. ``SGLANG_DIFFUSION_PLATFORM`` set — front-loading filter over the
       ``sglang.multimodal_gen.platforms`` entry point group: only the named
       plugin is imported and activated. Name not found, or ``activate()``
       returning None, is an error.
    3. ``SGLANG_DIFFUSION_PLATFORM`` unset — import and activate every
       discovered plugin. Exactly one activated wins, several is an error, none
       falls through to the built-in detection chain below.
    """
    forced_platform = envs.SGLANG_DIFFUSION_PLATFORM_OVERRIDE.get().strip()
    if forced_platform:
        qualname = _BUILTIN_PLATFORM_QUALNAMES.get(forced_platform.lower())
        if qualname is not None:
            return qualname
        # Not a short name: accept an explicit out-of-tree platform qualname.
        if "." in forced_platform or ":" in forced_platform:
            return forced_platform
        raise ValueError(
            f"Unsupported SGLANG_DIFFUSION_PLATFORM_OVERRIDE={forced_platform!r}. "
            f"Use one of {sorted(_BUILTIN_PLATFORM_QUALNAMES)} or a "
            f"'module.Class' qualname."
        )

    platform_cls_qualname = _resolve_platform_plugin_qualname()
    if platform_cls_qualname is not None:
        return platform_cls_qualname

    # Try MPS first on macOS
    platform_cls_qualname = mps_platform_plugin()
    if platform_cls_qualname is not None:
        return platform_cls_qualname

    # Try Intel XPU
    platform_cls_qualname = xpu_platform_plugin()
    if platform_cls_qualname is not None:
        return platform_cls_qualname

    # Fall back to ROCm
    platform_cls_qualname = rocm_platform_plugin()
    if platform_cls_qualname is not None:
        return platform_cls_qualname

    # Fall back to CUDA
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


def _resolve_platform_plugin_qualname() -> str | None:
    """Resolve an out-of-tree platform from the entry point group.

    Returns None when no plugin claims the machine, so the caller can fall back
    to the built-in detection chain.
    """
    selected = envs.SGLANG_DIFFUSION_PLATFORM.get().strip()

    if selected:
        # Front-loading filter: only the selected plugin is imported, so the
        # other vendors' modules never pull their hardware dependencies.
        ep_map = {
            ep.name: ep for ep in entry_points(group=DIFFUSION_PLATFORM_PLUGINS_GROUP)
        }
        if selected not in ep_map:
            available = ", ".join(f"'{name}'" for name in ep_map) if ep_map else "none"
            raise RuntimeError(
                f"SGLANG_DIFFUSION_PLATFORM={selected!r} not found in discovered "
                f"diffusion platform plugins (available: {available}). Install the "
                f"plugin with 'pip install -e' to register its entry_points."
            )
        try:
            result = ep_map[selected].load()()
        except Exception as e:
            logger.exception(
                "Failed to activate diffusion platform plugin: %s", selected
            )
            raise RuntimeError(
                f"Diffusion platform plugin {selected!r} failed to activate: {e}"
            ) from e
        if result is None:
            raise RuntimeError(
                f"Diffusion platform plugin {selected!r} is installed but activate() "
                f"returned None (hardware not available on this machine?)."
            )
        logger.info("OOT diffusion platform activated: %s -> %s", selected, result)
        return result

    # Auto-discover: activate every plugin and expect at most one to claim the
    # machine. An activation error only disables that plugin.
    activated: dict[str, str] = {}
    for name, (activate, _dist) in discover_diffusion_plugins(
        DIFFUSION_PLATFORM_PLUGINS_GROUP
    ).items():
        try:
            result = activate()
        except Exception:
            logger.exception("Failed to activate diffusion platform plugin: %s", name)
            continue
        if result is not None:
            activated[name] = result
            logger.info("OOT diffusion platform activated: %s -> %s", name, result)

    if not activated:
        return None
    if len(activated) == 1:
        return next(iter(activated.values()))

    names_str = ", ".join(f"'{name}'" for name in activated)
    raise RuntimeError(
        f"Multiple diffusion platform plugins activated: {names_str}. "
        f"Set SGLANG_DIFFUSION_PLATFORM to select one."
    )


def _load_platform_class(qualname: str) -> type[Platform]:
    """Load a Platform subclass from ``module.Class`` or ``module:Class``."""
    cls = pkgutil.resolve_name(qualname)
    if not isinstance(cls, type) or not issubclass(cls, Platform):
        raise TypeError(f"Expected a Platform subclass, got {type(cls)}: {qualname}")
    return cls


_current_platform: Platform | None = None
_init_trace: str = ""

current_platform: Platform


def __getattr__(name: str):
    if name == "current_platform":
        # lazy init current_platform.
        # 1. out-of-tree platform plugins need `from sglang.multimodal_gen.runtime.platforms import
        #    Platform` so that they can inherit `Platform` class. Therefore,
        #    we cannot resolve `current_platform` during the import of
        #    `sglang.multimodal_gen.runtime.platforms`.
        global _current_platform
        if _current_platform is None:
            platform_cls_qualname = resolve_current_platform_cls_qualname()
            _current_platform = _load_platform_class(platform_cls_qualname)()
            global _init_trace
            _init_trace = "".join(traceback.format_stack())
        return _current_platform
    elif name in globals():
        return globals()[name]
    else:
        raise AttributeError(f"No attribute named '{name}' exists in {__name__}.")


__all__ = ["Platform", "PlatformEnum", "current_platform", "_init_trace"]
